from collections import OrderedDict
import os
from pathlib import Path
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import pygame
from hydra import compose, initialize
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from PIL import Image
from huggingface_hub import hf_hub_download
from enum import Enum

from .agent import Agent
from .game.play_env import NamedEnv, PlayEnv
from .game.keymap import get_keymap_and_action_names
from .data import collate_segments_to_batch
from .data.dataset import Dataset
from .data.batch_sampler import BatchSampler
from .coroutines.collector import NumToCollect, make_collector
from .envs.env import TorchEnv, make_atari_env
from .envs.world_model_env import WorldModelEnv

from dweam import Game, Field, get_cache_dir


OmegaConf.register_new_resolver("eval", eval, replace=True)


def download(filename: str) -> Path:
    base_cache_dir = get_cache_dir()
    cache_dir = base_cache_dir / 'huggingface-data'
    path = hf_hub_download(repo_id="eloialonso/diamond", filename=filename, cache_dir=str(cache_dir))
    return Path(path)


def prepare_env(cfg: DictConfig, name: str, device: torch.device, num_steps_initial_collect: int = 1000):
    # Checkpoint
    path_ckpt = download(f"atari_100k/models/{name}.pt")

    # Override config
    # cfg.agent = OmegaConf.load(download("atari_100k/config/agent/default.yaml"))
    # cfg.env = OmegaConf.load(download("atari_100k/config/env/atari.yaml"))
    cfg.env.train.id = cfg.env.test.id = f"{name}NoFrameskip-v4"
    cfg.world_model_env.horizon = 50

    # Real envs
    train_env = make_atari_env(num_envs=1, device=device, **cfg.env.train)
    test_env = make_atari_env(num_envs=1, device=device, **cfg.env.test)

    # Models
    agent = Agent(instantiate(cfg.agent, num_actions=test_env.num_actions)).to(device).eval()
    agent.load(path_ckpt)

    # Collect for imagination's initialization
    n = num_steps_initial_collect
    dataset = Dataset(Path(f"dataset/{path_ckpt.stem}_{n}"))
    dataset.load_from_default_path()
    if len(dataset) == 0:
        print(f"Collecting {n} steps in real environment for world model initialization.")
        collector = make_collector(test_env, agent.actor_critic, dataset, epsilon=0)
        collector.send(NumToCollect(steps=n))
        dataset.save_to_default_path()

    # World model environment
    bs = BatchSampler(dataset, 0, 1, 1, cfg.agent.denoiser.inner_model.num_steps_conditioning, None, False)
    dl = DataLoader(dataset, batch_sampler=bs, collate_fn=collate_segments_to_batch)
    wm_env_cfg = instantiate(cfg.world_model_env, num_batches_to_preload=1)
    wm_env = WorldModelEnv(agent.denoiser, agent.rew_end_model, dl, wm_env_cfg, return_denoising_trajectory=True)

    envs = [
        NamedEnv("wm", wm_env),
        NamedEnv("test", test_env),
        NamedEnv("train", train_env),
    ]

    env_keymap, env_action_names = get_keymap_and_action_names(cfg.env.keymap)
    play_env = PlayEnv(
        agent,
        envs,
        env_action_names,
        env_keymap,
        recording_mode=False,
        store_denoising_trajectory=False,
        store_original_obs=False,
    )

    torch_envs = [env for _, env in envs if isinstance(env, TorchEnv)]

    return play_env, env_keymap, torch_envs


class Environment(int, Enum):
    WORLD = 0
    TRAIN = 1
    TEST = 2

    @property
    def title(self) -> str:
        return self.name.capitalize()

class DiamondGame(Game):
    class Params(Game.Params):
        """Parameters for Diamond Atari games"""
        environment_id: Environment = Field(
            default=Environment.WORLD,
            description="Environment to use",
            json_schema_extra={
                "title": "Environment",
                "enumNames": [e.title for e in Environment]
            }
        )
        human_player: bool = Field(default=True)
        training_steps: int = Field(default=1000, description="How many steps to collect for world model initialization")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        with initialize(version_base="1.3", config_path="config"):
            self.cfg = cfg = compose(config_name="trainer")

        device_id = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.log.info(f"Using device", device_id=device_id)
        self.device = torch.device(device_id)

        self.env, self.keymap, self.torch_envs = self._initialize_env(cfg, self.game_id, self.device)

        # Set size based on environment configuration
        default_width = 640  # TODO should we make this configurable?
        self.width = self.height = (default_width // cfg.env.train.size) * cfg.env.train.size

    def _initialize_env(self, cfg: DictConfig, game_id: str, device: torch.device, num_steps_initial_collect: int = 1000) -> None:
        env, keymap, torch_envs = prepare_env(
            cfg=cfg,
            name=game_id,
            device=device,
            num_steps_initial_collect=num_steps_initial_collect,
        )
        env.switch_controller()
        env.reset()
        ordered_keymap = OrderedDict()
        for keys, act in sorted(keymap.items(), key=lambda keys_act: -len(keys_act[0])):
            ordered_keymap[keys] = act
        
        return env, ordered_keymap, torch_envs

    def on_params_update(self, new_params: Params) -> None:
        self.env.is_human_player = new_params.human_player

        # Update environment
        if self.params.environment_id != new_params.environment_id:
            env_id = new_params.environment_id.value  # Use the integer value directly
            self.env.env_id = env_id
            self.env.env_name, self.env.env = self.env.envs[env_id]
            self.env.reset()
            self.log.info("Updated environment", env_id=new_params.environment_id)
            
        if self.params.training_steps != new_params.training_steps:
            for env in self.torch_envs:
                env.close()
            self.env, self.keymap, self.torch_envs = self._initialize_env(
                cfg=self.cfg,
                game_id=self.game_id,
                device=self.device,
                num_steps_initial_collect=new_params.training_steps,
            )

        super().on_params_update(new_params)

    def stop(self):
        super().stop()
        for env in self.torch_envs:
            env.close()

    def draw_game(self, obs) -> pygame.Surface:
        assert obs.ndim == 4 and obs.size(0) == 1
        img = Image.fromarray(obs[0].add(1).div(2).mul(255).byte().permute(1, 2, 0).cpu().numpy())
        pygame_image = np.array(img.resize((self.width, self.height), resample=Image.NEAREST)).transpose((1, 0, 2))
        return pygame.surfarray.make_surface(pygame_image)

    def step(self) -> pygame.Surface:
        # Check pressed keys for actions
        for keys, action in self.keymap.items():
            if all(key in self.keys_pressed for key in keys):
                break
        else:
            action = 0

        # Step the environment
        next_obs, rew, end, trunc, info = self.env.step(action)

        # Reset if episode ended
        if end or trunc:
            self.env.reset()

        return self.draw_game(next_obs)

    def on_key_down(self, key: int) -> None:
        # Handle special keys
        # if key == pygame.K_m:
            # self.env.next_mode()  # Switch between policy/human control
        if key == pygame.K_UP:
            self.env.next_axis_1()  # Increase imagination horizon
        elif key == pygame.K_DOWN:
            self.env.prev_axis_1()  # Decrease imagination horizon
        elif key == pygame.K_RIGHT:
            self.env.next_axis_2()  # Next environment
            self.env.reset()
        elif key == pygame.K_LEFT:
            self.env.prev_axis_2()  # Previous environment
            self.env.reset()
        elif key == pygame.K_RETURN:
            self.env.reset()
        elif key == pygame.K_PERIOD:
            self.paused = not self.paused
        elif key == pygame.K_e and self.paused:
            self.do_one_step()
