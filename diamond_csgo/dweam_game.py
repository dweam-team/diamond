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
from huggingface_hub import snapshot_download as hf_snapshot_download

from .agent import Agent
from .game.play_env import PlayEnv
from .data import collate_segments_to_batch
from .data.dataset import Dataset
from .data.batch_sampler import BatchSampler
from .envs.world_model_env import WorldModelEnv
from .csgo.action_processing import CSGOAction
from .csgo.keymap import CSGO_KEYMAP

from dweam import Game, GameInfo, get_cache_dir


OmegaConf.register_new_resolver("eval", eval, replace=True)


def snapshot_download(**kwargs) -> Path:
    base_cache_dir = get_cache_dir()
    cache_dir = base_cache_dir / 'huggingface-data'
    path = hf_snapshot_download(cache_dir=str(cache_dir), **kwargs)
    return Path(path)


def prepare_env(cfg: DictConfig, device: torch.device) -> tuple[PlayEnv, dict]:
    path_hf = Path(snapshot_download(repo_id="eloialonso/diamond", allow_patterns="csgo/*"))
    
    path_ckpt = path_hf / "csgo/model/csgo.pt"
    spawn_dir = path_hf / "csgo/spawn"

    # Override config
    # cfg.agent = OmegaConf.load(path_hf / "csgo/config/agent/csgo.yaml")
    # cfg.env = OmegaConf.load(path_hf / "csgo/config/env/csgo.yaml")

    assert cfg.env.train.id == "csgo"
    num_actions = cfg.env.num_actions

    # Models
    agent = Agent(instantiate(cfg.agent, num_actions=num_actions)).to(device).eval()
    agent.load(path_ckpt)
    
    # World model environment
    sl = cfg.agent.denoiser.inner_model.num_steps_conditioning
    if agent.upsampler is not None:
        sl = max(sl, cfg.agent.upsampler.inner_model.num_steps_conditioning)
    wm_env_cfg = instantiate(cfg.world_model_env, num_batches_to_preload=1)
    wm_env = WorldModelEnv(
        agent.denoiser, 
        agent.upsampler,
        agent.rew_end_model,
        spawn_dir,
        1,
        sl,
        wm_env_cfg,
        return_denoising_trajectory=True
    )

    play_env = PlayEnv(
        agent,
        wm_env,
        recording_mode=False,
        store_denoising_trajectory=False,
        store_original_obs=False,
    )

    return play_env, CSGO_KEYMAP


class CSGOGame(Game):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        with initialize(version_base="1.3", config_path="config"):
            cfg = compose(config_name="trainer")

        device_id = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.log.info(f"Using device", device_id=device_id)
        self.device = torch.device(device_id)

        self.env, self.keymap = prepare_env(
            cfg=cfg,
            device=self.device,
        )
        self.env.reset()

        # Set size
        self.height, self.width = (cfg.env.train.size,) * 2 if isinstance(cfg.env.train.size, int) else cfg.env.train.size

    def draw_game(self, obs) -> pygame.Surface:
        assert obs.ndim == 4 and obs.size(0) == 1
        img = Image.fromarray(obs[0].add(1).div(2).mul(255).byte().permute(1, 2, 0).cpu().numpy())
        pygame_image = np.array(img.resize((self.width, self.height), resample=Image.NEAREST)).transpose((1, 0, 2))
        return pygame.surfarray.make_surface(pygame_image)

    def step(self) -> pygame.Surface:
        # Create CSGOAction from current input state
        action = CSGOAction(
            keys=list(self.keys_pressed),
            mouse_x=self.mouse_motion[0],
            mouse_y=self.mouse_motion[1],
            l_click=1 in self.mouse_pressed,
            r_click=3 in self.mouse_pressed
        )

        # Step the environment with CSGO action
        next_obs, rew, end, trunc, info = self.env.step(action)

        # Reset if episode ended
        if end or trunc:
            self.env.reset()

        return self.draw_game(next_obs)

    def on_key_down(self, key: int) -> None:
        # Handle special keys
        if key == pygame.K_UP:
            self.env.next_axis_1()  # Increase imagination horizon
        elif key == pygame.K_DOWN:
            self.env.prev_axis_1()  # Decrease imagination horizon
        elif key == pygame.K_RIGHT:
            self.env.next_axis_2()  # Next environment
        elif key == pygame.K_LEFT:
            self.env.prev_axis_2()  # Previous environment
        elif key == pygame.K_RETURN:
            self.env.reset()
        elif key == pygame.K_PERIOD:
            self.paused = not self.paused
        elif key == pygame.K_e and self.paused:
            self.do_one_step()

    def stop(self) -> None:
        super().stop()
        # TODO deload the model from GPU memory
