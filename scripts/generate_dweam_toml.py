import re
from diamond_atari.game.keymap import get_keymap_and_action_names
from diamond_atari.utils import ATARI_100K_GAMES
from dweam.models import GameInfo, PackageMetadata
import pygame

GAME_DESCRIPTIONS = {
    "Alien": "Maze-based game inspired by the film 'Alien'",
    "Amidar": "Maze-based arcade game",
    "Assault": "Tank combat game",
    "Asterix": "Side-scrolling game based on the popular comic",
    "BankHeist": "Players act as getaway drivers, robbing banks and avoiding police",
    "BattleZone": "Revolutionary first-person tank combat game",
    "Boxing": "Overhead boxing game",
    "Breakout": "Iconic block-breaking game",
    "ChopperCommand": "Pilot a helicopter in a high-paced shooter game",
    "CrazyClimber": "Vertical climbing game",
    "DemonAttack": "Shooter featuring waves of alien invaders",
    "Freeway": "Play a chicken trying to cross the road",
    "Frostbite": "Build igloos by jumping on floating ice blocks",
    "Gopher": "Protect crops from gophers by filling holes and knocking them back",
    "Hero": "Action game where players rescue miners",
    "Jamesbond": "Play as James Bond in this action-adventure game",
    "Kangaroo": "Platformer featuring a kangaroo that must rescue its baby while dodging monkeys and collecting fruit.",
    "Krull": "Game based on the film 'Krull,' featuring a mix of combat, exploration, and puzzle-solving sequences.",
    "KungFuMaster": "Players control a martial artist battling waves of enemies across multiple levels in this fast-paced beat-'em-up game.",
    "MsPacman": "Sequel to Pac-Man, featuring a new character, varied mazes, and improved gameplay elements as players evade ghosts and eat pellets.",
    "Pong": "Classic table tennis simulation game that laid the foundation for the video game industry, challenging players to outscore their opponent.",
    "PrivateEye": "Detective adventure where players solve crimes and collect evidence while driving through city streets.",
    "Qbert": "Puzzle-platform game where players hop between cubes to change their colors, while avoiding enemies and traps in an isometric world.",
    "RoadRunner": "Based on the cartoon, players control the Road Runner, collecting birdseed and evading Wile E. Coyote across various environments.",
    "Seaquest": "Underwater-themed shooter where players control a submarine, rescuing divers and defeating enemy submarines and sharks.",
    "UpNDown": "Unique racing and jumping game where players navigate a car along twisting tracks, collecting flags and avoiding collisions.",
}


if not all(id_ in GAME_DESCRIPTIONS for id_ in ATARI_100K_GAMES):
    print("The difference between the two dictionaries is: ")
    print("Games in GAME_DESCRIPTIONS but not in ATARI_100K_GAMES: ", set(GAME_DESCRIPTIONS.keys()) - set(ATARI_100K_GAMES))
    print("Games in ATARI_100K_GAMES but not in GAME_DESCRIPTIONS: ", set(ATARI_100K_GAMES) - set(GAME_DESCRIPTIONS.keys()))
    raise ValueError("Not all games have descriptions")


def generate_game_infos() -> PackageMetadata:
    infos = {}
    for id_ in ATARI_100K_GAMES:
        # Get game-specific keymap and action names
        keymap, action_names = get_keymap_and_action_names('atari/' + id_)

        base_buttons = {
            '🔄 Restart': 'Enter',
        }
        
        # Update buttons based on game-specific controls
        allowed_actions = {
            'up': '⬆️ Up',
            'left': '⬅️ Left',
            'down': '⬇️ Down',
            'right': '➡️ Right',
            'fire': '🔥 Fire',
        }
        buttons = {
            name: '+'.join(pygame.key.name(k) for k in keys).capitalize()
            for action_name, name in allowed_actions.items()
            for keys, action in keymap.items()
            if action_names[action] == action_name
        }

        # CapitalCase => Capital Case
        title = ' '.join(re.findall('[A-Z][^A-Z]*', id_))

        infos[id_] = GameInfo(
            title=title,
            tags=["atari"],
            buttons=buttons | base_buttons,
            description=GAME_DESCRIPTIONS[id_],
        )
    return PackageMetadata(
        type="Diamond Atari",
        entrypoint="diamond_atari.dweam_game:DiamondGame",
        repo_link="https://github.com/eloialonso/diamond",
        games=infos,
    )


if __name__ == "__main__":
    import tomli_w
    with open("diamond_atari/dweam.toml", "wb") as f:
        tomli_w.dump(generate_game_infos().model_dump(exclude_none=True), f)
