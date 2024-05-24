from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import RGBImgObsWrapper
from minigrid.wrappers import OneHotPartialObsWrapper
from minigrid.wrappers import FullyObsWrapper
from minigrid.wrappers import ImgObsWrapper
from minigrid.wrappers import FlatObsWrapper
import numpy as np
from minigrid.core.actions import Actions
import gymnasium as gym
import pygame
from minigrid.wrappers import SymbolicObsWrapper
from minigrid.wrappers import NoDeath
from minigrid.wrappers import ViewSizeWrapper
class SimpleEnv(MiniGridEnv):
    def __init__(
        self,
        size=10,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "grand mission"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate vertical separation wall
        for i in range(0, height):
            self.grid.set(5, i, Wall())
        
        # Place the door and key
        self.grid.set(5, 6, Door(COLOR_NAMES[0], is_locked=True))
        self.grid.set(3, 6, Key(COLOR_NAMES[0]))
        self.grid.set(2, 6, Lava())
        self.grid.set(2, 5, Lava())
        self.grid.set(3, 1, Lava())
        self.grid.set(4, 1, Lava())
        self.grid.set(3, 4, Lava())
        self.grid.set(7, 7, Lava())

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "grand mission"

key_to_action = {
    "forward": Actions.forward,
    "left": Actions.left,
    "right": Actions.right,
    "toggle": Actions.toggle,
    "pickup": Actions.pickup,
    "drop": Actions.drop,
    "done": Actions.done,
}
def main():

    env = SimpleEnv(render_mode="human")
    #env = FlatObsWrapper(env)
    # enable manual control for testing
    env = SymbolicObsWrapper(env)
    env = NoDeath(env, no_death_types=("lava",), death_cost=-0.4)
    manual_control = ManualControl(env, seed=42, demo_file_name="test.txt")
    manual_control.start()

    
if __name__ == "__main__":
    main()























