#!/usr/bin/env python3

from __future__ import annotations

import gymnasium as gym
import pygame
from gymnasium import Env
from PIL import Image
from minigrid.core.actions import Actions
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
import matplotlib.pyplot as plt
import numpy as np

class ManualControl:
    def __init__(
        self,
        env: Env,
        seed=None,
        demo_file_name="demo.txt",
    ) -> None:
        self.env = env
        self.seed = seed
        self.closed = False
        self.demo_file_name = demo_file_name
        with open(demo_file_name, "w") as f:
            pass
        f.close()
        

    def start(self):
        """Start the window display with blocking event loop"""
        self.reset(self.seed)

        while not self.closed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.env.close()
                    break
                if event.type == pygame.KEYDOWN:
                    event.key = pygame.key.name(int(event.key))
                    self.key_handler(event)
    def convert_to_array(self, data):
    # Initialize an empty array of shape (10, 10, 1)
        shape = data.shape
        array = np.zeros((shape[0], shape[1]))

        # Fill the array with the values from the data
        for row in data:
            for col in row:
                i, j, value = col
                array[i, j] = value

        return array
    def step(self, action: Actions):
        obs, reward, terminated, truncated, _ = self.env.step(action)
        with open(self.demo_file_name, "a") as f:
            #print(obs.shape)     
            # img = Image.fromarray(obs['image'])
            # img = img.convert('L')
            # print(img.size)
            print(obs['image'].shape)
            print(obs['image'])
            print(self.convert_to_array(obs['image']).flatten())
            obs_array = self.convert_to_array(obs['image']).flatten()
            for num in obs_array:
                f.write(str(num) + " ")
            # f.write(str(list(img.getdata())) + "\n")
            f.write("\n")
            f.write(str(action) + "\n")
        f.close()

        print(f"step={self.env.step_count}, reward={reward:.2f}")

        if terminated:
            print("terminated!")
            self.reset(self.seed)
        elif truncated:
            print("truncated!")
            self.reset(self.seed)
        else:
            self.env.render()

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        self.env.render()

    def key_handler(self, event):
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            self.env.close()
            return
        if key == "backspace":
            self.reset()
            return

        key_to_action = {
            "left": Actions.left,
            "right": Actions.right,
            "up": Actions.forward,
            "space": Actions.toggle,
            "pageup": Actions.pickup,
            "pagedown": Actions.drop,
            "tab": Actions.pickup,
            "left shift": Actions.drop,
            "enter": Actions.done,
        }
        if key in key_to_action.keys():
            action = key_to_action[key]
            self.step(action)
        else:
            print(key)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        type=str,
        help="gym environment to load",
        choices=gym.envs.registry.keys(),
        default="MiniGrid-MultiRoom-N6-v0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=None,
    )
    parser.add_argument(
        "--tile-size", type=int, help="size at which to render tiles", default=32
    )
    parser.add_argument(
        "--agent-view",
        action="store_true",
        help="draw the agent sees (partially observable view)",
    )
    parser.add_argument(
        "--agent-view-size",
        type=int,
        default=7,
        help="set the number of grid spaces visible in agent-view ",
    )
    parser.add_argument(
        "--screen-size",
        type=int,
        default="640",
        help="set the resolution for pygame rendering (width and height)",
    )

    args = parser.parse_args()

    env: MiniGridEnv = gym.make(
        args.env_id,
        tile_size=args.tile_size,
        render_mode="human",
        agent_pov=args.agent_view,
        agent_view_size=args.agent_view_size,
        screen_size=args.screen_size,
    )

    # TODO: check if this can be removed
    if args.agent_view:
        print("Using agent view")
        env = RGBImgPartialObsWrapper(env, args.tile_size)
        env = ImgObsWrapper(env)

    manual_control = ManualControl(env, seed=args.seed)
    manual_control.start()
