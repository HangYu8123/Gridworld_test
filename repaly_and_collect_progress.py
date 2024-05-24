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
import time

key_to_action = {
    "forward": Actions.forward,
    "left": Actions.left,
    "right": Actions.right,
    "toggle": Actions.toggle,
    "pickup": Actions.pickup,
    "drop": Actions.drop,
    "done": Actions.done,
}

def read_actions_from_file( file_path: str, reward = False, progress = False):

    states = []
    next_states = []
    actions = []
    with open(file_path, "r") as f:
        cnt = 0
        for line in f:
            
            if cnt == 0:
                states.append(list(map(int, line.strip().split())))
            elif cnt == 1:
                actions.append(key_to_action[line.strip().split('.')[1]])
            elif cnt == 2:
                next_states.append(list(map(int, line.strip().split())))
            cnt+=1
            if cnt == 3:
                cnt = 0

            
    return states, actions, next_states

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
        self.step_count = 0
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

global lava_cnt

def if_lava(state):
    if state[0] == 2 and state[1] == 6:
        return True
    if state[0] == 2 and state[1] == 5:
        return True
    if state[0] == 3 and state[1] == 1:
        return True
    if state[0] == 4 and state[1] == 1:
        return True
    if state[0] == 3 and state[1] == 4:
        return True
    if state[0] == 7 and state[1] == 7:
        return True 

def distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def give_progress(state, next_state):
    global lava_cnt
    goal = [8, 8]
    progress = 0
    if next_state == goal:
        return 100 - 10 * lava_cnt
    if next_state[3] == 1:
        progress = 20
    if next_state[4] == 1:
        progress = 40
    dis_progress =  (14 - distance(next_state, goal)) *(60/14)
    progress += dis_progress
    return progress - 10 * lava_cnt
        


def main():
    global lava_cnt 
    lava_cnt = 0
    env = SimpleEnv(render_mode="human")
    env = FlatObsWrapper(env)
    env = NoDeath(env, no_death_types=("lava",), death_cost=-0.4)
    env.reset()
    progress = []
    delta_progress = [] 
    rewards = []
    file_name = "near_optimal_pose"
    states, actions, next_states = read_actions_from_file(file_name + ".txt")

    auto = True
    env.render()
    time.sleep(3)
    if auto:
        for action in actions:
            obs,reward, done, _, _ = env.step(action)
            env.render()
            #time.sleep(0.5) 
            if reward == -0.4:
                lava_cnt += 1
            progress.append(give_progress(states[len(progress)], next_states[len(progress)]))
            rewards.append(reward)
            with open(file = file_name + "_auto_annotated" + ".txt", mode= "w") as f:
                for i in range(len(progress)):
                    for pos in states[i]:
                        f.write(str(pos) + " ")
                    f.write("\n")
                    f.write(str(actions[i]) + "\n")
                    for pos in next_states[i]:
                        f.write(str(pos) + " ")
                    f.write("\n")
                    f.write(str(rewards[i]) + "\n")
                    f.write(str(progress[i]) + "\n")
                f.close()

            if done:
                lava_cnt = 0
                env.reset()
            
    else:
        for action in actions:
            print(action)
            obs,reward, done, _, _ = env.step(action)
            env.render()
            time.sleep(0.5) 
            # get progress via keyboard input
            inp = (input("Enter progress: "))
            if inp == '':
                inp = 0
            inp = int(inp)
            
            progress.append(int(inp))
            #delta_progress.append(progress[-1] - progress[len(progress) - 2])
            rewards.append(reward)
            with open(file = file_name + "_annotated" + ".txt", mode= "w") as f:
                for i in range(len(progress)):
                    for pos in states[i]:
                        f.write(str(pos) + " ")
                    f.write("\n")
                    f.write(str(actions[i]) + "\n")
                    for pos in next_states[i]:
                        f.write(str(pos) + " ")
                    f.write("\n")
                    f.write(str(rewards[i]) + "\n")
                    f.write(str(progress[i]) + "\n")
                f.close()

            if done:
                env.reset()
            
        print(progress)
        print(delta_progress)


    # print(actions)
    # print(obss[0])
    # print(len(obss[0]))
if __name__ == "__main__":
    main()