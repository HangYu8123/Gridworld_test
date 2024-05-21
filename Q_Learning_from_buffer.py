
from __future__ import annotations
import torch
import gym
import numpy as np
from GAIL import GAIL
import matplotlib.pyplot as plt
import utils
import Q_Learning_agnet
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
} 

def convert_to_array( data):
# Initialize an empty array of shape (10, 10, 1)
    shape = data.shape
    array = np.zeros((shape[0], shape[1]))

    # Fill the array with the values from the data
    for row in data:
        for col in row:
            i, j, value = col
            array[i, j] = value
    return array.flatten()
def convert_to_pose(data):
# Initialize an empty array of shape (10, 10, 1)
    shape = data.shape
    array = np.zeros((shape[0], shape[1]))

    # Fill the array with the values from the data
    for row in data:
        for col in row:
            i, j, value = col
            if value == 10:
                return [i, j]

    return None
def train():
    ######### Hyperparameters #########
    file_path = "near_optimal_poseannotated.txt"
    #env_name = "LunarLanderContinuous-v2"
    solved_reward = 0.2        # stop training if solved_reward > avg_reward
    random_seed = 0
    max_timesteps = 200        # max time steps in one episode
    n_eval_episodes = 200        # evaluate average reward over n episodes
    lr = 0.0002                 # learing rate
    betas = (0.5, 0.999)        # betas for adam optimizer
    n_epochs = 10000              # number of epochs
    n_iter = 100                # updates per epoch
    batch_size = 1            # num of transitions sampled from expert
    ###################################

    env = SimpleEnv()
    #env = FlatObsWrapper(env)
    # enable manual control for testing
    env = SymbolicObsWrapper(env)
    env = NoDeath(env, no_death_types=("lava",), death_cost=-0.4)


    state_dim = 3
    action_dim = 1
    max_action = 6

    epochs = []
    rewards = []
    env.reset()

    agent = Q_Learning_agnet.QLAgent(action_space= 7)
    expert_traj = utils.ExpertTraj(file_path, online=False)

    for epoch in range(n_epochs):
        state, next_state, action, reward, progress = expert_traj.sample(batch_size, return_next_state=True)
        state = state[0]
        next_state = next_state[0]
        #convert np.list back to list
        state = state.tolist()
        next_state = next_state.tolist()
       
        for i in range(3):
            state[i] = int(state[i])
            next_state[i] = int(next_state[i])
        #print("state", state)
        agent.learning(action[0][0], list(state), list(next_state), progress[0])
        #print("state", state)
    agent.epsilon = 0
    for i in range(100):
        state = env.reset()
        state, reward, done, _, _ = env.step(6)
        state = convert_to_pose(state['image']) + [state['direction']]
        done = False
        total_reward = 0
        cnt = 0
        while not done and cnt < 200:
            cnt += 1
            action = agent.choose_action(state)
            #print("state", state)   
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            state = next_state
            state = convert_to_pose(state['image']) + [state['direction']]
        print("total_reward", total_reward)
        rewards.append(total_reward)
        epochs.append(i)
    plt.plot(epochs, rewards)
    plt.show()
    print("Training done")
if __name__ == "__main__":
    train()