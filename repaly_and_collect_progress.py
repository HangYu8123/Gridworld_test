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
from minigrid.core.actions import Actions
import time

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

def read_actions_from_file(file_path: str) -> list[str]:

    observations = []
    actions = []

    # Read the file and process the data
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('Actions.'):
                actions.append(line.split('.')[1])
            else:
                observations.append(list(map(float, line.split())))
    return actions, observations

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
    env = FlatObsWrapper(env)
    env.reset()
    progress = []
    delta_progress = [] 
    rewards = []
    file_name = "near_optimal_pose"
    actions, obss = read_actions_from_file(file_name + ".txt")
    env.render()
    time.sleep(3)
    for action in actions:
        print(action, key_to_action[action])
        obs,reward, done, _, _ = env.step(action=key_to_action[action])
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
        with open(file = file_name + "annotated" + ".txt", mode= "w") as f:
            for i in range (len(progress)):
                for i in obss[i]:
                    f.write(str(i) + " ")
                f.write("\n")
                f.write(str(actions[i]) + "\n")
                f.write(str(rewards[i]) + "\n")
                f.write(str(progress[i]) + "\n")
                #f.write(str(delta_progress[i]) + "\n")
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