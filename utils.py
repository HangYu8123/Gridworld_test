import numpy as np

key_map = { 
    "left": 0,
    "right": 1,
    "forward": 2,
    "pickup": 3,
    "drop": 4,
    "toggle": 5,
    "done": 6
}
    


class ExpertTraj:
    def __init__(self, file_path: str, reward = False, progress = False):
        self.states, self.actions, self.next_states, self.rewards, self.progresses = self.read_actions_from_file(file_path, reward, progress)
        self.n_transitions = len(self.actions)

    
    def read_actions_from_file(self, file_path: str, reward = False, progress = False):

        states = []
        next_states = []
        actions = []
        rewards = []
        progresses =[]
        count = 3
        if reward:
            count = 4
        if progress:
            count = 4
        if reward and progress:
            count = 5
        with open(file_path, "r") as f:
            cnt = 0
            for line in f:
                
                if cnt == 0:
                    states.append(list(map(int, line.strip().split())))
                elif cnt == 1:
                    actions.append(key_map[line.strip().split('.')[1]])
                elif cnt == 2:
                    next_states.append(list(map(int, line.strip().split())))
                if cnt == 3 and count == 4:
                    if reward:
                        rewards.append(float(line.strip()))
                    if progress:
                        progresses.append(float(line.strip()))
                
                if count == 5:
                    if cnt == 3:
                        rewards.append(float(line.strip()))
                    if cnt == 4:
                        progresses.append(float(line.strip()))
                    


                cnt += 1
                if cnt == count:
                    cnt = 0
                
        return states, actions, next_states, rewards, progresses

    
    def sample(self, batch_size, reward = False, progress = False):

        indexes = np.random.randint(0, self.n_transitions, size=batch_size)
        states, actions, next_states, rewards, progresses = [], [], [], [], []
        for i in indexes:
            states.append(self.states[i])
            actions.append(self.actions[i])
            next_states.append(self.next_states[i])
            if reward:
                rewards.append(self.rewards[i])
            if progress:
                progresses.append(self.progresses[i])
        if batch_size == 1:
            return states[0], actions[0], next_states[0], rewards[0], progresses[0]
        return states, actions, next_states, rewards, progresses


# demo = ExpertTraj("optimal.txt")
# obs, _  = demo.sample(1)
# print(len(obs[0]))