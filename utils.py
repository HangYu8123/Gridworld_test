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
    def __init__(self, file_path: str, online=True):
        self.online = online
        if online:
            self.exp_states, self.exp_actions = self.read_actions_from_file(file_path)
            self.n_transitions = len(self.exp_actions)
        else:
            self.exp_states, self.exp_actions, self.exp_rewards, self.exp_progresses = self.read_actions_from_file(file_path, online=online)
            self.n_transitions = len(self.exp_actions)
    
    def read_actions_from_file(self, file_path: str, online=True):

        observations = []
        actions = []
        rewards = []
        progresses =[]
        if online:
            # Read the file and process the data
            with open(file_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if line.startswith('Actions.'):
                        actions.append([key_map[line.split('.')[1]]])
                    else:
                        observations.append(list(map(float, line.split())))
            return observations, actions 
        else:
            # Read the file and process the data
            with open(file_path, 'r') as file:
                cnt = 0
                for line in file:
                    line = line.strip()
                    if cnt == 0:
                        if line.startswith('['): 
                            # cut the brackets
                            line = line[1:-1]
                            observations.append(list(map(float, line.split(","))))
                        else:
                            observations.append(list(map(float, line.split())))
                    if cnt == 1:
                        actions.append([key_map[line]])
                    if cnt == 2:
                        rewards.append(float(line))
                    if cnt == 3:
                        progresses.append(float(line))
                    cnt += 1
                    if cnt == 4:
                        cnt = 0
                        
            return observations, actions, rewards, progresses
    
    def sample(self, batch_size, online=True, return_next_state=False):
        online = self.online
        if online:
            indexes = np.random.randint(0, self.n_transitions, size=batch_size)
            state, action = [], []
            for i in indexes:
                s = self.exp_states[i]
                a = self.exp_actions[i]
                state.append(np.array(s, copy=False))
                action.append(np.array(a, copy=False))
            return np.array(state), np.array(action)
        else:
            indexes = np.random.randint(0, self.n_transitions, size=batch_size)
            state, next_state, action, reward, progress = [], [], [], [], []
            for i in indexes:
                s = self.exp_states[i]
                ns = self.exp_states[min(i+1, self.n_transitions-1)]
                a = self.exp_actions[i]
                r = self.exp_rewards[i]
                p = self.exp_progresses[i]
                state.append(np.array(s, copy=False))
                next_state.append(np.array(ns, copy=False))
                action.append(np.array(a, copy=False))
                reward.append(np.array(r, copy=False))
                progress.append(np.array(p, copy=False))
            if return_next_state:
                return np.array(state), np.array(next_state), np.array(action), np.array(reward), np.array(progress)
            return np.array(state), np.array(action), np.array(reward), np.array(progress)


# demo = ExpertTraj("optimal.txt")
# obs, _  = demo.sample(1)
# print(len(obs[0]))