import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import json  # Import json for dictionary serialization

class QLAgent:
    def __init__(self, action_space, alpha=0.5, gamma=0.9, temp=1, epsilon=0, mini_epsilon=0, decay=0.9999):
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.temp = temp
        self.epsilon = epsilon
        self.mini_epsilon = mini_epsilon
        self.decay = decay
        self.qtable = pd.DataFrame(columns=[i for i in range(self.action_space)])
    
    def trans(self, state):
        #print("state", state)
        # if isinstance(state, np.int32):
        #     state = int(state)
        #     #print("error state", state )
        # elif isinstance(state, list):
        #     state = [int(s) if isinstance(s, np.int32) else s for s in state]
        #print("state in Q Learning###############################################", state)
        return str(state)


    
    def check_add(self, state):
        #print(goal_x, goal_y)
        #print("##########################state in check_add", state, "###############################################   ")
        serialized_state = self.trans(state)
        if serialized_state not in self.qtable.index:
            self.qtable.loc[serialized_state] = pd.Series(np.zeros(self.action_space), index=[i for i in range(self.action_space)])



    def learning(self,  state, action, next_state, reward):

        self.check_add(state)

        self.check_add(next_state)
        
        q_sa = self.qtable.loc[self.trans(state), action]
        max_next_q_sa = self.qtable.loc[self.trans(next_state), :].max()

        new_q_sa = q_sa + self.alpha * (reward + self.gamma * max_next_q_sa - q_sa)
        self.qtable.loc[self.trans(state), action] = new_q_sa


    def choose_action(self, state):
        self.check_add(state)
        p = np.random.uniform(0, 1)
        if self.epsilon >= self.mini_epsilon:
            self.epsilon *= self.decay
        if p <= self.epsilon:
            return np.random.choice([i for i in range(self.action_space)])
        else:
            #prob = F.softmax(torch.tensor(self.qtable_norms.loc[self.trans(state)].to_list()), dim=0).detach().numpy()
            #print(state)
            prob = F.softmax(torch.tensor(self.qtable.loc[self.trans(state)].to_list()), dim=0).detach().numpy()

            #print(prob)
            #action = np.random.choice([i for i in range(self.action_space)], p=prob)
            #chose the action with the highest probability, if there are multiple actions with the same probability, randomly choose one
            #action = np.random.choice([i for i in range(self.action_space)], p=prob)
            action = np.random.choice(np.flatnonzero(prob == np.max(prob)))
            return action
