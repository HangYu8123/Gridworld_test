import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import ExpertTraj

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        
        self.max_action = max_action
        
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x)) * self.max_action
        x = torch.round(x)  # Round the actions to the nearest integer
        return x.int()  # Convert the actions to integers


class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()
        
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)
    
    def forward(self, state, action):
        # print("state shape: ", state.shape)
        # print("action shape: ", action.shape)

        if action.shape != tuple([10,1]):
            action = action.unsqueeze(1)
        state_action = torch.cat([state, action], 1)
        x = torch.tanh(self.l1(state_action))
        x = torch.tanh(self.l2(x))
        x = torch.sigmoid(self.l3(x))
        return x
    
    
class GAIL:
    def __init__(self, file_path, state_dim, action_dim, max_action, lr, betas):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr, betas=betas)
        
        self.discriminator = Discriminator(state_dim, action_dim).to(device)
        self.optim_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        
        self.max_action = max_action
        self.expert = ExpertTraj(file_path, reward=True, progress=True)
        
        self.loss_fn = nn.BCELoss()
    
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten().astype(int)  # Convert to integers
        
    def update(self, n_iter, batch_size=100):
        mini_lose = []
        for i in range(n_iter):
            # sample expert transitions
            exp_state, exp_action, _, _, _ = self.expert.sample(batch_size)
            exp_state = torch.FloatTensor(exp_state).to(device)
            exp_action = torch.FloatTensor(exp_action).to(device)
            #exp_action = exp_action.unsqueeze(1)
            
            # sample expert states for actor
            state, _, _, _, _ = self.expert.sample(batch_size)
            state = torch.FloatTensor(state).to(device)
            action = self.actor(state)
            # print("action: ", action)
            # print("exp_action: ", exp_action)
            #######################
            # update discriminator
            #######################
            self.optim_discriminator.zero_grad()
            
            # label tensors
            exp_label = torch.full((batch_size, 1), 1, device=device).float()
            policy_label = torch.full((batch_size, 1), 0, device=device).float()
            
            # with expert transitions
            prob_exp = self.discriminator(exp_state, exp_action)
            #print("prob_exp: ", prob_exp)

            loss = self.loss_fn(prob_exp, exp_label)
            # print("########################################################### ")
            # print("loss: ", loss)
            # with policy transitions
            prob_policy = self.discriminator(state, action.detach())
            #print("prob_policy: ", prob_policy)
            loss += self.loss_fn(prob_policy, policy_label)
            # print("loss: ", loss)
            # print("########################################################### ")

            mini_lose.append(loss)
            # take gradient step
            loss.backward()
            self.optim_discriminator.step()
            
            ################
            # update policy
            ################
            self.optim_actor.zero_grad()
            
            loss_actor = -self.discriminator(state, action)
            loss_actor.mean().backward()
            self.optim_actor.step()
        print("loss: ", sum(mini_lose)/len(mini_lose))
            
    def save(self, directory='', name='GAIL'):
        torch.save(self.actor.state_dict(), '{}_actor.pth'.format(directory, name))
        torch.save(self.discriminator.state_dict(), '{}_discriminator.pth'.format(directory, name))
        
    def load(self, directory='', name='GAIL'):
        self.actor.load_state_dict(torch.load('{}_actor.pth'.format(directory, name)))
        self.discriminator.load_state_dict(torch.load('{}_discriminator.pth'.format(directory, name)))
