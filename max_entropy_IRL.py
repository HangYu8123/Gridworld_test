import scipy.optimize
import numpy as np

def max_ent_irl(n_states, n_actions, transition_probability, feature_matrix, discount, Rmax, trajectories, learning_rate, n_iters):
    """
    Maximum Entropy Inverse Reinforcement Learning (MaxEnt IRL)
    
    Args:
        n_states (int): number of states
        n_actions (int): number of actions
        transition_probability (numpy.ndarray): transition_probability[s', s, a] is the transition probability from state s to s' under action a
        feature_matrix (numpy.ndarray): feature_matrix[s] is the feature of state s
        discount (float): discount factor
        Rmax (float): maximum possible value of recoverred reward
        trajectories (list): a list of expert trajectories
        learning_rate (float): learning rate
        n_iters (int): number of iterations
    
    Returns:
        numpy.ndarray: recoverred reward
    """
    def __init__(self, env, trajectories, learning_rate=0.1, n_iters=100):
        self.env = env
        self.trajectories = trajectories
        self.learning_rate = learning_rate
        self.n_iters = n_iters