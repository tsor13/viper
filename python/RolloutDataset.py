import numpy as np
import torch
from torch.utils.data import Dataset

class RolloutDataset(Dataset):
    def __init__(self, rollouts, n_ahead = 1):
        super().__init__()
        # split into actions and states
        self.actions = rollouts[:, :, :3]
        self.states = rollouts[:, :, 3:]
        self.n_ahead = n_ahead
        
        # n is number of rollouts, t is time steps
        n, t, _ = self.actions.shape
        self.n = n
        self.t = t
    
    def __getitem__(self, index):
        n = index // (self.t-self.n_ahead)
        t = index % (self.t-self.n_ahead)
        actions = self.actions[n, t:t+self.n_ahead]
        state_t = self.states[n, t]
        state_t1 = self.states[n, t+1:t+self.n_ahead+1]
        delta = state_t1 - state_t
        return actions, state_t, delta
        
    def __len__(self):
        return self.n * (self.t-self.n_ahead)
