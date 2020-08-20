import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import pdb

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class FCNetwork(nn.Module):
    def __init__(self, action_size,
                       state_size,
                       output_size,
                       hidden_layers = 3,
                       hidden_size = 64,
                       dropout = 0,
                       state_mean = None,
                       state_std = None,
                       action_mean = None,
                       action_std = None):
        super().__init__()
        self.action_linear = nn.Sequential(nn.Linear(action_size, hidden_size), nn.ReLU())
        self.action_hidden = nn.Sequential(*[nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, hidden_size), nn.ReLU())])
        
        self.state_linear = nn.Sequential(nn.Linear(state_size, hidden_size), nn.ReLU())
        self.state_hidden = nn.Sequential(*[nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, hidden_size), nn.ReLU())])
        
        self.hidden = nn.Sequential(*[nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size*2, hidden_size*2), nn.ReLU())])
        self.out = nn.Linear(hidden_size*2, output_size)

        if state_mean == None or state_std == None:
            state_mean = torch.zeros(state_size)
            state_std = torch.ones(state_size)
        self.state_mean = nn.Parameter(state_mean)
        self.state_mean.requires_grad = False
        self.state_std = nn.Parameter(state_std)
        self.state_std.requires_grad = False

        if action_mean == None or action_std == None:
            action_mean = torch.zeros(action_size)
            action_std = torch.ones(action_size)
        self.action_mean = nn.Parameter(action_mean)
        self.action_mean.requires_grad = False
        self.action_std = nn.Parameter(action_std)
        self.action_std.requires_grad = False
    
    def forward(self, a, s):
        a = (a - self.action_mean) / self.action_std
        a = self.action_linear(a)
        a = self.action_hidden(a)
        
        s = (s - self.state_mean) / self.state_std
        s = self.state_linear(s)
        s = self.state_hidden(s)
        
        x = torch.cat([a, s], dim=1)
        x = self.hidden(x)
        x = self.out(x)
        return x
