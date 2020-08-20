import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import glob
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from tqdm import tqdm
import pdb
from RolloutDataset import RolloutDataset
from FCNetwork import FCNetwork
from hyperparams import params
device = 'cuda' if torch.cuda.is_available else 'cpu'

# read in rollout files
files = glob.glob('rollouts/*.npy')
# max number of rollouts (hits memory limit if too high)
n_rollouts = 500
files = files[:n_rollouts]
rollouts = []
# read in rollouts
for file in tqdm(files):
    rollout = np.load(file)
    rollouts.append(rollout)
rollouts = np.stack(rollouts)
print('Loaded')

# split into test and val
cutoff = int(round(.7*len(rollouts)))
train_rollouts = rollouts[:cutoff]
val_rollouts = rollouts[cutoff:]
del rollouts

# calculate mean and std for actions
mean = train_rollouts.mean(axis=(0, 1))
std = train_rollouts.std(axis=(0, 1)) + .1

mean = torch.Tensor(mean).to(device)
std = torch.Tensor(std).to(device)

action_mean = mean[:3]
action_std = std[:3]
state_mean = mean[3:]
state_std = std[3:]
del mean
del std

n_ahead = 1

train_dataset = RolloutDataset(train_rollouts, n_ahead = n_ahead)
val_dataset = RolloutDataset(val_rollouts, n_ahead = n_ahead)
action_size = len(train_dataset[0][0][0])
state_size = len(train_dataset[0][1])
output_size = len(train_dataset[0][2][0])
model_path = params['model path']
dropout = params['dropout']
hidden_layers = params['hidden layers']
hidden_size = params['hidden size']
print(action_size, state_size, output_size)

model = FCNetwork(action_size = action_size,
                  state_size = state_size,
                  output_size = output_size,
                  hidden_layers = hidden_layers,
                  hidden_size = hidden_size,
                  dropout = dropout)
                  # state_mean = state_mean,
                  # state_std = state_std,
                  # action_mean = action_mean,
                  # action_std = action_std)

model.to(device)
# model.load_state_dict(torch.load(model_path))

learning_rate = 1e-4
batch_size = 128

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)

train_losses = []
val_losses = []

best_loss = np.inf

epochs = 20
max_batches = np.inf
loop = tqdm(total = min(len(train_dataloader), max_batches) * epochs)

def loss_function(d, d_hat):
    scale = 1
    return (scale*d - scale*d_hat).pow(2).mean()

def step(state, deltas):
    s = state + deltas
    return s

for epoch in range(epochs):
    model.train()
    # increase number to predict ahead
    new_n_ahead = min((epoch + 1) * 10, 100)
    if new_n_ahead != n_ahead:
        best_loss = np.inf
        train_dataset = RolloutDataset(train_rollouts, n_ahead = n_ahead)
        val_dataset = RolloutDataset(val_rollouts, n_ahead = n_ahead)
        train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
        val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)
    for b, (a, s, d) in enumerate(train_dataloader):
        s = s.float().to(device)
        a = a.float().to(device)
        d = d.float().to(device)
        
        d_est = torch.zeros(d.shape).to(device)

        for i in range(n_ahead):
            d_hat = model(a[:,i], s)
            if i == 0:
                d_est[:,i] = d_est[:,i] + d_hat
            else:
                d_est[:,i] = d_est[:,i-1] + d_hat
            s = step(s, d_hat)
        
        loss = loss_function(d, d_est)
        if not val_losses:
            loop.set_description('loss: {:.3f}'.format(loss.item()))
        else:
            loop.set_description('loss: {:.4f}, val loss: {:.4f}'.format(loss.item(), val_losses[-1]))
        train_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loop.update(1)
        if b > max_batches:
            break
    with torch.no_grad():
        model.eval()
        epoch_losses = []
        for b, (a, s, d) in enumerate(train_dataloader):
            s = s.float().to(device)
            a = a.float().to(device)
            d = d.float().to(device)

            d_est = torch.zeros(d.shape).to(device)

            for i in range(n_ahead):
                # d_hat = model((a[:,i] - action_mean) / action_std,
                #               (s - state_mean) / state_std)
                d_hat = model(a[:,i], s)
                if i == 0:
                    d_est[:,i] = d_est[:,i] + d_hat
                else:
                    d_est[:,i] = d_est[:,i-1] + d_hat
                s = step(s, d_hat)

            loss = loss_function(d, d_est)
            
            epoch_losses.append(loss.item())
            if b > max_batches:
                break
        val_losses.append(np.mean(epoch_losses))
        
        if np.mean(epoch_losses) < best_loss:
            best_loss = np.mean(epoch_losses)
            torch.save(model.state_dict(), model_path)
            print('Saved! {:.6f}'.format(best_loss))
