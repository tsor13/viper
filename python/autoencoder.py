import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical, Normal, Beta
from tqdm import tqdm
from time import time
import pdb

from ViperEnvironment import ViperEnvironment

device = 'cuda'

# initialize environment
encoder_path = 'encoder-relu.pth'
decoder_path = 'decoder-relu.pth'
hidden_size = 4096
hidden_layers = 1
env = ViperEnvironment()
state_size = env.state_size
action_size = env.action_size
action_high = env.action_max
action_low = env.action_min
max_actions = np.inf
repeat = 25
gamma = .95

obs = env.reset()
# classes

class Encoder(nn.Module):
    def __init__(self, state_size=1404, latent_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_size)
        )

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, latent_size=64, state_size=1404):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, state_size)
        )

    def forward(self, x):
        return self.net(x)


def random_action():
    l = np.random.rand(3)
    return l * action_low + (1-l) * action_high

def rollout():
    state = env.reset()
    done = False
    states = []
    while not done:
        action = random_action()
        for _ in range(repeat):
            state, _, done = env.step(action)
            states.append(state)
            if done: 
                break
    return states

def do_rollouts(n):
    states = []
    for _ in tqdm(range(n)):
        s = rollout()
        states.extend(s)
    return states


class StateDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    epochs = 100
    batch_size = 16
    learning_rate = 3e-5

    train_states = do_rollouts(700)
    test_states = do_rollouts(300)
    # train_states = do_rollouts(70)
    # test_states = do_rollouts(30)

    train_dataset = StateDataset(train_states)
    test_dataset = StateDataset(test_states)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

    loop = tqdm(total = len(train_loader) * epochs)

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr = learning_rate)

    test_loss = 0
    best_test = np.inf

    for epoch in range(epochs):
        for s in train_loader:
            s = s.float().to(device)
            s_hat = decoder(encoder(s))

            loss = (s - s_hat).pow(2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.update(1)
            loop.set_description('loss: {:.3f}, test: {:.3f}'.format(loss, test_loss))
        with torch.no_grad():
            epoch_losses = []
            for s in test_loader:
                s = s.float().to(device)
                s_hat = decoder(encoder(s))

                loss = (s - s_hat).pow(2).mean()
                epoch_losses.append(loss.item())
            test_loss = np.mean(epoch_losses)
            loop.set_description('loss: {:.3f}, test: {:.3f}'.format(loss, test_loss))
            if test_loss < best_test:
                best_test = test_loss
                torch.save(encoder.state_dict(), encoder_path)
                torch.save(decoder.state_dict(), decoder_path)
                print('Saved!')

    print(best_test)
