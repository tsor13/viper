import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical, Normal, Beta
from tqdm import tqdm
from time import time
import pdb
from datetime import datetime

from ViperEnvironment import ViperEnvironment

device = 'cuda'

# initialize environment
env = ViperEnvironment()
# env._max_episode_steps = 1600
state_size = env.state_size
action_size = env.action_size
action_high = env.action_max
action_low = env.action_min

# rollout functions
def get_action():
    global action_size, action_high, action_low
    np.random.seed(int(time()*1000)%(2**32))
    action = np.random.rand(action_size)
    # action = np.array([0, 1, 0])
    action = action * action_high + (1-action) * action_low
    return action

def rollout(repeat = 10):
    state = env.reset()
    done = False
    states, actions = [], []
    while not done:
        action = get_action()

        for _ in range(repeat):
            actions.append(action)
            states.append(state)
            state, r, done = env.step(action)
            if done:
                break

    actions = np.array(actions)
    states = np.array(states)
    memory = np.hstack([actions, states])
    # since first couple doesn't include shifting of tentacle
    memory = memory[2:]
    return memory

def save_rollouts(n):
    for _ in tqdm(range(n)):
        repeat = np.random.randint(100, 200)
        memory = rollout(repeat)
        now_obj = datetime.now()
        now_string = now_obj.strftime('%d-%b-%Y-%H-%M-%s')
        np.save(f'rollouts/{now_string}.npy', memory)

save_rollouts(500)
