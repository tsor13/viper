import numpy as np
from ViperEnvironment import ViperEnvironment
from tqdm import tqdm
import pdb

env = ViperEnvironment()
s = env.reset()

action_min, action_max = .2, 2

# do one minute long simulation, write data
with open('output.txt', 'w') as file:
    rand = np.random.rand(3)
    action = action_min * rand + (1-rand) * action_max
    for i in tqdm(range(3600)):
        if i % 50:
            rand = np.random.rand(3)
            action = action_min * rand + (1-rand) * action_max
        # action = np.array([1.5, 1.5, .2])
        state, reward = env.step(action)
        combined = np.hstack([state, action])
        row = ' '.join([str(f) for f in combined])
        file.write(row)
        file.write('\n')
        if i == 1800:
            s  = env.reset()
    file.close()

# check reward distribution for trial
# rewards = []
# for _ in tqdm(range(50)):
#     for i in range(1000):
#         rand = np.random.rand(3)
#         action = action_min * rand + (1-rand) * action_max
#         s = env.reset()
#         if i % 50:
#             rand = np.random.rand(3)
#             action = action_min * rand + (1-rand) * action_max
#         state, reward = env.step(action)
#         rewards.append(reward)
# 
# rewards = np.array(rewards)
# quantiles = [np.quantile(rewards, q) for q in np.linspace(0, 1, 5)]
# print(quantiles)
# pdb.set_trace()
# a = 3
