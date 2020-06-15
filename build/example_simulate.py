import numpy as np
from ViperEnvironment import ViperEnvironment
from tqdm import tqdm
import pdb

env = ViperEnvironment()
s = env.reset()

action_min, action_max = .2, 2

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
