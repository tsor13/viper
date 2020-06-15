import numpy as np
from ViperEnvironment import ViperEnvironment
from tqdm import tqdm
import pdb

env = ViperEnvironment()
s = env.reset()

action_min, action_max = .2, 2

with open('output.txt', 'w') as file:
    for i in tqdm(range(20000)):
        rand = np.random.rand(3)
        action = action_min * rand + (1-rand) * action_max
        # action = np.array([1.5, 1.5, .2])
        state, reward = env.step(action)
        combined = np.hstack([state, action])
        row = ' '.join([str(f) for f in combined])
        file.write(row)
        file.write('\n')
    file.close()
