import numpy as np
from ViperEnvironment import ViperEnvironment
import pdb

env = ViperEnvironment()
s = env.reset()

action_min, action_max = .2, 2

with open('output.txt', 'w') as file:
    for i in range(1400):
        rand = np.random.rand(3)
        action = action_min * rand + (1-rand) * action_max
        state, reward = env.step(action)
        combined = np.hstack([state, action])
        row = ' '.join([str(f) for f in combined])
        file.write(row)
        file.write('\n')
    file.close()
