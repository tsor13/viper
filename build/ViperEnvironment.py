import interface
import numpy as np
import pdb

class ViperEnvironment:
    def __init__(self):
        self.env = interface.Environment()
        self.env.reset()
    def step(self, action):
        assert len(action) == 3
        assert action.max() <= 2 and action.min() >= .2
        self.env.step(*action)
        state = np.array([self.env.get_state(i) for i in range(1404)])

        # penalty is euclidean distance between tip of tentacle and point on cube
        tentacle_index = 12 * 89
        tentacle_point = state[tentacle_index:tentacle_index+3]
        cube_point = 12 * 90
        cube_point = state[cube_point:cube_point+3]
        dist = ((tentacle_point - cube_point)**2).mean()
        reward = -dist
        return state, reward
    def reset(self):
        self.env.reset()
        state = np.array([self.env.get_state(i) for i in range(1404)])
        return state
