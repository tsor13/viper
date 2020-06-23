import interface
import numpy as np
import pdb

class ViperEnvironment:
    def __init__(self):
        self.env = interface.Environment()
        # reset environment to inital position
        self.env.reset()

        # sizes of action and state vector
        self.action_size = 3
        self.state_size = 1404

        # action bounds
        self.action_max = 2
        self.action_min = .2

        self.time_step = 0
        self.max_steps = 2000

    def step(self, action):
        '''
        Step 1/60th of a second with action.
        Action is either a 3-size numpy array or length 3 iterable.
        Each element of action represents a contraction or extending of a thread of tentacle pills.
        Less than 1 is a contraction, 1 is the original length, and greater than 1 is an extension.
        '''
        action = np.array(action)
        if len(action) != 3:
            raise Exception(f'Action must be of size {self.action_size}')
        if action.max() > 2 or action.min() < .2:
            raise Exception(f'Action values must be within {self.action_min} and {self.action_max}')
        # step
        self.env.step(*action)

        # STEP
        state = np.array([self.env.get_state(i) for i in range(1404)])

        # REWARD
        # penalty is euclidean distance between tip of tentacle and point on cube
        tentacle_index = 12 * 89
        tentacle_point = state[tentacle_index:tentacle_index+3]
        cube_point = 12 * 90
        cube_point = state[cube_point:cube_point+3]
        dist = np.sqrt(((tentacle_point - cube_point)**2).sum())
        reward = -dist

        # TERMINAL
        self.time_step += 1
        done = self.time_step >= self.max_steps

        return state, reward, done

    def reset(self):
        ''' 
        Resets environment to random initial position.
        The initial velocity of the tentacle and position of the cube is random
        '''
        self.env.reset()
        self.time_step = 0
        state = np.array([self.env.get_state(i) for i in range(self.state_size)])
        return state

    def __repr__(self):
        s = f'Viper Environment: t = {self.time_step}'
        return s

    def __str__(self):
        s = f'Viper Environment: t = {self.time_step}'
        return s
