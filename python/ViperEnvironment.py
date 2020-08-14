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
        # self.state_size = 24

        # action bounds
        self.action_max = 1.8
        self.action_min = .2

        self.time_step = 0
        self.max_steps = 2000
        # self.max_steps = 1000

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
        # STATE
        # for each point:
        # 0, 1, 2: current coordinate (x, y, z)
        # 3, 4, 5: previous coordinate (x, y, z)
        # 6, 7, 8: current - previous (dx, dy, xz)
        # 9, 10, 11: rotation, previous rottaion, rotation delta (rotation - previous)
        state = np.array([self.env.get_state(i) for i in range(1404)])

        # REWARD
        # penalty is euclidean distance between tip of tentacle and point on cube
        point_size = 12

        tentacle_index = 71 * point_size
        cube_index = 103 * point_size

        tentacle_point = state[tentacle_index:tentacle_index + 3]
        cube_point = state[cube_index:cube_index + 3]
        tentacle_velocity = state[tentacle_index+6:tentacle_index+9]
        velocity_reward = -np.sqrt((tentacle_velocity**2).sum())
        velocity_reward = -((10*tentacle_velocity)**2).sum()
        velocity_reward += 25000 / self.max_steps
        velocity_reward *= 1 / 800
        # reward equal to movement in x direction
        # velocity_reward = tentacle_velocity[0]

        # just a test target point to see if it can learn well
        target_point = np.array([0, 0, 0])
        # dist = (cube_point - tentacle_point)**2).sum()
        dist = np.sqrt(((cube_point - tentacle_point)**2).sum())
        # dist = np.sqrt(((target_point - tentacle_point)**2).sum())
        dist_reward = -dist
        # shift by mean
        dist_reward += 55345 / self.max_steps
        # scale by std
        dist_reward *= 1 / 3000

        # reward = velocity_reward + dist_reward
        # reward = velocity_reward/2 + 10*dist_reward
        reward = velocity_reward + dist_reward

        # TERMINAL
        self.time_step += 1
        done = self.time_step >= self.max_steps

        # shrink state to be only end and cube
        # tentacle_vector = state[tentacle_index: tentacle_index+point_size]
        # cube_vector = state[cube_index: cube_index+point_size]
        # state = np.hstack([tentacle_vector, cube_vector])

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
