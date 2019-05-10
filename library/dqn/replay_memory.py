"""
This module defines the replay memory dataset
used to store the states seen by the DQN agent.
"""
import numpy as np


class ReplayMemory:
    """
    The replay memory of a DQN agent
    represented as a circular buffer.
    """
    def __init__(self, height, width, rng, buffer_size=100000, m=4, batch_size=32):
        """
        Initialize the replay memory.
        :param height: the image height
        :param width: the image width
        :param rng: the random number generator
        :param buffer_size: size of the replay memory
        :param m: number of frames used as input to the network
        :param batch_size: mini batch size
        """
        self.height = height
        self.width = width
        self.rng = rng
        self.buffer_size = buffer_size
        self.m = m
        self.batch_size = batch_size
        self.size = 0
        self.current = 0

        # initialize buffers for history
        self.frames = np.zeros(shape=(buffer_size, height, width), dtype=np.float32)
        self.actions = np.zeros(shape=buffer_size, dtype=np.int)
        self.rewards = np.zeros(shape=buffer_size, dtype=np.float)
        self.done = np.zeros(shape=buffer_size, dtype=np.bool)

        # initialize mini batches for states and new states
        self.states = np.zeros(shape=(batch_size, m, height, width), dtype=np.float32)
        self.new_states = np.zeros(shape=(batch_size, m, height, width), dtype=np.float32)
        self.indices = np.zeros(shape=batch_size, dtype=np.int)

    def add_sample(self, frame, action, reward, done):
        """
        Adds a sample to the memory
        :param frame: the frame sample
        :param action: the action sample
        :param reward: the reward sample
        :param done: terminal state or not
        """
        if frame.shape != (self.height, self.width):
            raise ValueError('Wrong frame dimension')
        self.frames[self.current] = frame
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.done[self.current] = done
        self.size = max(self.size, self.current + 1)
        self.current = (self.current + 1) % self.buffer_size

    def _get_valid_indices(self):
        """
        Gets valid indices from the replay memory
        :return: the indices
        """
        for i in range(self.batch_size):
            while True:
                index = self.rng.randint(self.m, self.size)
                if index < self.m:
                    continue
                if index - self.m <= self.current <= index:
                    continue
                if self.done[index - self.m:index].any():
                    continue
                break
            self.indices[i] = index

    def _get_state(self, index):
        """
        Gets the state given an index
        :param index: the index
        :return: the state
        """
        if self.size == 0:
            raise ValueError('Empty memory')
        if index < self.m - 1:
            raise ValueError('Index must be minimum {}'.format(self.m - 1))
        return self.frames[index - self.m + 1:index + 1, ...]

    def sample_minibatch(self):
        """
        Samples mini batch from memory.
        :return: the mini batch
        """
        if self.size < self.m:
            raise ValueError('Not enough memory to sample mini batch.')
        self._get_valid_indices()
        for i, index in enumerate(self.indices):
            self.states[i] = self._get_state(index - 1)
            self.new_states[i] = self._get_state(index)

        return np.transpose(self.states, axes=(0, 2, 3, 1)),\
            self.actions[self.indices],\
            self.rewards[self.indices],\
            np.transpose(self.new_states, axes=(0, 2, 3, 1)),\
            self.done[self.indices]
