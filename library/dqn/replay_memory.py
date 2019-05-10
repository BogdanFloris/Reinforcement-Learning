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
        self.frames = np.zeros(shape=(buffer_size, height, width), dtype=np.uint8)
        self.actions = np.zeros(shape=buffer_size, dtype=np.int)
        self.rewards = np.zeros(shape=buffer_size, dtype=np.float)
        self.done = np.zeros(shape=buffer_size, dtype=np.bool)

        # initialize mini batches for states and new states
        self.states = np.zeros(shape=(batch_size, m, height, width), dtype=np.uint8)
        self.new_states = np.zeros(shape=(batch_size, m, height, width), dtype=np.uint8)
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
        self.current = (self.current + 1) % self.size

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


class ReplayMemoryOld:
    """
    The replay memory of a DQN agent
    represented a circular buffer.
    """
    def __init__(self, height, width, rng, buffer_size=1000, m=4):
        """
        Initializes the replay memory.
        :param height: the image height
        :param width: the image width
        :param rng: the random number generator
        :param buffer_size: size of the buffer
        :param m: number of frames used as input to the network
        """
        raise ValueError('Obsolete')
        self.height = height
        self.width = width
        self.rng = rng
        self.buffer_size = buffer_size
        self.m = m
        # initialize buffers
        self.states = np.zeros(shape=(buffer_size, height, width, m), dtype=np.float)
        self.actions = np.zeros(shape=buffer_size, dtype=np.int)
        self.rewards = np.zeros(shape=buffer_size, dtype=np.float)
        self.next_states = np.zeros(shape=(buffer_size, height, width, m), dtype=np.float)
        self.done = np.zeros(shape=buffer_size, dtype=np.bool)
        # number of elements in the buffer
        self.size = 0
        # pointers in the buffer
        self.bottom = 0
        self.top = 0

    def __len__(self):
        return self.size

    def add_sample(self, state, action, reward, next_state, done):
        """
        Adds a sample to the memory.
        :param state: the state sample
        :param action: the action sample
        :param reward: the reward sample
        :param next_state: the next state sample
        :param done: is the sample done or not
        """
        # put the elements in the buffer
        self.states[self.top] = state
        self.actions[self.top] = action
        self.rewards[self.top] = reward
        self.next_states[self.top] = next_state
        self.done[self.top] = done
        # update the pointers in the buffer
        if self.size == self.buffer_size:
            # the buffer is full so wrap around
            self.bottom = (self.bottom + 1) % self.buffer_size
        else:
            # just increase the size
            self.size += 1
        self.top = (self.top + 1) % self.buffer_size

    def sample_minibatches(self, batch_size):
        """
        Samples batch_size samples from the replay memory
        :param batch_size: the minibatch size
        :return: the samples
        """
        # initialize the returns
        states = np.zeros((batch_size, self.height, self.width, self.m), dtype=np.float)
        actions = np.zeros(batch_size, dtype=np.int)
        rewards = np.zeros(batch_size, dtype=np.float)
        next_states = np.zeros((batch_size, self.height, self.width, self.m), dtype=np.float)
        done = np.zeros(batch_size, dtype=np.bool)
        # counter for batches
        batch_index = 0
        while batch_index < batch_size:
            # choose index for replay memory
            index = self.rng.randint(self.bottom, self.bottom + self.size)
            # assign to the returns (wrap around)
            states[batch_index] = self.states.take(index, axis=0, mode='wrap')
            actions[batch_index] = self.actions.take(index, mode='wrap')
            rewards[batch_index] = self.rewards.take(index, mode='wrap')
            next_states[batch_index] = self.next_states.take(index, axis=0, mode='wrap')
            done[batch_index] = self.done.take(index, mode='wrap')
            # increment batch index
            batch_index += 1
        return states, actions, rewards, next_states, done


class FrameQueue:
    """
    Queue of size m that holds the last seen m frames.
    """
    def __init__(self, height, width, m=4):
        # initialize the queue
        self.queue = np.zeros(shape=(height, width, m), dtype=np.float)
        raise ValueError('Obsolete')

    def append(self, frame):
        """
        Appends a frame to the end of the queue
        :param frame: an (84, 84) frame
        """
        self.queue = np.append(self.queue[:, :, 1:], np.expand_dims(frame, 2), axis=2)

    def get_queue(self):
        """
        :return: the queue
        """
        return self.queue
