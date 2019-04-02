"""
This module defines the replay memory dataset
used to store the states seen by the DQN agent.
"""
import numpy as np


class ReplayMemory:
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
        self.height = height
        self.width = width
        self.rng = rng
        self.buffer_size = buffer_size
        self.m = m
        # initialize buffers
        self.states = np.zeros(shape=(buffer_size, m, height, width), dtype=np.float)
        self.actions = np.zeros(shape=buffer_size, dtype=np.int)
        self.rewards = np.zeros(shape=buffer_size, dtype=np.float)
        self.next_states = np.zeros(shape=(buffer_size, m, height, width), dtype=np.float)
        self.done = np.zeros(shape=buffer_size, dtype=np.bool)
        # number of elements in the buffer
        self.size = 0
        # pointers in the buffer
        self.bottom = 0
        self.top = 0

    def __len__(self):
        return self.size

    def add_sample(self, state, action, reward, next_state, terminal):
        """
        Adds a sample to the memory.
        :param state: the state sample
        :param action: the action sample
        :param reward: the reward sample
        :param next_state: the next state sample
        :param terminal: is the sample done or not
        """
        # put the elements in the buffer
        self.states[self.top] = state
        self.actions[self.top] = action
        self.rewards[self.top] = reward
        self.next_states[self.top] = next_state
        self.done[self.top] = terminal
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
        states = np.zeros((batch_size, self.m, self.height, self.width), dtype=np.float)
        actions = np.zeros(batch_size, dtype=np.int)
        rewards = np.zeros(batch_size, dtype=np.float)
        next_states = np.zeros((batch_size, self.m, self.height, self.width), dtype=np.float)
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
