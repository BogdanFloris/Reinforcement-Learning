"""
Training for the game of breakout
"""
import sys
sys.path.append('../')
import gym
from library.dqn.dqn import DQNAgent

# make environment
env = gym.make('Breakout-v0')
agent = DQNAgent(env, num_episodes=10000,
                 buffer_size=100000,
                 init_buffer=10000,
                 eps_decay_steps=100000)


if __name__ == '__main__':
    # start training
    pass
