"""
Training for the game of breakout
"""
import sys
sys.path.append('../')
import gym
from library.dqn.dqn import DQNAgent

# make environment
env = gym.make('Breakout-v0')
agent = DQNAgent(env, num_episodes=10000)


if __name__ == '__main__':
    # start training
    agent.dqn(num_episodes=10000)
