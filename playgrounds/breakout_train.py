"""
Training for the game of breakout
"""
import sys
from pathlib import Path
p = Path(__file__).resolve().parents[1]
sys.path.append(str(p))
from library.dqn.dqn import DQNAgent

# make environment
# !set the train flag if training
agent = DQNAgent('BreakoutDeterministic-v4', train=True)


if __name__ == '__main__':
    # start training
    agent.dqn()
