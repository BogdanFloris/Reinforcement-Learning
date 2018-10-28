import numpy as np
import sys
from gym.envs.toy_text import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
ROWS = 8
COLUMNS = 8


class GridWorldEnv(discrete.DiscreteEnv):
    """
    Custom GridWorld environment.
    You are an agent on a 8x8 grid and your goal is to reach the terminal
    state in the lower right corner while avoiding the snake pit.
    There are also walls on the grid that the agent cannot step into.

    The grid is depicted below:
    o  o  o  o  o  o  o  o
    o  o  W  W  W  W  o  o
    o  o  o  o  o  W  o  o
    o  o  o  o  o  W  o  o
    o  o  o  o  o  W  o  o
    o  o  o  o  S  o  o  o
    o  W  W  W  o  o  o  o
    o  o  o  o  o  o  o  T

    The agent can start from any state that is not terminal or a wall.
    Walls are represented by W, the snake pit by S and the terminal state by T.
    Moving into the snake pit incurs a reward of -20 and finishes the episode.
    Moving into the terminal state incurs a reward of +10 and finishes the episode.
    Moving into a wall or outside the grid incurs a reward of -1, but leaves the state unchanged.
    All other actions also incur a reward of -1.
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def render(self, mode='human'):
        pass
