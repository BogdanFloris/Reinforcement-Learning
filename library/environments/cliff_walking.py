import numpy as np
import sys
from gym.envs.toy_text import discrete


UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class CliffWalkingEnv(discrete.DiscreteEnv):
    """
    Simple implementation of the Cliff Walking Reinforcement Learning
    exercise as described in Example 6.6 (page 132) from
    Reinforcement Learning: An Introduction by Sutton and Barto.

    This variant is slightly different from the one in the book.
    The board is a 5x10 with the following properties:
        [4, 0] is the starting state
        [4, 9] is the end state
        [4, 1..8] is the cliff
    All transitions have a reward of -1, while stepping off the cliff ([4, 1..9])
    has a reward of -100. Reaching the end state incurs a rewards of +10.
    Inspiration for the code:
    https://github.com/openai/gym/blob/master/gym/envs/toy_text/cliffwalking.py
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.shape = (5, 10)
        self.start_state_index = np.ravel_multi_index((4, 0), self.shape)

        nS = np.prod(self.shape)
        nA = 4

        # Cliff location
        self._cliff = np.zeros(self.shape, dtype=np.bool)
        self._cliff[4, 1:-1] = True

        # Calculate transition probabilities and rewards
        P = {}

        # Calculate initial state distribution
        # We always start in state (4, 0)
        isd = np.zeros(nS)
        isd[self.start_state_index] = 1.0

        super(CliffWalkingEnv, self).__init__(nS, nA, P, isd)

    def _calculate_transition_prob(self, current, delta):
        """
        Determines the outcome for an action. Transition Prob is always 1.
        :param current: Current position on the grid as (row, col)
        :param delta: Change in position from transition
        :return: (1.0, new_state, reward, done)
        """
        pass

    def render(self, mode='human'):
        pass
