import numpy as np
import sys
from gym.envs.toy_text import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
ROWS = 4
COLUMNS = 12


class CliffWalkingEnv(discrete.DiscreteEnv):
    """
    Simple implementation of the Cliff Walking Reinforcement Learning
    exercise as described in Example 6.6 (page 132) from
    Reinforcement Learning: An Introduction by Sutton and Barto.

    This variant is slightly different from the one in the book.
    The board is a ROWSxCOLUMNS with the following properties:
        [ROWS - 1, 0] is the starting state
        [ROWS - 1, COLUMNS - 1] is the end state
        [ROWS - 1, 1..COLUMNS - 2] is the cliff
    All transitions have a reward of -1, while stepping off the cliff ([ROWS - 1, 1..COLUMNS - 2])
    has a reward of -100. Reaching the end state incurs a rewards of +10.
    Inspiration for the code:
    https://github.com/openai/gym/blob/master/gym/envs/toy_text/cliffwalking.py
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.shape = (ROWS, COLUMNS)
        self.start_state_index = np.ravel_multi_index((ROWS - 1, 0), self.shape)

        state_count: int = np.prod(self.shape)
        action_count = 4

        # Cliff location
        self._cliff = np.zeros(self.shape, dtype=np.bool)
        self._cliff[ROWS - 1, 1:-1] = True

        # Calculate transition probabilities and rewards
        transition_prob = {}
        for s in range(state_count):
            position = np.unravel_index(s, self.shape)
            transition_prob[s] = {a: [] for a in range(action_count)}
            transition_prob[s][UP] = self._calculate_transition_prob(position, [-1, 0])
            transition_prob[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
            transition_prob[s][DOWN] = self._calculate_transition_prob(position, [1, 0])
            transition_prob[s][LEFT] = self._calculate_transition_prob(position, [0, -1])

        # Calculate initial state distribution
        # We always start in state (4, 0)
        isd = np.zeros(state_count)
        isd[self.start_state_index] = 1.0

        super(CliffWalkingEnv, self).__init__(state_count, action_count, transition_prob, isd)

    def _calculate_transition_prob(self, current, delta):
        """
        Determines the outcome for an action. Transition Prob is always 1.
        :param current: Current position on the grid as (row, col)
        :param delta: Change in position from transition
        :return: (1.0, new_state, reward, done)
        """
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        if self._cliff[tuple(new_position)]:
            return [(1.0, self.start_state_index, -100, False)]

        terminal_state = (self.shape[0] - 1, self.shape[1] - 1)
        is_done = tuple(new_position) == terminal_state
        if is_done:
            return [(1.0, new_state, 10, True)]
        else:
            return [(1.0, new_state, -1, False)]

    def _limit_coordinates(self, position):
        """
        Prevents the agent from falling of the grid.
        :param position: Current position on the grid as (row, col)
        :return: Adjusted position
        """
        position[0] = max(0, min(position[0], self.shape[0] - 1))
        position[1] = max(0, min(position[1], self.shape[1] - 1))
        return position

    def render(self, mode='human'):
        outfile = sys.stdout

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            if self.s == s:
                output = " x "
            elif position == (ROWS - 1, COLUMNS - 1):
                output = " G "
            elif self._cliff[position]:
                output = " C "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += '\n'

            outfile.write(output)
        outfile.write('\n')

    def render_policy(self, policy):
        """
        Renders the policy of the grid using the given policy function
        :param policy: the policy function given
        """
        outfile = sys.stdout

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            if position == (ROWS - 1, COLUMNS - 1):
                output = " G "
            elif self._cliff[position]:
                output = " C "
            else:
                action_prob = policy(s)
                action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
                if action == 0:
                    output = " \u2191 "
                elif action == 1:
                    output = " \u2192 "
                elif action == 2:
                    output = " \u2193 "
                else:
                    output = " \u2190 "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += '\n'

            outfile.write(output)
        outfile.write('\n')
