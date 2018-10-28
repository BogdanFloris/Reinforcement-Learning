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

    def __init__(self):
        self.shape = (8, 8)

        state_count: int = np.prod(self.shape)
        action_count = 4

        # wall location
        self._wall = np.zeros(self.shape, dtype=np.bool)
        self._wall[1, 2:6] = True
        self._wall[6, 1:4] = True
        self._wall[2:5, 5] = True

        # Calculate transition probabilities and rewards
        # Also calculate initial state distribution
        # We can start in any state that is not terminal or a wall
        transition_prob = {}
        isd = np.ones(state_count, dtype=np.float)
        for s in range(state_count):
            # transition probabilities
            position = np.unravel_index(s, self.shape)
            transition_prob[s] = {a: [] for a in range(action_count)}
            transition_prob[s][UP] = self._calculate_transition_prob(position, [-1, 0])
            transition_prob[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
            transition_prob[s][DOWN] = self._calculate_transition_prob(position, [1, 0])
            transition_prob[s][LEFT] = self._calculate_transition_prob(position, [0, -1])

            # initial state distribution
            if self._wall[position] or position == (5, 4) or position == (self.shape[0] - 1, self.shape[1] - 1):
                isd[s] = 0.0

        # Calculate actual probabilities for isd
        isd = isd / np.sum(isd)

        super(GridWorldEnv, self).__init__(state_count, action_count, transition_prob, isd)

    def _calculate_transition_prob(self, current, delta):
        """
        Determines the outcome for an action. Transition Prob is always 1.
        :param current: Current position on the grid as (row, col)
        :param delta: Change in position from transition
        :return: (1.0, new_state, reward, done)
        """
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        # check to see if we hit a wall
        if self._wall[tuple(new_position)]:
            old_state = np.ravel_multi_index(current, self.shape)
            return [(1.0, old_state, -1, False)]

        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        snake_pit = (5, 4)
        terminal_state = (self.shape[0] - 1, self.shape[1] - 1)
        is_done_snake_pit = tuple(new_position) == snake_pit
        is_done_terminal = tuple(new_position) == terminal_state
        if is_done_snake_pit:
            return [(1.0, new_state, -20, True)]
        elif is_done_terminal:
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
            elif self._wall[position]:
                output = " W "
            elif position == (5, 4):
                output = " S "
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
            elif self._wall[position]:
                output = " W "
            elif position == (5, 4):
                output = " S "
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
