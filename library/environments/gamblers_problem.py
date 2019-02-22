import sys
import numpy as np
import seaborn as sns
from gym import spaces
from gym.utils import seeding
from gym.core import Env
from matplotlib import pyplot as plt
from collections import defaultdict
from library.dynamic_programming.dynamic_programming import policy_iteration


MAX_CAPITAL: int = 100


class GamblersProblem(Env):
    """
    Gambler's Problem environment for the problem defined in example 4.3 of Sutton and Barto.

    A gambler has the opportunity to make bets on the outcomes of a sequence of coin ﬂips.
    If the coin comes up heads, he wins as many dollars as he has staked on that ﬂip;
    if it is tails, he loses his stake.
    The game ends when the gambler wins by reaching his goal of MAX_CAPITAL, or loses by running out of money.
    On each ﬂip, the gambler must decide what portion of his capital to stake, in integer numbers of dollars.

    The state space is defined as s \in {0, ..., MAX_CAPITAL}, where 0 and MAX_CAPITAL are terminal states:
     - state 0: the agent is out of money and he loses
     - state 100: the agent has reached his goal and he wins
     The action space is based on a particular state s: a \in {0, ..., min(s, MAX_CAPITAL - s),
     NOTE: When using the step method, a legal action for a specific state can be chosen
     using the P dictionary like this: np.random.choice(list(P[state].keys()))
    """
    def __init__(self, heads_prob):
        # probability of the coin being heads
        self.heads_prob = heads_prob
        # state space
        self.observation_space = spaces.Discrete(MAX_CAPITAL + 1)
        max_action = MAX_CAPITAL // 2
        # action space
        self.action_space = spaces.Discrete(max_action + 1)
        # calculate transition probabilities
        self.P = {}
        for state in range(1, MAX_CAPITAL + 1):
            self.P[state] = {a: [] for a in range(self._get_max_action(state) + 1)}
            for action in self.P[state].keys():
                self._calculate_transition_prob(state, action)
        # initialize random generator
        self.np_random = None
        self.seed()
        # initialize state and reset the environment
        self.state = None
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: int):
        assert 0 <= action <= self._get_max_action(self.state)
        if self.np_random.binomial(n=1, p=self.heads_prob):
            # heads -> the agent wins
            self.state += action
        else:
            # tails -> the agent loses
            self.state -= action
        reward = 0
        done = False
        if self.state == MAX_CAPITAL:
            reward = 1
            done = True
        if self.state == 0:
            done = True
        return self.state, reward, done, {}

    def reset(self):
        self.state = self.np_random.choice(range(1, MAX_CAPITAL))
        return self.state

    def render(self, mode='human'):
        outfile = sys.stdout
        outfile.write('Gambler\'s capital: {}\n'.format(self.state))

    def render_policy(self, policy):
        x = range(1, MAX_CAPITAL)
        y = np.zeros(len(x), dtype=np.int)
        for index, state in enumerate(x):
            if type(policy) is defaultdict or type(policy) is dict:
                action_values = policy[state]
            else:
                action_values = policy(state)
            y[index] = np.argmax(action_values)
        sns.lineplot(x, y)
        plt.xlabel('Final\npolicy\n(stake)')
        plt.ylabel('Capital')
        plt.title('Policy for the Gambler\'s Problem with p_h = {}'.format(self.heads_prob))
        plt.show()

    def _calculate_transition_prob(self, state, action):
        done_down = state - action == 0
        done_up = state + action == MAX_CAPITAL
        reward = 0
        self.P[state][action].append((1 - self.heads_prob, state - action, reward, done_down))
        if state + action == MAX_CAPITAL:
            reward = 1
        self.P[state][action].append((self.heads_prob, state + action, reward, done_up))

    @staticmethod
    def _get_max_action(state):
        return np.min((state, MAX_CAPITAL - state))


if __name__ == '__main__':
    # TODO: try grid world, make docstring, remove print statements
    from pprint import pprint
    env = GamblersProblem(0.4)
    policy, state_values = policy_iteration(env, iterations=1)
    pprint(policy)
    env.render_policy(policy)
