import sys
import numpy as np
from gym import spaces
from gym.utils import seeding
from gym.core import Env


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
     but in this environment is defined to be s \in {0, ..., MAX_CAPITAL // 2} for compatibility
     with the OpenAI Gym environment. When an illegal state is picked (out of range),
     the step method simply selects another legal action.
     This makes no difference for the theoretical problem.
    """
    def __init__(self, heads_prob):
        # probability of the coin being heads
        self.heads_prob = heads_prob
        # state space
        self.observation_space = spaces.Discrete(MAX_CAPITAL + 1)
        max_action = MAX_CAPITAL // 2
        # action space
        self.action_space = spaces.Discrete(max_action + 1)
        self.np_random = None
        self.seed()
        self.state = None
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: int):
        assert 0 <= action <= MAX_CAPITAL // 2
        max_action = np.min((self.state, MAX_CAPITAL - self.state))
        if action > max_action:
            # the action_space is defined to be between 0 and MAX_CAPITAL // 2 in the environment,
            # but for each state we can only choose an action in the range 0..min(s, MAX_CAPITAL - s);
            # to keep things compatible with all the algorithms, if the input action is illegal,
            # meaning not in range, we can just select another random action that is legal;
            # this makes no difference theoretically, and is convenient for backwards compatibility;
            action = self.np_random.choice(range(0, max_action + 1))
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

    def render(self, mode='human'):
        outfile = sys.stdout
        outfile.write('Gambler\'s capital: {}\n'.format(self.state))


if __name__ == '__main__':
    env = GamblersProblem(0.4)
    for _ in range(100):
        env.render()
        s, r, d, _ = env.step(np.random.choice(range(MAX_CAPITAL // 2 + 1)))
        if d:
            break
