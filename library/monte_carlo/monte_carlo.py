"""
Monte Carlo methods that learn directly from experience and do not assume
complete knowledge of the environment as described in Chapter 5 of Sutton and Barto.
Implemented algorithms:
 - MC Prediction
 -
 -
"""
import numpy as np
from collections import defaultdict
from library.utils import make_random_policy


def mc_prediction(env, no_episodes, policy=None, discount_factor=1.0):
    """
    Monte Carlo Prediction algorithm as described in Sutton and Barto 5.1
    The algorithm calculates the state value functions by sampling a given
    number of episodes using the given policy.
    :param env: OpenAI gym environment
    :param no_episodes: how many episodes we sample
    :param policy: the policy used to sample episodes
    :param discount_factor: the gamma discount factor
    :return: the state value function
    """
    if policy is None:
        # make random policy
        policy = make_random_policy(env.action_space.n)
    # initialize returns dictionaries
    sum_returns = defaultdict(float)
    count_returns = defaultdict(float)
    # initialize the state values
    state_values = defaultdict(float)
    # start looping over episodes
    for episode in range(no_episodes):
        if episode % 1000 == 0:
            print('Episode {}'.format(episode))
        # initialize episode list
        episode = []
        # reset the environment and get the state
        state = env.reset()
        # initialize done boolean
        done = False
        # loop while the episode isn't finished
        while not done:
            # get the action given the policy
            if type(policy) is defaultdict or type(policy) is dict:
                action_prob = policy[state]
            else:
                action_prob = policy(state)
            action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
            # take the action
            next_state, reward, done, _ = env.step(action)
            # append the results to the episode list
            episode.append((state, action, reward))
            # break if the episode is finished
            if done:
                break
            state = next_state
        # set of states that were seen in the episode
        states_in_ep = set([obs[0] for obs in episode])
        for state in states_in_ep:
            # determine the first occurrence of the state in the episode
            first_occurrence = [t for t, obs in enumerate(episode) if obs[0] == state][0]
            # calculate G
            g = np.sum([obs[2] * np.power(discount_factor, t)
                        for t, obs in enumerate(episode[first_occurrence:])])
            # update returns
            sum_returns[state] += g
            count_returns[state] += 1.0
            # update state value
            state_values[state] = sum_returns[state] / count_returns[state]
    return state_values
