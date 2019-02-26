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
from library.utils import make_random_policy, make_epsilon_greedy_policy


def mc_control_eps_greedy(env, no_episodes, epsilon=0.1, discount_factor=1.0):
    """
    On-policy first-visit Monte Carlo Control with epsilon greedy policies,
    as described in section 5.4 of Sutton and Barto.
    Finds an optimal epsilon greedy policy for the given environment and
    returns and policy and the q function.
    :param env: the OpenAI Gym environment
    :param no_episodes: how many episodes we sample
    :param epsilon: epsilon value for the greedy policies
    :param discount_factor: the gamma discount factor
    :return: the optimal epsilon greedy policy and the q function
    """
    # initialize returns dictionaries
    sum_returns = defaultdict(float)
    count_returns = defaultdict(float)
    # initialize the q function
    q = defaultdict(lambda: np.zeros(env.action_space.n))
    # initialize the epsilon greedy policy
    policy = make_epsilon_greedy_policy(q, env.action_space.n, epsilon)
    # start looping over episodes
    for episode in range(no_episodes):
        if episode % 1000 == 0:
            print('Episode {}'.format(episode))
        # generate the episode
        episode = generate_episode(env, policy)
        # set of state action tuples that were seen in the episode
        sa_tuples_in_ep = set([(obs[0], obs[1]) for obs in episode])
        for state, action in sa_tuples_in_ep:
            # determine the first occurrence of the state action tuple in the episode
            first_occurrence = [t for t, obs in enumerate(episode)
                                if obs[0] == state and obs[1] == action][0]
            # calculate G
            g = np.sum([obs[2] * np.power(discount_factor, t)
                        for t, obs in enumerate(episode[first_occurrence:])])
            # update returns
            pair = (state, action)
            sum_returns[pair] += g
            count_returns[pair] += 1.0
            # update q function
            q[state][action] = sum_returns[pair] / count_returns[pair]
        # update policy after the episode
        policy = make_epsilon_greedy_policy(q, env.action_space.n, epsilon)
    return q, policy


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
        # generate the episode
        episode = generate_episode(env, policy, )
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


def generate_episode(env, policy):
    """
    Generates an episode of the environment given the policy
    :param env: the OpenAI Gym environment
    :param policy: the policy used to generate the episode
    :return: the episode list
    """
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
    return episode
