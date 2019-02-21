"""
Module for the temporal difference learning algorithms.
"""
import itertools
import numpy as np
from collections import defaultdict
from library import plotting


def q_learning(env, num_episodes: int, q=None, discount_factor=1.0,
               alpha=0.5, epsilon=0.1, distribute_prob=True):
    """
    Q-Learning (off-policy control) algorithm implementation as described on page 131
    of the book Reinforcement Learning: An Introduction by Sutton and Barto.
    :param env: The OpenAI Env used
    :param num_episodes: Number of episodes to run the algorithm for
    :param q: Q action state values to start from
    :param discount_factor: The gamma discount factor
    :param alpha: The learning rate
    :param epsilon: Chance to sample a random action
    :param distribute_prob: Whether or not to distribute the probability between best actions
                            or just choose the first best action an assign it all the probability mass.
    :return: a tuple (q, stats) with q the optimal value function,
             and stats, statistics to be used for plotting
    """
    # initialize the action value function
    if q is None:
        q = defaultdict(lambda: np.zeros(env.action_space.n))
    # initialize the statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))
    # initialize the policy
    policy = make_epsilon_greedy_policy(q, epsilon, env.action_space.n, distribute_prob)
    # loop for each episode
    for episode in range(num_episodes):
        # initialize the state
        state = env.reset()
        # loop for each step in the episode
        for t in itertools.count():
            # choose action from state based on the policy
            action_prob = policy(state)
            action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
            next_state, reward, done, _ = env.step(action)
            # update statistics
            stats.episode_rewards[episode] += reward
            stats.episode_lengths[episode] = t
            # q learning update
            best_next_action = np.argmax(q[next_state])
            q[state][action] += alpha * (reward + discount_factor *
                                         q[next_state][best_next_action] - q[state][action])
            # check for finished episode
            if done:
                break
            # otherwise update state
            state = next_state
    return q, stats


def sarsa(env, num_episodes: int, q=None, discount_factor=1.0,
          alpha=0.5, epsilon=0.1, distribute_prob=True):
    """
    SARSA (on-policy control) algorithm implementation as described on page 130
    of the book Reinforcement Learning: An Introduction by Sutton and Barto.
    :param env: The OpenAI Env used
    :param num_episodes: Number of episodes to run the algorithm for
    :param q: Q action state values to start from
    :param discount_factor: The gamma discount factor
    :param alpha: The learning rate
    :param epsilon: Chance to sample a random action
    :param distribute_prob: Whether or not to distribute the probability between best actions
                            or just choose the first best action an assign it all the probability mass.
    :return: a tuple (q, stats) with q the optimal value function,
             and stats, statistics to be used for plotting
    """
    # initialize the action value function
    if q is None:
        q = defaultdict(lambda: np.zeros(env.action_space.n))
    # initialize the statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))
    # initialize the policy
    policy = make_epsilon_greedy_policy(q, epsilon, env.action_space.n, distribute_prob)
    # loop for each episode
    for episode in range(num_episodes):
        # initialize state
        state = env.reset()
        # choose action from state based on the policy
        action_prob = policy(state)
        action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
        # loop for each step in the episode
        for step in itertools.count():
            # take action and observing the next state, reward, if we are done
            # we ignore the probability since action are deterministic
            next_state, reward, done, _ = env.step(action)
            # choose an action from next_state using the policy
            next_action_prob = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_prob)), p=next_action_prob)
            # update the statistics
            stats.episode_rewards[episode] += reward
            stats.episode_lengths[episode] += step
            # sarsa q update
            q[state][action] += alpha * (reward + discount_factor * q[next_state][next_action] - q[state][action])
            # check for finished episode
            if done:
                break
            # otherwise update state and action
            state = next_state
            action = next_action
    return q, stats


def make_epsilon_greedy_policy(q: dict, epsilon: float, action_count: int, distribute_prob=True):
    """
    This function creates an epsilon greedy policy based on the given Q.
    :param q: A dictionary that maps from a state to the action values
              for all possible nA actions (represented as an array)
    :param epsilon: Probability to select a random action
    :param action_count: Number of actions
    :param distribute_prob: Whether or not to distribute the probability between best actions
                            or just choose the first best action an assign it all the probability mass.
    :return: A function that takes as argument an observation and returns
             the probabilities of each action.
    """
    if q is None:
        raise ValueError('Q is None')

    def policy_func(observation):
        actions = np.ones(action_count, dtype=float) * epsilon / action_count
        if distribute_prob:
            best_actions = np.argwhere(q[observation] == np.max(q[observation])).flatten()
            for i in best_actions:
                actions[i] += (1.0 - epsilon) / len(best_actions)
        else:
            best_action = np.argmax(q[observation])
            actions[best_action] += (1.0 - epsilon)
        return actions
    return policy_func
