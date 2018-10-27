"""
Module for the temporal difference learning algorithms.
"""
import itertools
import numpy as np
from collections import defaultdict
from library import plotting


def sarsa(env, num_episodes: int, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    SARSA (on-policy control) algorithm implementation as described on page 130
    of the book Reinforcement Learning: An Introduction by Sutton and Barto.
    :param env: The OpenAI Env used
    :param num_episodes: Number of episodes to run the algorithm for
    :param discount_factor: The gamma discount factor
    :param alpha: The learning rate
    :param epsilon: Chance to sample a random action
    :return: a tuple (q, stats) with q the optimal value function,
             and stats, statistics to be used for plotting
    """
    # initialize the action value function
    q = defaultdict(lambda: np.zeros(env.action_space.n))
    # initialize the statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))
    # initialize the policy
    policy: function = make_epsilon_greedy_policy(q, epsilon, env.action_space.n)

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


def make_epsilon_greedy_policy(q: dict, epsilon: float, action_count: int):
    """
    This function creates an epsilon greedy policy based on the given Q.
    :param q: A dictionary that maps from a state to the action values
              for all possible nA actions (represented as an array)
    :param epsilon: Probability to select a random action
    :param action_count: Number of actions
    :return: A function that takes as argument an observation and returns
             the probabilities of each action.
    """
    def policy_func(observation):
        actions = np.ones(action_count, dtype=float) * epsilon / action_count
        best_action = np.argmax(q[observation])
        actions[best_action] += (1.0 - epsilon)
        return actions
    return policy_func
