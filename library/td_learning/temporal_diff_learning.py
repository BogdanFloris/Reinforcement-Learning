"""
Module for the temporal difference learning algorithms.
"""
import itertools
import numpy as np
from collections import defaultdict
from library import plotting
from library.utils import make_epsilon_greedy_policy
from library.estimators.estimators import Estimator
from tqdm import tqdm


def q_learning(env, num_episodes: int, q=None,
               estimator: Estimator = None,
               discount_factor=1.0, alpha=0.5,
               epsilon=0.1, epsilon_decay=1.0,
               distribute_prob=True):
    """
    Q-Learning (off-policy control) algorithm implementation as described on page 131
    of the book Reinforcement Learning: An Introduction by Sutton and Barto.
    :param env: The OpenAI Env used
    :param num_episodes: Number of episodes to run the algorithm for
    :param q: Q action state values to start from
    :param estimator: The estimator used to predict the q function
                      in case of Q learning with function approximation
    :param discount_factor: The gamma discount factor
    :param alpha: The learning rate
    :param epsilon: Chance to sample a random action
    :param epsilon_decay: Used to update the epsilon value as we move in time
    :param distribute_prob: Whether or not to distribute the probability between best actions
                            or just choose the first best action an assign it all the probability mass.
    :return: a tuple (q, stats) with q the optimal value function,
             and stats, statistics to be used for plotting
    """
    # initialize the action value function
    if q is None and estimator is None:
        q = defaultdict(lambda: np.zeros(env.action_space.n))
    # determine if we are in the discrete state case and using Q function
    # or in the continuous state case and we are using an estimator
    discrete = q is not None and estimator is None
    # initialize the statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))
    policy = None
    # initialize the policy for the Q function case
    if discrete:
        policy = make_epsilon_greedy_policy(env.action_space.n, epsilon=epsilon,
                                            q=q, distribute_prob=distribute_prob)
    # loop for each episode
    for episode in tqdm(range(num_episodes)):
        # initialize policy for the estimator case
        if not discrete:
            policy = make_epsilon_greedy_policy(env.action_space.n,
                                                epsilon=epsilon * epsilon_decay ** episode,
                                                estimator=estimator)
        # initialize the state
        state = env.reset()
        # loop for each step in the episode
        for t in itertools.count():
            # choose action from state based on the policy
            action_prob = policy(state)
            action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
            # take a step in the environment
            next_state, reward, done, _ = env.step(action)
            # update statistics
            stats.episode_rewards[episode] += reward
            stats.episode_lengths[episode] = t
            if discrete:
                # q learning update for Q function
                best_next_action = np.argmax(q[next_state])
                q[state][action] += alpha * (reward + discount_factor *
                                             q[next_state][best_next_action] - q[state][action])
            else:
                # q learning update for the estimator
                q_values = estimator.predict(next_state)
                td_update = reward + discount_factor * np.max(q_values)
                estimator.update(state, action, td_update)
            # check for finished episode
            if done:
                break
            # otherwise update state
            state = next_state
    return q, stats if discrete else estimator, stats


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
    policy = make_epsilon_greedy_policy(env.action_space.n, epsilon=epsilon,
                                        q=q, distribute_prob=distribute_prob)
    # loop for each episode
    for episode in tqdm(range(num_episodes)):
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
