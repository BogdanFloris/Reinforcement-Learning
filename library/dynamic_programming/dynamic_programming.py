"""
Dynamic programming algorithms as described in Sutton and Barto Chapter 4
 - Policy Iteration (made up of Iterative Policy Evaluation and Policy Improvement)
 - Value Iteration
"""
import numpy as np
from collections import defaultdict


def value_iteration(env, discount_factor=1.0, theta=0.00001):
    """
    Performs value iteration algorithm as described in the Sutton and Barto book.
    :param env: The OpenAI Gym environment:
                 - env.P - transition probabilities of the environment
                 - env.P[state][action] - a list of transition tuples
                 - env.observation_space.n - number of states
                 - env.action_space.n - number of actions
    :param discount_factor: gamma the discount factor
    :param theta: we stop iterating once the state value changes are less than theta
    :return:
    """
    # initialize the state values
    state_values = np.zeros(env.observation_space.n, dtype=np.float64)
    # initialize delta check
    delta = theta + 1.0
    while delta > theta:
        delta = 0
        # loop over all the states
        for state in env.P.keys():
            # get the action values
            action_values = get_action_values(state, env, state_values)
            # get the best action value
            best_action_value = np.max(action_values)
            # update delta
            delta = np.max((delta, np.abs(best_action_value - state_values[state])))
            # update state value
            state_values[state] = best_action_value
    # initialize the policy
    policy = defaultdict(lambda: np.ones(env.action_space.n) / env.action_space.n)
    # loop over each state
    for state in env.P.keys():
        # compute action values
        action_values = get_action_values(state, env, state_values, discount_factor)
        # get the best action
        best_action = np.argmax(action_values)
        # update policy
        policy[state] = np.eye(len(env.P[state]))[best_action]
    return policy, state_values


def policy_iteration(env, policy=None, discount_factor=1.0, theta=0.00001):
    """
    Performs policy iteration by performing policy evaluation, then policy iteration
    sequentially until the policy does not change anymore.
    :param env: The OpenAI Gym environment
    :param policy: the policy to be evaluated (if None, initialize an equidistant policy)
    :param discount_factor: gamma the discount factor
    :param theta: we stop iterating once the state value changes are less than theta
    :return: the policy and the state values
    """
    # initialize the policy if it is None
    if policy is None:
        policy = defaultdict(lambda: np.ones(env.action_space.n) / env.action_space.n)
        for state in env.P.keys():
            policy[state] = np.ones(len(env.P[state])) / len(env.P[state])
    state_values = None
    policy_stable = False
    # iterate while the policy isn't stable
    while policy_stable is False:
        # perform policy evaluation
        state_values = policy_evaluation(env, policy, discount_factor, theta)
        # perform policy iteration
        policy_stable = policy_improvement(env, policy, state_values, discount_factor)
    return policy, state_values


def policy_evaluation(env, policy=None, discount_factor=1.0, theta=0.00001):
    """
    Evaluates a policy and computes its state values given an environment
    and a full description of that environment (an Markov Decision Process).
    The environment should be a subclass from the OpenAI Gym environments.
    :param env: The OpenAI Gym environment:
                 - env.P - transition probabilities of the environment
                 - env.P[state][action] - a list of transition tuples
                 - env.observation_space.n - number of states
                 - env.action_space.n - number of actions
    :param policy: the policy to be evaluated (if None, initialize an equidistant policy)
    :param discount_factor: gamma the discount factor
    :param theta: we stop iterating once the state value changes are less than theta
    :return: the state values computes
    """
    if policy is None:
        raise ValueError('No policy to evaluate')
    # initialize the state values
    state_values = np.zeros(env.observation_space.n, dtype=np.float64)
    # initialize delta check
    delta = theta + 1.0
    while delta > theta:
        delta = 0
        # loop over all the states
        for state in env.P.keys():
            value = state_values[state]
            state_values[state] = 0.0
            # loop over all the actions
            for action, action_prob in zip(env.P[state].keys(), policy[state]):
                # loop over all possible new states
                for prob, new_state, reward, _ in env.P[state][action]:
                    # calculate expected value using Bellman equation
                    state_values[state] += action_prob * prob *\
                                           (reward + discount_factor * state_values[new_state])
            delta = np.max((delta, np.abs(value - state_values[state])))
    return state_values


def policy_improvement(env, policy, state_values, discount_factor=1.0):
    """
    Improves the policy based on the state values calculated using policy evaluation.
    :param env: The OpenAI Gym environment:
                 - env.P - transition probabilities of the environment
                 - env.P[state][action] - a list of transition tuples
                 - env.observation_space.n - number of states
                 - env.action_space.n - number of actions
    :param policy: the policy to be evaluated
    :param state_values: the state values used in updating the policy
    :param discount_factor: gamma the discount factor
    :return: True if the policy is stable, and False otherwise.
    """
    # we assume that the policy is stable
    policy_stable = True
    # loop over each state
    for state in env.P.keys():
        # get the previous best action
        old_best_action = np.argmax(policy[state])
        # compute action values
        action_values = get_action_values(state, env, state_values, discount_factor)
        # get the new best action
        new_best_action = np.argmax(action_values)
        if old_best_action != new_best_action:
            # if the actions are different, then the policy isn't stable
            policy_stable = False
        # update the policy of the state
        policy[state] = np.eye(len(env.P[state]))[new_best_action]
    return policy_stable


def get_action_values(s, env, state_values, discount_factor=1.0):
    """
    Helper function used to look for the best action from state s.
    It returns the action values, and those can be subsequently
    used to determine the best action.
    :param s: the state for which we want to compute the action values
    :param env: The OpenAI Gym environment:
                 - env.P - transition probabilities of the environment
                 - env.P[state][action] - a list of transition tuples
                 - env.observation_space.n - number of states
                 - env.action_space.n - number of actions
    :param state_values: the state values used in updating the policy
    :param discount_factor: gamma the discount factor
    :return: the action values
    """
    action_vals = np.zeros(len(env.P[s]))
    for a in env.P[s].keys():
        for prob, next_state, reward, _ in env.P[s][a]:
            action_vals[a] += prob * (reward + discount_factor * state_values[next_state])
    return action_vals
