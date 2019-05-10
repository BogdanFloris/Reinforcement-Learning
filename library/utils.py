"""
Utilities module
"""
import imageio
import numpy as np
from collections import defaultdict
from skimage.transform import resize


def make_random_policy(action_count):
    """
    Creates a random policy based on the number of possible actions
    :param action_count: number of actions
    :return: policy function
    """
    actions = np.ones(action_count, dtype=float) / action_count

    def policy_func(observation):
        return actions

    return policy_func


def make_epsilon_greedy_policy(action_count: int, epsilon=0.0, q: dict = None,
                               estimator=None, distribute_prob=True):
    """
    This function creates an epsilon greedy policy based on the given Q.
    :param action_count: Number of actions
    :param epsilon: Probability to select a random action
    :param q: A dictionary that maps from a state to the action values
              for all possible nA actions (represented as an array)
    :param estimator: If q is None, then we use an estimator to predict the q function
    :param distribute_prob: Whether or not to distribute the probability between best actions
                            or just choose the first best action an assign it all the probability mass.
    :return: A function that takes as argument an observation and returns
             the probabilities of each action.
    """
    if q is None and estimator is None:
        raise ValueError('Cannot make policy: both q and estimator are none')

    def policy_func(observation, eps=epsilon):
        actions = np.ones(action_count, dtype=float) * eps / action_count
        if q is not None:
            q_values = q[observation]
        else:
            q_values = estimator.predict(observation)
        if distribute_prob:
            best_actions = np.argwhere(q_values == np.max(q_values)).flatten()
            for i in best_actions:
                actions[i] += (1.0 - eps) / len(best_actions)
        else:
            best_action = np.argmax(q_values)
            actions[best_action] += (1.0 - eps)
        return actions

    return policy_func


def print_q(q):
    """
    Prints Q dictionary. Used for debugging
    :param q: The Q action value dictionary
    """
    for key in sorted(q.keys()):
        print(key, end=" ")
        value = q[key]
        for i in range(len(value)):
            print(value[i], end=" ")
        print()


def create_state_value_function(q):
    """
    Creates a state value function from a q function.
    :param q: the q function
    :return: the state value function
    """
    state_values = defaultdict(float)
    for state, action_values in q.items():
        state_values[state] = np.max(action_values)
    return state_values


def generate_gif(frames, reward, path, number=None, evaluation=False):
    """
    Creates a GIF from the given frames.
    :param frames: sequence of (210, 160, 3) Atari frames
    :param reward: total reward for the sequence
    :param path: where to save the GIF
    :param number: index used for saving
    :param evaluation: whether we are evaluating or just playing
    """
    for i, frame in enumerate(frames):
        frames[i] = resize(frame, (420, 320, 3),
                           order=0, preserve_range=True).astype(np.uint8)
    if evaluation:
        path += '/atari-step-{}-reward-{}.gif'.format(number, reward)
    else:
        path += '/atari-play-reward-{}.gif'.format(reward)
    imageio.mimsave(path, frames, duration=1/30)
