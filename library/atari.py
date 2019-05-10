"""
Atari Wrapper
"""
import gym
import numpy as np
import tensorflow as tf


class AtariWrapper:
    """
    Class that wraps an OpenAI Gym environment
    """
    def __init__(self, env_name, rng, no_steps=10, m=4):
        self.env = gym.make(env_name)
        self.rng = rng
        self.state = None
        self.lives = 0
        self.no_steps = no_steps
        self.m = m

    def reset(self, evaluate=False):
        """
        Resets the environment
        :param evaluate: evaluation or not
        """
        frame = self.env.reset()
        self.lives = 0

        # this is set to True so that the agent
        # fires the first time we reset the environment
        life_lost = True

        # if we are evaluating
        if evaluate:
            for _ in range(self.rng.randint(1, self.no_steps + 1)):
                frame, _, _, _ = self.env.step(1)
        # process frame
        processed_frame = process_atari_frame(frame)
        self.state = np.repeat(processed_frame, self.m, axis=2)

        return life_lost

    def step(self, action):
        """
        Takes a step in the environment using the action
        :param action: the action to take
        :return: the new processed frame,
                 reward,
                 done,
                 terminal life lost
                 the new frame
        """
        new_frame, reward, done, info = self.env.step(action)

        # determine lives
        if info['ale.lives'] < self.lives:
            life_lost = True
        else:
            life_lost = done
        self.lives = info['ale.lives']

        # process frame and append it to state
        processed_frame = process_atari_frame(new_frame)
        self.state = np.append(self.state[:, :, 1:], processed_frame, axis=2)

        return processed_frame, reward, done, life_lost, new_frame


def process_atari_frame(frame):
    """
    Transforms the given atari frame by applying the following steps:
     - transform the frame to grayscale
     - crop the frame to remove the score to (160, 160)
     - resize the image to (84, 84)
    :param frame: the atari frame of shape (210, 160, 3)
    :return: the processed frame of size (84, 84, 1)
    """
    output = tf.image.rgb_to_grayscale(frame)
    output = tf.image.crop_to_bounding_box(output, 34, 0, 160, 160)
    output = tf.image.resize(output, (84, 84), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return output
