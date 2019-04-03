"""
DQN implementation that replicates the results from the seminal paper
Human-level control through deep reinforcement learning.
TODO: 1. The input to the Q Network is the last M processed frames
TODO: 2. Use deque to store the last M processed frames
TODO: 3. Divide by 255 somewhere
"""
import os
import tensorflow as tf
from collections import deque
from library.estimators.q_network import QNetwork


# parameters from the paper with a few minor adjustments
BUFFER_SIZE = 500000
INIT_BUFFER = 50000
BATCH_SIZE = 32
DISCOUNT_FACTOR = 0.99
M = 4
UPDATE_FREQUENCY = 10000
LR = 0.00025
INITIAL_EPS = 1.0
FINAL_EPS = 0.1
EPS_DECAY_STEPS = 500000


class DQNAgent:
    def __init__(self):
        pass


def process_atari_frame(frame):
    """
    Transforms the given atari frame by applying the following steps:
     - transform the frame to grayscale
     - crop the frame to remove the score to (160, 160)
     - resize the image to (84, 84)
    :param frame: the atari frame of shape (210, 160, 3)
    :return: the processed frame of size (84, 84)
    """
    output = tf.image.rgb_to_grayscale(frame)
    output = tf.image.crop_to_bounding_box(output, 34, 0, 160, 160)
    output = tf.image.resize(output, (84, 84))
    output = tf.squeeze(output)
    return output
