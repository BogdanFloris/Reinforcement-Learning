"""
Contains the Q Network TensorFlow model used in DQN
"""
import tensorflow as tf
from numpy.random import RandomState


class QNetwork(tf.keras.Model):
    """
    QNetwork is the deep neural network used
    """
    def __init__(self, input_shape, no_actions, rng: RandomState, name='q_network', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.no_actions = no_actions
        # set random seed
        tf.random.set_seed(rng.seed())
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=8,
            strides=4,
            activation='relu',
            input_shape=input_shape
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=4,
            strides=2,
            activation='relu'
        )
        self.conv3 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            activation='relu'
        )
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dense2 = tf.keras.layers.Dense(self.no_actions)

    def call(self, inputs):
        # first scale to 0..1
        output = tf.cast(inputs, dtype=tf.float32) / 255.0
        # apply convolution layers
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.conv3(output)
        # flatten
        output = self.flatten(output)
        # apply the two dense layers
        output = self.dense1(output)
        output = self.dense2(output)
        return output
