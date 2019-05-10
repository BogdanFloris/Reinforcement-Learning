"""
Contains the Q Network TensorFlow model used in DQN
"""
import tensorflow as tf
from numpy.random import RandomState


class QNetwork(tf.keras.Model):
    """
    QNetwork is the deep neural network used
    Implements the Dueling Architecture described in Wang et al. 2016.
    """
    def __init__(self, input_shape, no_actions, rng: RandomState,
                 n_hidden=1024, name='q_network', *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.no_actions = no_actions
        self.n_hidden = n_hidden
        # set random seed
        tf.random.set_seed(rng.seed())
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=8,
            strides=4,
            activation='relu',
            kernel_initializer=tf.initializers.VarianceScaling(scale=0.2),
            use_bias=False,
            input_shape=input_shape
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=4,
            strides=2,
            activation='relu',
            use_bias=False,
            kernel_initializer=tf.initializers.VarianceScaling(scale=0.2)
        )
        self.conv3 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            activation='relu',
            use_bias=False,
            kernel_initializer=tf.initializers.VarianceScaling(scale=0.2)
        )
        self.conv4 = tf.keras.layers.Conv2D(
            filters=self.n_hidden,
            kernel_size=7,
            strides=1,
            activation='relu',
            use_bias=False,
            kernel_initializer=tf.initializers.VarianceScaling(scale=0.2)
        )

        # flatten layers for both value and advantage streams
        self.flatten_value = tf.keras.layers.Flatten()
        self.flatten_advantage = tf.keras.layers.Flatten()

        # final dense layers for both value and advantage streams
        self.dense_value = tf.keras.layers.Dense(
            units=1,
            kernel_initializer=tf.initializers.VarianceScaling(scale=0.2)
        )
        self.dense_advantage = tf.keras.layers.Dense(
            units=self.no_actions,
            kernel_initializer=tf.initializers.VarianceScaling(scale=0.2)
        )

    def call(self, inputs):
        # apply convolution layers
        output = self.conv1(inputs)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        # split the last conv layer into value and advantage streams
        value_stream, adv_stream = tf.split(output, 2, axis=3)
        # flatten the value and advantage streams
        value_stream = self.flatten_value(value_stream)
        adv_stream = self.flatten_advantage(adv_stream)
        # apply the two dense layers
        value = self.dense_value(value_stream)
        advantage = self.dense_advantage(adv_stream)
        # combine value and advantage into q-values as described in Wang et al. 2016
        q_values = value + tf.subtract(advantage, tf.reduce_mean(
            advantage, axis=1, keepdims=True))
        return q_values
