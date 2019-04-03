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

    def call(self, inputs, actions=None, training=None):
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
        if training:
            if actions is None:
                raise ValueError('Argument actions cannot be None in training')
            # if we are not training, then we can leave the output as is
            # if we are training, we need to return only the output
            # that corresponds to the played action
            indices = tf.range(output.shape[0]) * output.shape[1] + actions
            output = tf.gather(tf.reshape(output, [-1]), indices)
        return output

    def train_batch(self, inputs, actions, targets, optimizer):
        """
        Trains one batch.
        :param inputs: of shape (batch_size, 84, 84, 4)
        :param actions: of shape (batch_size)
        :param targets: of shape (batch_size)
        :param optimizer: the optimizer used to train
        :return:
        """
        mse_loss = tf.keras.losses.MeanSquaredError()
        loss_metric = tf.keras.metrics.Mean()

        print('Performing update on Q-Network...', sep='\t')
        with tf.GradientTape() as tape:
            # compute the output
            outputs = self.call(inputs, actions=actions, training=True)
            # compute the loss between the output and the target
            loss = mse_loss(outputs, targets)
        # compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        # apply gradients using the optimizer
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # get the loss metric
        loss_metric(loss)
        print('mean loss = {}'.format(loss_metric.result()))
