import unittest
import tensorflow as tf
from numpy.random import RandomState, random, randint
from library.estimators.q_network import QNetwork


class TestQNetwork(unittest.TestCase):
    def setUp(self):
        self.batch = 10
        self.model = QNetwork((84, 84, 4), 4, RandomState(42))
        self.optimizer = tf.keras.optimizers.RMSprop()
        self.input = random((self.batch, 84, 84, 4))

    def test_predict(self):
        predictions = self.model.predict(self.input)
        self.assertEqual(predictions.shape, (self.batch, 4))

    def test_train(self):
        targets = random(self.batch)
        actions = randint(0, 4, self.batch)
        self.model.train_batch(self.input, actions, targets, self.optimizer)
