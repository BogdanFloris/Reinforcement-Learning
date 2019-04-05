import unittest
import tensorflow as tf
from numpy.random import RandomState, random, randint
from library.estimators.q_network import QNetwork


class TestQNetwork(unittest.TestCase):
    def setUp(self):
        self.batch = 32
        self.model = QNetwork((84, 84, 4), 4, RandomState(42))
        self.optimizer = tf.keras.optimizers.RMSprop()
        self.model.compile(optimizer=self.optimizer,
                           loss='mse')
        self.input = random((self.batch, 84, 84, 4))

    def test_predict(self):
        predictions = self.model.predict(self.input)
        self.assertEqual(predictions.shape, (self.batch, 4))

    def test_train(self):
        targets = random((self.batch, 4))
        loss = self.model.train_on_batch(self.input, targets)
        self.assertTrue(type(loss), float)
