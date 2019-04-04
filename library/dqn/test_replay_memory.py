import unittest
import numpy as np
from library.dqn.replay_memory import ReplayMemory, FrameQueue


class TestReplayMemory(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.RandomState(42)
        self.memory = ReplayMemory(84, 84, rng=rng, buffer_size=10)
        # initialize it with 8 samples
        for _ in range(8):
            state = np.random.random((84, 84, 4))
            action = np.random.randint(4)
            reward = np.random.random()
            next_state = np.random.random((84, 84, 4))
            done = False
            self.memory.add_sample(state, action, reward, next_state, done)

    def test_add_sample(self):
        self.assertEqual(len(self.memory), 8)

    def test_add_sample_wrap(self):
        for _ in range(4):
            state = np.random.random((84, 84, 4))
            action = np.random.randint(4)
            reward = np.random.random()
            next_state = np.random.random((84, 84, 4))
            done = False
            self.memory.add_sample(state, action, reward, next_state, done)
        self.assertEqual(len(self.memory), 10)
        self.assertEqual(self.memory.bottom, 2)
        self.assertEqual(self.memory.top, 2)

    def test_sample_minibatches_smaller_than_size(self):
        for _ in range(4):
            state = np.random.random((84, 84, 4))
            action = np.random.randint(4)
            reward = np.random.random()
            next_state = np.random.random((84, 84, 4))
            done = False
            self.memory.add_sample(state, action, reward, next_state, done)
        self.assertEqual(len(self.memory), 10)
        states, actions, rewards, next_states, done = self.memory.sample_minibatches(8)
        self.assertEqual(states.shape, (8, 84, 84, 4))
        self.assertEqual(actions.shape, (8, ))
        self.assertEqual(rewards.shape, (8,))
        self.assertEqual(next_states.shape, (8, 84, 84, 4))
        self.assertEqual(done.shape, (8,))

    def test_samples_minibatches_bigger_than_size(self):
        for _ in range(4):
            state = np.random.random((84, 84, 4))
            action = np.random.randint(4)
            reward = np.random.random()
            next_state = np.random.random((84, 84, 4))
            done = False
            self.memory.add_sample(state, action, reward, next_state, done)
        self.assertEqual(len(self.memory), 10)
        states, actions, rewards, next_states, done = self.memory.sample_minibatches(15)
        self.assertEqual(states.shape, (15, 84, 84, 4))
        self.assertEqual(actions.shape, (15, ))
        self.assertEqual(rewards.shape, (15,))
        self.assertEqual(next_states.shape, (15, 84, 84, 4))
        self.assertEqual(done.shape, (15,))


class TestFrameQueue(unittest.TestCase):
    def setUp(self) -> None:
        self.queue = FrameQueue(84, 84, 4)
        self.frame1 = np.ones((84, 84), dtype=np.float)
        self.frame2 = np.ones((84, 84), dtype=np.float) * 2.0
        self.frame3 = np.ones((84, 84), dtype=np.float) * 3.0
        self.frame4 = np.ones((84, 84), dtype=np.float) * 4.0
        self.frame5 = np.ones((84, 84), dtype=np.float) * 5.0
        # initialize it with 3 samples
        self.queue.append(frame=self.frame1)
        self.queue.append(frame=self.frame2)
        self.queue.append(frame=self.frame3)

    def test_add_sample(self):
        self.queue.append(self.frame4)
        self.assertTrue(np.array_equal(self.queue.queue[0, 0],
                                       np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float)))
        self.queue.append(self.frame5)
        self.assertTrue(np.array_equal(self.queue.queue[0, 0],
                                       np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float)))
