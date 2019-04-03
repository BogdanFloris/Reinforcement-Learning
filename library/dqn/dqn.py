"""
DQN implementation that replicates the results from the seminal paper
Human-level control through deep reinforcement learning.
TODO: 1. The input to the Q Network is the last M processed frames
TODO: 2. Use deque to store the last M processed frames
TODO: 3. Divide by 255 somewhere
"""
import os
import tensorflow as tf
import numpy as np
from collections import deque
from library.estimators.q_network import QNetwork
from library.dqn.replay_memory import ReplayMemory
from library.utils import make_epsilon_greedy_policy
from library.plotting import EpisodeStats


class DQNAgent:
    def __init__(self,
                 env,
                 num_episodes,
                 seed=42,
                 experiment_dir=None,
                 input_shape=(84, 84, 4),
                 buffer_size=500000,
                 init_buffer=50000,
                 batch_size=32,
                 m=4,
                 update_frequency=10000,
                 learning_rate=0.00025,
                 discount_factor=0.99,
                 initial_epsilon=1.0,
                 final_epsilon=0.1,
                 eps_decay_steps=500000):
        """
        Performs the initialization of the DQN Agent:
         - copies the given parameters
         - initializes the directories for the checkpoints and the video
         - initializes the checkpoint and restores if necessary
         - initializes the networks: Q and target and their optimizer
         - initializes the replay memory and populates it
        :param env: the environment on which the agent performs on
        :param num_episodes: the number of episodes to train the agent for
        :param seed: the random generator seed
        :param experiment_dir: where to save the experiments;
            can be left None and it will be initialized automatically
        :param input_shape: the size of the input that is fed to the Q network
        :param buffer_size: size of the replay memory buffer
        :param init_buffer: how many samples should initially be in the replay memory
        :param batch_size: how many samples to use for a training session
        :param m: number of frames in the input
        :param update_frequency: how often to update the target network
        :param learning_rate: the learning rate for the optimizer
        :param discount_factor: discount factor used to calculate the targets
        :param initial_epsilon: initial epsilon value
        :param final_epsilon: final epsilon value
        :param eps_decay_steps: how many epsilons between initial and final
        """
        # initialize parameters
        self.rng = np.random.RandomState(seed)
        self.input_shape = input_shape
        self.buffer_size = buffer_size
        self.init_buffer = init_buffer
        self.batch_size = batch_size
        self.m = m
        self.update_frequency = update_frequency
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.eps_decay_steps = eps_decay_steps
        # initialize the directories
        if experiment_dir is None:
            experiment_dir = os.path.abspath('./experiments/{}'.format(env.spec.id))
        self.checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
        self.video_dir = os.path.join(experiment_dir, 'video')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.video_dir):
            os.makedirs(self.video_dir)
        # initialize checkpoint
        ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                                   optimizer=self.optimizer,
                                   net=self.q_network)
        manager = tf.train.CheckpointManager(ckpt, self.checkpoint_dir, max_to_keep=3)
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print('Restored from {}'.format(manager.latest_checkpoint))
        else:
            print('Initializing Q network from scratch')
        # initialize episode statistics
        self.stats = EpisodeStats(
            episode_lengths=np.zeros(num_episodes),
            episode_rewards=np.zeros(num_episodes)
        )
        # initialize networks
        self.q_network = QNetwork(input_shape=input_shape,
                                  no_actions=env.action_space.n,
                                  rng=self.rng)
        self.target_q_network = QNetwork(input_shape=input_shape,
                                         no_actions=env.action_space.n,
                                         rng=self.rng)
        # initialize optimizer
        self.optimizer = tf.keras.optimizers.RMSProp(learning_rate=learning_rate)
        # initialize replay memory
        self.memory = ReplayMemory(self.input_shape[0],
                                   self.input_shape[1],
                                   rng=self.rng,
                                   buffer_size=buffer_size)
        # initialize last m frames queue
        self.last_m_frames = deque(maxlen=self.m)
        # make a list of all the epsilons used
        self.epsilons = np.linspace(initial_epsilon, final_epsilon, eps_decay_steps)
        # initialize the policy
        # !Note: always use it like this: policy(state, epsilon)
        # if epsilon is not given, it's going to default to initial_epsilon
        self.policy = make_epsilon_greedy_policy(env.action_space.n,
                                                 epsilon=self.initial_epsilon,
                                                 estimator=self.q_network)
        self.populate_init_replay_memory()

    def populate_init_replay_memory(self):
        """
        Called at the beginning of learning in order to populate
        the replay memory with an initial self.init_buffer experiences.
        """
        pass

    def update_target_network(self):
        """
        Updates the parameters of the target network
        with the parameters of the Q network.
        """
        self.target_q_network.set_weights(self.q_network.get_weights())


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
