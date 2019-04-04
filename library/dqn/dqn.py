"""
DQN implementation that replicates the results from the seminal paper
Human-level control through deep reinforcement learning.
"""
import os
import sys
sys.path.append('../')
import itertools
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from gym import wrappers
from PIL import Image
from library.estimators.q_network import QNetwork
from library.dqn.replay_memory import ReplayMemory, FrameQueue
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
        self.env = env
        self.num_episodes = num_episodes
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
        self.weights_dir = os.path.join(experiment_dir, 'weights')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.video_dir):
            os.makedirs(self.video_dir)
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)
        # initialize episode statistics
        self.stats = EpisodeStats(
            episode_lengths=np.zeros(self.num_episodes),
            episode_rewards=np.zeros(self.num_episodes)
        )
        # initialize networks
        self.q_network = QNetwork(input_shape=input_shape,
                                  no_actions=env.action_space.n,
                                  rng=self.rng)
        self.target_q_network = QNetwork(input_shape=input_shape,
                                         no_actions=env.action_space.n,
                                         rng=self.rng)
        # initialize optimizer
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        # initialize checkpoint
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                                        optimizer=self.optimizer,
                                        net=self.q_network)
        self.manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_dir, max_to_keep=3)
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print('Restored from {}'.format(self.manager.latest_checkpoint))
        else:
            print('Initializing Q network from scratch')
        # initialize replay memory
        self.memory = ReplayMemory(self.input_shape[0],
                                   self.input_shape[1],
                                   rng=self.rng,
                                   buffer_size=buffer_size)
        # initialize last m frames queue
        self.frame_queue = FrameQueue(self.input_shape[0],
                                      self.input_shape[1],
                                      m=self.m)
        # make a list of all the epsilons used
        self.epsilons = np.linspace(initial_epsilon, final_epsilon, eps_decay_steps)
        # initialize the policy
        # !Note: always use it like this: policy(state, epsilon)
        # if epsilon is not given, it's going to default to initial_epsilon
        self.policy = make_epsilon_greedy_policy(env.action_space.n,
                                                 epsilon=self.initial_epsilon,
                                                 estimator=self.q_network,
                                                 distribute_prob=False)

    def dqn(self, num_episodes=None):
        """
        Perform the Deep Q-Learning algorithm
        :param num_episodes: number of episodes to perform dqn for
        """
        if num_episodes is None:
            num_episodes = self.num_episodes
        # populate initial replay memory
        self.populate_init_replay_memory()
        print('Starting training...')
        # loop over episodes
        for i_episode in tqdm(range(num_episodes)):
            # get 4 initial frames in the queue
            frame = process_atari_frame(self.env.reset())
            for _ in range(self.m):
                self.frame_queue.append(frame)
            state = self.frame_queue.get_queue()
            # initialize loss
            loss = None
            for t in itertools.count():
                # update target network
                if int(self.ckpt.step) % self.update_frequency:
                    self.update_target_network()
                # take a step using the action given by the policy
                action_probs = self.policy(np.expand_dims(state, 0), self.get_epsilon())
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                frame, reward, done, _ = self.env.step(action)
                frame = process_atari_frame(frame)
                self.frame_queue.append(frame)
                next_state = self.frame_queue.get_queue()
                # add the sample to memory
                self.memory.add_sample(state, action, reward, next_state, done)
                # update statistics
                self.stats.episode_rewards[i_episode] += reward
                self.stats.episode_lengths[i_episode] = t
                # sample batch
                states_batch, actions_batch, rewards_batch, next_states_batch, done_batch = \
                    self.memory.sample_minibatches(self.batch_size)
                # perform Q update
                q_values_next = self.target_q_network.predict(next_states_batch)
                targets_batch = rewards_batch + np.invert(done_batch).astype(
                    np.float) * self.discount_factor * np.amax(q_values_next, axis=1)
                loss = self.q_network.train_batch(states_batch,
                                                  actions_batch,
                                                  targets_batch,
                                                  self.optimizer)
                # check done
                if done:
                    break
                # update state
                state = next_state
                # update checkpoint step
                self.ckpt.step.assign_add(1)
            # print loss and save
            if i_episode % 10 == 0:
                self.manager.save()
                print("loss {:1.2f} at episode {}".format(loss.numpy(), i_episode))
        print('Finished training')
        print('Saving weights...')
        self.q_network.save_weights(self.weights_dir)
        print('Done')

    def populate_init_replay_memory(self):
        """
        Called at the beginning of learning in order to populate
        the replay memory with an initial self.init_buffer experiences.
        """
        print('Populating memory with initial experience...')
        # get 4 initial frames in the queue
        frame = process_atari_frame(self.env.reset())
        for _ in range(self.m):
            self.frame_queue.append(frame)
        state = self.frame_queue.get_queue()
        # loop until we have populated the memory with init_buffer samples
        for _ in tqdm(range(self.init_buffer)):
            # take a step using the action given by the policy
            action_probs = self.policy(np.expand_dims(state, 0), self.get_epsilon())
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            frame, reward, done, _ = self.env.step(action)
            frame = process_atari_frame(frame)
            self.frame_queue.append(frame)
            next_state = self.frame_queue.get_queue()
            # add the sample to memory
            self.memory.add_sample(state, action, reward, next_state, done)
            # reset environment if done
            if done:
                frame = process_atari_frame(self.env.reset())
                for _ in range(self.m):
                    self.frame_queue.append(frame)
                state = self.frame_queue.get_queue()
            # else continue
            else:
                state = next_state

    def get_epsilon(self):
        """
        :return: epsilon value to be used
        """
        return self.epsilons[min(int(self.ckpt.step),
                                 self.eps_decay_steps - 1)]

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
