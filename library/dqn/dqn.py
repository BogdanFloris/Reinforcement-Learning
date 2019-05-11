"""
DQN implementation that replicates the results from the seminal paper
Human-level control through deep reinforcement learning.
"""
import os
import sys
import itertools
from datetime import datetime
import tensorflow as tf
import numpy as np
from library.estimators.q_network import QNetwork
from library.dqn.replay_memory import ReplayMemory
from library.utils import make_epsilon_greedy_policy, generate_gif
from library.atari import AtariWrapper
from library.plotting import EpisodeStats


class DQNAgent:
    def __init__(self,
                 env_name,
                 num_episodes=10000,
                 seed=42,
                 experiment_dir=None,
                 input_shape=(84, 84, 4),
                 buffer_size=100000,
                 init_buffer=10000,
                 batch_size=32,
                 m=4,
                 update_frequency=4,
                 target_update_frequency=10000,
                 eval_frequency=10000,
                 eval_steps=10000,
                 learning_rate=0.00001,
                 discount_factor=0.99,
                 initial_epsilon=1.0,
                 final_epsilon=0.1,
                 eps_decay_steps=1000000,
                 max_episode_length=18000,
                 print_every=1,
                 train=False):
        """
        Performs the initialization of the DQN Agent:
         - copies the given parameters
         - initializes the directories for the checkpoints and the video
         - initializes the checkpoint and restores if necessary
         - initializes the networks: Q and target and their optimizer
         - initializes the replay memory and populates it
        :param env_name: the environment on which the agent performs on
        :param num_episodes: the number of episodes to train the agent for
        :param seed: the random generator seed
        :param experiment_dir: where to save the experiments;
            can be left None and it will be initialized automatically
        :param input_shape: the size of the input that is fed to the Q network
        :param buffer_size: size of the replay memory buffer
        :param init_buffer: how many samples should initially be in the replay memory
        :param batch_size: how many samples to use for a training session
        :param m: number of frames in the input
        :param update_frequency: how often to perform a gradient descent update
        :param target_update_frequency: how often to update the target network
        :param eval_frequency: number of frames between evaluations
        :param eval_steps: number of frames in the evaluation
        :param learning_rate: the learning rate for the optimizer
        :param discount_factor: discount factor used to calculate the targets
        :param initial_epsilon: initial epsilon value
        :param final_epsilon: final epsilon value
        :param eps_decay_steps: how many epsilons between initial and final
        :param max_episode_length: the maximum number of episode steps (5 minutes)
        :param print_every: how often we print during training
        :param train: whether we are training the agent or not
        """
        # initialize parameters
        self.num_episodes = num_episodes
        self.rng = np.random.RandomState(seed)
        self.env = AtariWrapper(env_name, rng=self.rng)
        self.no_actions = self.env.env.action_space.n
        self.input_shape = input_shape
        self.buffer_size = buffer_size
        self.init_buffer = init_buffer
        self.batch_size = batch_size
        self.m = m
        self.update_frequency = update_frequency
        self.target_update_frequency = target_update_frequency
        self.eval_frequency = eval_frequency
        self.eval_steps = eval_steps
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.eps_decay_steps = eps_decay_steps
        self.max_episode_length = max_episode_length
        self.print_every = print_every

        # initialize the directories
        if experiment_dir is None:
            experiment_dir = os.path.abspath('./experiments/{}'.format(self.env.env.spec.id))
        self.checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
        self.video_dir = os.path.join(experiment_dir, 'video')
        self.weights_dir = os.path.join(experiment_dir, 'weights')
        print(str(self.video_dir))
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
                                  no_actions=self.no_actions,
                                  rng=self.rng)
        self.target_q_network = QNetwork(input_shape=input_shape,
                                         no_actions=self.no_actions,
                                         rng=self.rng)
        # initialize optimizer and loss function
        self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)
        self.loss_fn = tf.keras.losses.Huber()

        # initialize checkpoint
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(0),
                                        optimizer=self.optimizer,
                                        net=self.q_network)
        self.manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_dir, max_to_keep=3)
        if not train:
            self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint and not train:
            print('Restored checkpoint from {}'.format(self.manager.latest_checkpoint))
        else:
            print('Initializing Q network from scratch')

        # initialize replay memory
        self.memory = ReplayMemory(self.input_shape[0],
                                   self.input_shape[1],
                                   rng=self.rng,
                                   buffer_size=buffer_size,
                                   m=m,
                                   batch_size=batch_size)

        # make a list of all the epsilons used
        self.epsilons = np.linspace(initial_epsilon, final_epsilon, eps_decay_steps)
        # initialize the policy
        # if epsilon is not given, it's going to default to 0
        # !Thus, always use it like this: policy(state, epsilon) in training
        self.policy = make_epsilon_greedy_policy(self.no_actions,
                                                 epsilon=0.0,
                                                 estimator=self.q_network,
                                                 distribute_prob=False)

    def dqn(self, num_episodes=None):
        """
        Perform the Deep Q-Learning algorithm
        :param num_episodes: number of episodes to perform dqn for
        """
        if num_episodes is None:
            num_episodes = self.num_episodes

        # Train the DQN agent
        # -------------------
        loss_list = []
        print('Training...')
        # loop over episodes
        for i_episode in range(num_episodes):

            # reset the environment
            _ = self.env.reset()
            # initialize loss
            loss = None

            # execute a episode
            for t in itertools.count():

                # get the probabilities of the actions
                action_probs = self.policy(tf.expand_dims(self.env.state, axis=0),
                                           self.get_epsilon())
                # choose an action given the probabilities
                action = np.random.choice(tf.range(len(action_probs)), p=action_probs)

                # take a step in the environment
                processed_frame, reward, done, life_lost, _ = self.env.step(action)

                # add sample to memory
                self.memory.add_sample(processed_frame[:, :, 0], action, reward, life_lost)

                # update statistics
                self.stats.episode_rewards[i_episode] += reward
                self.stats.episode_lengths[i_episode] = t

                # perform a training update if we hit the update frequency
                if int(self.ckpt.step) % self.update_frequency == 0\
                        and int(self.ckpt.step) > self.init_buffer:
                    loss = self.q_update()
                    loss_list.append(loss.numpy())

                # update target network if we hit the update frequency
                if int(self.ckpt.step) % self.target_update_frequency == 0:
                    self.update_target_network()

                # evaluate the model so far if we hit the evaluation frequency
                if int(self.ckpt.step) % self.eval_frequency == 0:
                    self.play()

                # update checkpoint step
                self.ckpt.step.assign_add(1)

                # check done
                if done:
                    break
                # check max episode number
                if t == self.max_episode_length:
                    break

            # save model and print statistics for debugging
            if i_episode % self.print_every == 0:
                if loss is not None:
                    self.manager.save()
                    print("[{}] Episode {}/{}: loss {:.10f}, reward: {}, episode length: {}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"),
                        i_episode,
                        num_episodes,
                        loss,
                        self.stats.episode_rewards[i_episode],
                        self.stats.episode_lengths[i_episode]))
                else:
                    print('[{}] Initializing replay memory: Episode {}/{}, reward: {}, episode length: {}'.format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"),
                        i_episode,
                        num_episodes,
                        self.stats.episode_rewards[i_episode],
                        self.stats.episode_lengths[i_episode]
                    ))

        print('Finished training')
        print('Saving weights...')
        self.q_network.save_weights(self.weights_dir)
        print('Done')

    def q_update(self):
        """
        Performs a Q update on the DQN network.
        This update uses Double-Q learning.
        :return: the loss
        """
        # sample a mini batch from the memory
        states, actions, rewards, new_states, done = self.memory.sample_minibatch()

        # get the Q values for the new states using the main network
        q_values_new_main = self.q_network.predict(new_states)
        # get the best action for each batch
        best_actions = tf.argmax(q_values_new_main, axis=1)

        # get the Q values for the new states using the target network
        q_values_new_target = self.target_q_network.predict(new_states)
        # get the double Q values
        double_q = q_values_new_target[np.arange(self.batch_size), best_actions]
        # compute the target Q values
        target_q = rewards + (self.discount_factor * double_q * (1 - done))
        # one hot vector
        one_hot = tf.one_hot(actions, self.no_actions, dtype=tf.float32)

        with tf.GradientTape() as tape:
            # predict the Q values for the states using the main network
            q_values = self.q_network(states)
            # compute the q value predictions using the actions taken
            q = tf.reduce_sum(tf.multiply(q_values, one_hot), axis=1)

            # compute the loss
            loss = self.loss_fn(y_true=target_q, y_pred=q)

        # compute the gradients using the tape
        gradients = tape.gradient(loss, self.q_network.trainable_variables)

        # apply the gradients using the optimizer
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))

        return loss

    def play(self, no_games=1):
        """
        Play a number of games using the learned policy
        and produce GIFs for each game.
        """
        done = True
        life_lost = True
        reward_sum = 0
        frames = []
        print('Evaluation...')

        for _ in range(self.eval_steps):

            if done:
                life_lost = self.env.reset(evaluate=True)

            # get an action
            if life_lost:
                # Always play 1 at first
                # NOTE: will probably need to change this for other games
                action = 1
            else:
                # get the probabilities of the actions (epsilon is 0)
                action_probs = self.policy(tf.expand_dims(self.env.state, axis=0))
                # choose an action given the probabilities
                action = np.random.choice(tf.range(len(action_probs)), p=action_probs)

            # take the step
            processed_frame, reward, done, life_lost, frame = self.env.step(action)

            # update the reward
            reward_sum += reward
            # add frame
            frames.append(frame)

            # if the game is truly finished make the gif and reset everything
            if self.env.lives == 0:
                print('Making GIF...')
                generate_gif(frames, reward_sum, self.video_dir,
                             number=int(self.ckpt.step), evaluation=True)
                break

    def get_epsilon(self):
        """
        :return: epsilon value to be used
        """
        return self.epsilons[min(int(self.ckpt.step), self.eps_decay_steps - 1)]

    def update_target_network(self):
        """
        Updates the parameters of the target network
        with the parameters of the Q network.
        """
        print('Updating Target Network...')
        self.target_q_network.set_weights(self.q_network.get_weights())
