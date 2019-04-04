"""
Script used to train an agent on the Mountain Cart environment
and then produce a video of that agent solving the problem.
"""
import sys
sys.path.append('/Users/bogdanfloris/Workspace/Code/Reinforcement-Learning/')
import gym
import numpy as np
import io
import pybase64
from gym import wrappers
from IPython.display import HTML
from library.td_learning import temporal_diff_learning as td
from library.estimators.estimators import QFunctionSGD
from library.utils import make_epsilon_greedy_policy
# TODO: adjust directory to new folder structure (with experiments folder)

# initialize environment
env = gym.make('MountainCar-v0')

# initialize estimator for Q function
estimator = QFunctionSGD(env)

# train the estimator
epochs = 100
_, _ = td.q_learning(env, num_episodes=epochs, estimator=estimator, epsilon=0.0)

# wrap in monitor
env = wrappers.Monitor(env, "./video", force=True)
# initialize policy
policy = make_epsilon_greedy_policy(env.action_space.n, epsilon=0.0, estimator=estimator)
# play in the environment
state = env.reset()
done = False
while not done:
    action_prob = policy(state)
    best_action = np.argmax(action_prob)
    next_state, _, done, _ = env.step(best_action)
    state = next_state
env.close()

# make the video
video = io.open('./video/openaigym.video.%s.video000000.mp4' % env.file_infix,
                'r+b').read()
encoded = pybase64.b64encode(video)
HTML(data='''
<video width="360" height="auto" alt="test" controls><source src="data:video/mp4;base64,{0}" type="video/mp4" /></video>'''
     .format(encoded.decode('ascii')))
