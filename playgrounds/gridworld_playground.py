from library.environments.gridworld import GridWorldEnv
from library.td_learning.temporal_diff_learning import sarsa, q_learning, make_epsilon_greedy_policy
from library.plotting import plot_episode_stats

num_episodes = 1000
epsilon = 0.1
learning_algorithms = ["SARSA", "Q-Learning"]
learning = learning_algorithms[1]
env = GridWorldEnv()
if learning == "SARSA":
    q, stats = sarsa(env, num_episodes, epsilon=epsilon)
elif learning == "Q-Learning":
    q, stats = q_learning(env, num_episodes, epsilon=epsilon)
else:
    q = None
    stats = None
policy = make_epsilon_greedy_policy(q=q, epsilon=0.0, action_count=env.action_space.n)
print("Policy after {} episodes of {} for \u03B5 = {}".format(num_episodes, learning, epsilon))
env.render_policy(policy)
plot_episode_stats(stats)
