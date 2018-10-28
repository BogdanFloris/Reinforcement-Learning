from library.environments.gridworld import GridWorldEnv
from TemporalDifferenceLearning.temporal_diff_learning import sarsa, q_learning, make_epsilon_greedy_policy

num_episodes = 1000
epsilon = 0.0
learning_algorithms = ["SARSA", "Q-Learning"]
learning = learning_algorithms[0]
env = GridWorldEnv()
if learning == "SARSA":
    q, _ = sarsa(env, num_episodes, epsilon=epsilon)
elif learning == "Q-Learning":
    q, _ = q_learning(env, num_episodes, epsilon=epsilon)
else:
    q = None
policy = make_epsilon_greedy_policy(q=q, epsilon=epsilon, action_count=env.action_space.n)
print("Policy after {} episodes of {} for \u03B5 = {}".format(num_episodes, learning, epsilon))
env.render_policy(policy)
