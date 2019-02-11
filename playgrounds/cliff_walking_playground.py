from library.environments.cliff_walking import CliffWalkingEnv
from library.td_learning import sarsa, q_learning, make_epsilon_greedy_policy

num_episodes = 1000
epsilon = 0.1
learning_algorithms = ["SARSA", "Q-Learning"]
learning = learning_algorithms[1]
env = CliffWalkingEnv()
if learning == "SARSA":
    q, _ = sarsa(env, num_episodes, epsilon=epsilon)
elif learning == "Q-Learning":
    q, _ = q_learning(env, num_episodes, epsilon=epsilon)
else:
    q = None
policy = make_epsilon_greedy_policy(q=q, epsilon=epsilon, action_count=env.action_space.n)
print("Policy after {} episodes of {} for \u03B5 = {}".format(num_episodes, learning, epsilon))
env.render_policy(policy)


def print_q():
    """
    Prints q dictionary. Used for debugging
    """
    for key in sorted(q.keys()):
        print(key, end=" ")
        value = q[key]
        for i in range(len(value)):
            print(value[i], end=" ")
        print()
