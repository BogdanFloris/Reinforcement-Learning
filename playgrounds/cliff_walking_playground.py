from library.environments.cliff_walking import CliffWalkingEnv
from TemporalDifferenceLearning.temporal_diff_learning import sarsa, make_epsilon_greedy_policy

num_episodes = 100
epsilon = 0.0
env = CliffWalkingEnv()
q, stats = sarsa(env, num_episodes)
policy = make_epsilon_greedy_policy(q=q, epsilon=epsilon, action_count=env.action_space.n)
print("Policy after {} episodes of SARSA".format(num_episodes))
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
