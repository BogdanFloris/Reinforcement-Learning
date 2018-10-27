from library.environments.cliff_walking import CliffWalkingEnv
from TemporalDifferenceLearning.temporal_diff_learning import sarsa, make_epsilon_greedy_policy

env = CliffWalkingEnv()
q, stats = sarsa(env, 200)
policy = make_epsilon_greedy_policy(q=q, epsilon=0.0, action_count=env.action_space.n)
print("Policy after 200 episodes of SARSA")
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
