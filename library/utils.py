"""
Utilities module
"""


def print_q(q):
    """
    Prints Q dictionary. Used for debugging
    :param q: The Q action value dictionary
    """
    for key in sorted(q.keys()):
        print(key, end=" ")
        value = q[key]
        for i in range(len(value)):
            print(value[i], end=" ")
        print()
