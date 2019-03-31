"""
Module that contains estimators used to approximate the value functions
"""


class Estimator:
    """
    General estimator interface. Subclass this and implement
    the methods and use it to approximate the value functions.
    """
    def __init__(self):
        # one model for each discrete action
        self.models = None

    def predict(self, state, action=None):
        """
        Predicts the value of the given state.
        If a is not None, then it returns the value
        of the state action pair.
        If a is None, then it returns a list of values
        for each state action pairs
        :param state: the state for which we wish to make a prediction
        :param action: the action chosen (can be None)
        :return: the value of the state
        """
        raise NotImplementedError

    def update(self, state, action, target):
        """
        Updates the parameters of the estimator using the given target
        :param state: the state
        :param action: the action
        :param target: the target used to update
        :return:
        """
        raise NotImplementedError
