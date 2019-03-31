"""
Module that contains estimators used to approximate the value functions
"""
import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor


class Estimator:
    """
    General estimator interface. Subclass this and implement
    the methods and use it to approximate the value functions.
    """
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


class QFunctionSGD(Estimator):
    def __init__(self, env):
        # get state samples from the environment
        # to fit the scaler and featurizer
        env_samples = np.array([env.observation_space.sample() for _ in range(10000)])
        # initialize standard scaler (scales to 0 mean and 1 std)
        self.scaler = StandardScaler()
        # 4 Radial Basis Functions with 100 components each concatenated
        self.featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
        # fit the scaler and the featurizer
        self.featurizer.fit(self.scaler.fit_transform(env_samples))
        # initialize models, one model for each action
        self.models = []
        for _ in range(env.action_space.n):
            # initialize model
            model = SGDRegressor(learning_rate='constant')
            # fit a value (from reset)
            model.partial_fit([self.get_representation(env.reset())], [0])
            # add to models
            self.models.append(model)

    def predict(self, state, action=None):
        representation = self.get_representation(state)
        if action is None:
            return np.array([m.predict([representation])[0] for m in self.models])
        else:
            return self.models[action].predict([representation])[0]

    def update(self, state, action, target):
        representation = self.get_representation(state)
        self.models[action].partial_fit([representation], [target])

    def get_representation(self, state):
        """
        Get the representation of the given state
        using the scaler and the featurizer
        :param state: the state for which we are getting the representation
        :return: the representation of the state
        """
        scaled = self.scaler.transform([state])
        return self.featurizer.transform(scaled)[0]
