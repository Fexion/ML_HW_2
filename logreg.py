import numpy as np
from sklearn.base import BaseEstimator
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from scipy.special import expit
from numpy import logaddexp


class LogReg(BaseEstimator):
    def __init__(self, lambda_1=0.0, lambda_2=1.0, gd_type='stochastic',
                 tolerance=1e-6, max_iter=1000, w0=None, alpha=1e-2):
        """
        lambda_1: L1 regularization param
        lambda_2: L2 regularization param
        gd_type: 'full' or 'stochastic'
        tolerance: for stopping gradient descent
        max_iter: maximum number of steps in gradient descent
        w0: np.array of shape (d) - init weights
        alpha: learning rate
        """
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.gd_type = gd_type
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.w0 = w0
        self.alpha = alpha
        self.w = None
        self.loss_history = None

    def fit(self, X, y):
        """
        X: np.array of shape (l, d)
        y: np.array of shape (l)
        ---
        output: self
        """
        self.loss_history = []
        l, d = X.shape
        X = np.c_[X, np.ones(l)]
        d += 1

        if self.w0:
            self.w = np.array(self.w0)
        else:
            self.w = np.zeros(d)

        for i in range(self.max_iter):
            qr = self.calc_gradient(X, y)
            self.loss_history.append(self.calc_loss(X, y))

            if np.linalg.norm(self.alpha * qr) < self.tolerance:
                return self

            self.w -= self.alpha * qr

        return self

    def predict_proba(self, X):
        """
        X: np.array of shape (l, d)
        ---
        output: np.array of shape (l, 2) where
        first column has probabilities of -1
        second column has probabilities of +1
        """
        if self.w is None:
            raise Exception('Not trained yet')

        l, d = X.shape
        proba = np.ones((l, 2))
        proba[:, 1] = expit(np.dot(X, self.w))

        proba[:, 0] -= proba[:, 1]
        return proba

        pass

    def calc_gradient(self, X, y):
        """
        X: np.array of shape (l, d) (l can be equal to 1 if stochastic)
        y: np.array of shape (l)
        ---
        output: np.array of shape (d)
        """
        l, d = X.shape

        if self.gd_type == 'stochastic':
            i = np.random.randint(0, l)
            qg = (-y[i] * X[i] / (1 + np.exp(y[i] *
                                             np.dot(self.w, X[i]))) + self.lambda_2 * self.w / l)
            return qg

        y = np.array([y])
        qg = -y.T * X * expit(y * np.dot(X, self.w)).T + self.lambda_2 * self.w

        return np.mean(qg, axis=0)

    def calc_loss(self, X, y):
        """
        X: np.array of shape (l, d)
        y: np.array of shape (l)
        ---
        output: float
        """
        l, d = X.shape

        Q = logaddexp(np.ones(l) * np.exp(1), -y * np.dot(X, self.w)) + \
            self.lambda_2 * np.dot(self.w, self.w) / 2

        return np.mean(Q, axis=0)
