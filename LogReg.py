from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import special
from sklearn.datasets import make_classification
from numpy.linalg import norm
from scipy.special import expit


class LogReg(BaseEstimator):
    def __init__(self, lambda_1=0.0, lambda_2=1.0, gd_type='stochastic',
                 tolerance=1e-4, max_iter=1000, w0=None, alpha=1e-3):
        """lambda_1: L1 regularization param lambda_2: L2 regularization param
        gd_type: 'full' or 'stochastic' tolerance: for stopping gradient
        descent max_iter: maximum number of steps in gradient descent.

        w0: np.array of shape (d) - init weights
        alpha: learning rate

        """
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.gd_type = gd_type
        self.stochastic_gd = gd_type == 'stochastic'
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

        if self.w0 is not None:
            self.w = np.array(self.w0, dtype='float64', copy=True)
        else:
            self.w = np.zeros(d, dtype='float64')

        # if self.stochastic_gd:
        #     self.w = self.calc_gradient(X, y, True)

        for i in range(self.max_iter):
            if self.stochastic_gd:
                n = randint(0, l - 1)
                gd = self.calc_gradient(
                    X[n].reshape((1, d)), y[n].reshape((1, 1)))
            else:
                gd = self.calc_gradient(X, y)

            w = self.w - self.alpha * gd
            if norm(w - self.w) < self.tolerance:
                print('break by tolerance')
                break

            self.loss_history.append(self.calc_loss(X, y))
            self.w = w

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
        p = expit((X * self.w).sum(axis=1))
        return np.array([1 - p, p]).T

    def calc_gradient(self, X, y):
        """
        X: np.array of shape (l, d) (l can be equal to 1 if stochastic)
        y: np.array of shape (l)
        ---
        output: np.array of shape (d)
        """
        # ∇_w Q(w,xi)= -y_i * x_i / (1 + exp(y_i * ⟨w,x_i⟩))+λ_2* w

        l = X.shape[0]

        a = np.multiply(X.T, -y, dtype='float64').T
        b = expit(-y * np.multiply(X, self.w, dtype='float64').sum(axis=1))
        c = self.lambda_2 * self.w
        f = (a.T * b).T + c

        return f.sum(axis=0) / l

    def calc_loss(self, X, y):
        """
        X: np.array of shape (l, d)
        y: np.array of shape (l)
        ---
        output: float
        """
        # Q(w,X)=1/l * ∑_{i=1}^l log(1+exp(−y_i * ⟨w,x_i⟩))+λ_2/2 * ∥w∥_2_2

        l = X.shape[0]
        b = 1 + np.exp(-y * (X * self.w).sum(axis=1))
        c = self.lambda_2 * (norm(self.w)**2) / 2.
        f = np.log(b).sum() / l + c
        return f
