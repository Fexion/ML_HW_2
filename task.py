import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, train_size=0.7, random_state=241)

from sklearn import linear_model
import numpy as np
from matplotlib import pyplot as plt

eps = 10**-1

def not_zero(reg, alpha, X_train, y_train):
    lx = reg(alpha = alpha)
    lx.fit(X_train, y_train)
    lx_coeffs = lx.coef_
    return (np.abs(lx_coeffs) > eps).sum()
rng = range(-3, 4)
l1_not_zero = np.array([not_zero(linear_model.Lasso, 10**alpha, X_train, y_train) for alpha in rng])
l2_not_zero = np.array([not_zero(linear_model.Ridge, 10**alpha, X_train, y_train) for alpha in rng])

f, ax1 = plt.subplots(1,  figsize=(8, 5))

ax1.plot(rng, l1_not_zero, label = u"L1")
ax1.plot(rng, l2_not_zero, label = u"L2")
ax1.set_title(u"not zero/alpha")
ax1.set_xlabel(u"10**x")
ax1.set_ylabel(u"not zero el")
ax1.grid()
ax1.legend()


f.show()

from sklearn.metrics import r2_score as R2
from sklearn import Gr
