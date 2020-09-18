import numpy as np


def compute_theta(X: np.array, Y: np.array):
    xtx_inv = np.linalg.pinv(X.transpose().dot(X))
    xty = X.transpose().dot(Y)
    return xtx_inv.dot(xty)
