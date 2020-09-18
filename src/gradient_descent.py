import numpy as np


def descent(X: np.array, Y: np.array, theta: np.array, alpha):
    m = len(Y)
    h = X.dot(theta)
    theta = theta - ((h - Y).dot(X) * (alpha / m))

    return theta
