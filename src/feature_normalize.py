import numpy as np


def normalize(X: np.array):
    # Given an X matrix of col features and row examples
    # normalize the given matrix by subtracting it's mean
    # and dividing by the standard deviations
    mu = np.mean(X, axis=0)
    std = np.std(X, axis=0, ddof=1)

    return [(X - mu) / std, mu, std]
