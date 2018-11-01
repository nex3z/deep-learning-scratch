import numpy as np


def mean_square_error(y, y_hat):
    return 0.5 * np.sum(np.square((y - y_hat)))


def cross_entropy_error(y, y_hat):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        y_hat = y_hat.reshape(1, y_hat.size)
    return -np.sum(y_hat * np.log(y + 1e-7))
