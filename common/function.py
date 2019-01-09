import numpy as np


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    if x.ndim == 1:
        x = x.reshape(1, len(x))
    x = x - np.max(x, axis=1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def relu(x):
    return np.maximum(0, x)


def mean_square_error(y, y_hat):
    return 0.5 * np.sum(np.square(y - y_hat))


def cross_entropy_error(y, y_hat):
    if y_hat.ndim == 1:
        y = y.reshape(1, len(y))
        y_hat = y_hat.reshape(1, len(y_hat))

    batch_size = y_hat.shape[0]
    is_one_hot = y.size == y_hat.size
    if is_one_hot:
        return -np.sum(y * np.log(y_hat + 1e-7)) / batch_size
    else:
        y_hat = y_hat[np.arange(batch_size), y]
        return -np.sum(np.log(y_hat + 1e-7)) / batch_size
