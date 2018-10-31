import numpy as np


def numerical_diff(f, x):
    delta = 1e-4
    return (f(x + delta) - f(x - delta)) / (2 * delta)


def numerical_gradient(f, x):
    delta = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp = x[idx]

        x[idx] = tmp + delta
        y_1 = f(x)

        x[idx] = tmp - delta
        y_2 = f(x)

        grad[idx] = (y_1 - y_2) / (2 * delta)
        x[idx] = tmp

    return grad
