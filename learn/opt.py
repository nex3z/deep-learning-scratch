import numpy as np


def numerical_diff(f, x):
    delta = 1e-4
    return (f(x + delta) - f(x - delta)) / (2 * delta)


def numerical_gradient_1d(f, x):
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


def numerical_gradient(f, x):
    if x.ndim == 1:
        return numerical_gradient_1d(f, x)
    else:
        return np.apply_along_axis(lambda row: numerical_gradient_1d(f, row), 1, x)


def gradient_decent(f, initial_x, learning_rate=0.01, num_step=100):
    x = initial_x.copy()
    for i in range(num_step):
        grad = numerical_gradient(f, x)
        x -= learning_rate * grad
    return x
