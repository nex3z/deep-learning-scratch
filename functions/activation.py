import numpy as np


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# def softmax(x):
#     exp = np.exp(x)
#     return exp / np.sum(exp)


def softmax(x):
    exp = np.exp(x - np.max(x))
    return exp / np.sum(exp)


def relu(x):
    return np.maximum(0, x)
