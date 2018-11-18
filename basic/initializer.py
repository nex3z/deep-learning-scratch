import numpy as np


def normal(*shape, std=0.01):
    return np.random.randn(*shape) * std


def xavier(*shape):
    return np.random.randn(*shape) * np.sqrt(1.0 / shape[0])


def he(*shape):
    return np.random.randn(*shape) * np.sqrt(2.0 / shape[0])
