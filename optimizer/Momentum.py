import numpy as np
from optimizer.Optimizer import Optimizer


class Momentum(Optimizer):
    def __init__(self, learning_rate=0.01, alpha=0.9):
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, value in params.items():
                self.v[key] = np.zeros_like(value)

        for key in params.keys():
            self.v[key] = self.alpha * self.v[key] - self.learning_rate * grads[key]
            params[key] += self.v[key]
