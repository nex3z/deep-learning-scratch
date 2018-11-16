import numpy as np
from optimizer.Optimizer import Optimizer


class RmsProp(Optimizer):
    def __init__(self, learning_rate=0.01, decay_rate=0.99):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] = self.decay_rate * self.h[key] + (1 - self.decay_rate) * grads[key] * grads[key]
            params[key] -= self.learning_rate * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
