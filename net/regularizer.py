import numpy as np

from common.exception import UnimplementedMethodException


class Regularizer(object):
    def loss(self, weight):
        raise UnimplementedMethodException()

    def dloss(self, weight):
        raise UnimplementedMethodException()


class L2(Regularizer):
    def __init__(self, decay_lambda):
        self.decay_lambda = decay_lambda

    def loss(self, weight):
        return 0.5 * self.decay_lambda * np.sum(weight ** 2)

    def dloss(self, weight):
        return self.decay_lambda * weight
