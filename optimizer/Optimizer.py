from common.exception import UnimplementedMethodException


class Optimizer(object):
    def update(self, params, grads):
        raise UnimplementedMethodException()
