import collections


class DataSet(object):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels


Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
