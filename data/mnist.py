import gzip
import os
import os.path as path
import urllib.request
import numpy as np

from data.DataSet import DataSet, Datasets

IMAGE_SIZE = 784
TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
BASE_URL = 'http://yann.lecun.com/exdb/mnist/'


def read_data(base_dir, normalize=True, validation_size=5000, one_hot=False):
    check_data(base_dir)

    train_images = read_image(path.join(base_dir, TRAIN_IMAGES))
    test_images = read_image(path.join(base_dir, TEST_IMAGES))

    if normalize:
        train_images = train_images / 255.0
        test_images = test_images / 255.0

    train_labels = read_label(path.join(base_dir, TRAIN_LABELS), one_hot)
    validation = DataSet(images=train_images[:validation_size], labels=train_labels[:validation_size])
    train = DataSet(images=train_images[validation_size:], labels=train_labels[validation_size:])

    test_labels = read_label(path.join(base_dir, TEST_LABELS), one_hot)
    test = DataSet(images=test_images, labels=test_labels)

    return Datasets(train=train, validation=validation, test=test)


def check_data(base_dir):
    if not path.exists(base_dir):
        os.makedirs(base_dir)

    for file_name in [TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, TEST_LABELS]:
        if not path.isfile(path.join(base_dir, file_name)):
            download(base_dir, file_name)
            print("downloading {}".format(file_name))


def download(base_dir, file_name):
    urllib.request.urlretrieve(BASE_URL + file_name, path.join(base_dir, file_name))


def read_label(file_path, one_hot):
    print("reading {}".format(file_path))
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    if one_hot:
        labels = convert_to_one_hot(labels, num_classes=10)
    return labels


def convert_to_one_hot(labels, num_classes=10):
    num_labels = labels.shape[0]
    one_hot = np.zeros((num_labels, num_classes))
    ones_offset = np.arange(num_labels) * num_classes + labels.ravel()
    one_hot.flat[ones_offset] = 1
    return one_hot


def read_image(file_path):
    print("reading {}".format(file_path))
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, IMAGE_SIZE)
    return data
