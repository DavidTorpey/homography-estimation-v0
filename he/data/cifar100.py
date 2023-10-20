import logging

import numpy as np
from torchvision.datasets import CIFAR100

from he.cfg import Config
from he.data.dataset import SimCLRNumPyDataset


def get_cifar100(config: Config, train_aug, val_aug):
    dataset = CIFAR100(root='~/data', download=True)

    idx = np.random.permutation(len(dataset.data))
    images = dataset.data[idx]

    if config.data.max_images is not None:
        images = images[:config.data.max_images]

    n_train = int(len(images) * 0.8)

    train_paths = images[:n_train]
    val_paths = images[n_train:]

    train_dataset = SimCLRNumPyDataset(train_paths, train_aug)

    val_dataset = SimCLRNumPyDataset(val_paths, val_aug)

    logging.info('Initialised CIFAR10 dataset: Train=%s, Val=%s', len(train_dataset), len(val_dataset))

    return train_dataset, val_dataset
