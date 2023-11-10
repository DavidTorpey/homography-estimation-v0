import logging
import os
import random
from pathlib import Path

import numpy as np
from torchvision import datasets

from he.configuration import Config
from he.data.augmentations import get_simclr_data_transforms
from he.data.dataset import CustomDataset, DefaultDataset
from he.data.multiview_injector import MultiViewDataInjector
from he.data.utils import get_train_validation_data_loaders


def get_default(config: Config):
    data_transform = get_simclr_data_transforms(config)
    dataset = config.data.dataset

    logging.info('Initialising dataset: %s', dataset)

    if dataset == 'stl10':
        train_dataset = datasets.STL10(
            './data', split='train+unlabeled', download=True,
            transform=MultiViewDataInjector([data_transform, data_transform])
        )
    elif dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(
            './data', train=True, download=True,
            transform=MultiViewDataInjector([data_transform, data_transform])
        )
    elif dataset == 'svhn':
        train_dataset = datasets.SVHN(
            './data', split='train', download=True,
            transform=MultiViewDataInjector([data_transform, data_transform])
        )
    elif dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(
            './data', train=True, download=True,
            transform=MultiViewDataInjector([data_transform, data_transform])
        )
    elif dataset == 'tiny_imagenet':
        root = config.data.root
        image_paths = list(Path(os.path.join(root, 'tiny-imagenet-200', 'train')).rglob('*.JPEG'))
        random.shuffle(image_paths)
        train_dataset = DefaultDataset(
            image_paths,
            transform=data_transform
        )
    else:
        raise Exception(f'Dataset not supported: {dataset}')

    return train_dataset


def get_affine(config: Config):
    dataset = config.data.dataset
    logging.info('Initialising dataset: %s', dataset)

    if dataset == 'stl10':
        train_dataset = datasets.STL10('./data', split='train+unlabeled', download=True)
        trainset = train_dataset.data
        trainset = np.swapaxes(np.swapaxes(trainset, 1, 2), 2, 3)
    elif dataset == 'cifar10':
        train_dataset = datasets.CIFAR10('./data', train=True, download=True)
        trainset = train_dataset.data
    elif dataset == 'svhn':
        train_dataset = datasets.SVHN('./data', split='train', download=True)
        trainset = train_dataset.data
        trainset = np.swapaxes(np.swapaxes(trainset, 1, 2), 2, 3)
    elif dataset == 'cifar100':
        train_dataset = datasets.CIFAR100('./data', train=True, download=True)
        trainset = train_dataset.data
    else:
        raise Exception(f'Dataset not supported: {dataset}')

    train_dataset = CustomDataset(trainset, config)

    return train_dataset


def get_data(config: Config):
    logging.info('Initialising dataset for dataset_type=%s', config.data.dataset_type)
    if config.data.dataset_type == 'default':
        train_dataset = get_default(config)
    elif config.data.dataset_type == 'affine':
        train_dataset = get_affine(config)
    else:
        raise Exception(f'Dataset type not supported: {config.data.dataset_type}')

    train_loader, valid_loader = get_train_validation_data_loaders(
        train_dataset, 0.05, config.trainer.batch_size,
        config.trainer.num_workers
    )

    return train_loader, valid_loader
