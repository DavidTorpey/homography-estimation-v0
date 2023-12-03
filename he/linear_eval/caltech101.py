import logging

import numpy as np
import torch
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder

from he.configuration import Config


def dataset_split(dataset, test_size=0.2, random_state=0):
    num_total = len(dataset)
    num_val = int(np.ceil(num_total * test_size))
    num_train = num_total - num_val
    train_dataset, val_dataset = random_split(
        dataset, [num_train, num_val],
        generator=torch.Generator().manual_seed(random_state)
    )
    return train_dataset, val_dataset


def get_caltech101(
        config: Config, train_transform, val_transform, test_transform
):
    logging.warning('Unable to use specified val and test transforms.')

    full_dataset = ImageFolder(config.data.root, transform=train_transform)

    train_and_val_dataset, test_dataset = dataset_split(full_dataset, random_state=42, test_size=0.2)

    train_dataset, val_dataset = dataset_split(train_and_val_dataset, random_state=42, test_size=0.2)

    return train_dataset, val_dataset, test_dataset
