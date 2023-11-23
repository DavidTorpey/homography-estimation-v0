import os

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from he.configuration import Config


class Food101Dataset(Dataset):
    def __init__(self, relative_image_paths, transform, class_map, config: Config):
        self.relative_image_paths = relative_image_paths
        self.transform = transform
        self.class_map = class_map
        self.config = config

    def __len__(self):
        return len(self.relative_image_paths)

    def __getitem__(self, item):
        chosen_relative_image_path = self.relative_image_paths[item]
        label = os.path.dirname(chosen_relative_image_path)
        image_path = os.path.join(self.config.data.root, 'images', chosen_relative_image_path)

        image = self.transform(Image.open(image_path).convert('RGB'))
        label = torch.from_numpy(np.array(self.class_map[label])).long()

        return image, label


def add_ext(l):
    return [e + '.jpg' for e in l]


def get_food101(
        config: Config, train_transform, test_transform
):
    train_images_path = os.path.join(config.data.root, 'meta/train.txt')
    with open(train_images_path) as file:
        train_image_paths = file.read().splitlines()

    test_images_path = os.path.join(config.data.root, 'meta/test.txt')
    with open(test_images_path) as file:
        test_image_paths = file.read().splitlines()
    test_image_paths = add_ext(test_image_paths)

    classes = sorted(list(set([os.path.dirname(e) for e in train_image_paths])))
    class_map = {e: i for i, e in enumerate(classes)}

    train_dataset = Food101Dataset(train_image_paths, train_transform, class_map, config)
    test_dataset = Food101Dataset(test_image_paths, test_transform, class_map, config)

    return train_dataset, test_dataset
