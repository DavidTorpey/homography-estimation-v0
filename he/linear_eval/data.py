import torchvision
from torch.utils.data import random_split
from torchvision import transforms, datasets

from he.linear_eval.caltech101 import get_caltech101
from he.linear_eval.food101 import get_food101


def get_datasets(dataset, config, image_size, val_p=0.1):
    train_transforms = torchvision.transforms.Compose([
        transforms.Resize(image_size),
        torchvision.transforms.RandomCrop((image_size, image_size)),
        transforms.ToTensor()
    ])

    test_transforms = torchvision.transforms.Compose([
        transforms.Resize(image_size),
        torchvision.transforms.RandomCrop((image_size, image_size)),
        transforms.ToTensor()
    ])

    if dataset == 'stl10':
        train_dataset = datasets.STL10(
            './data', split='train', download=True, transform=train_transforms
        )

        test_dataset = datasets.STL10(
            './data', split='test', download=True, transform=test_transforms
        )
        num_val = int(len(train_dataset) * val_p)
        num_train = len(train_dataset) - num_val
        train_dataset, val_dataset = random_split(train_dataset, [num_train, num_val])
    elif dataset == 'cifar10':
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=train_transforms)
        test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=test_transforms)
        num_val = int(len(train_dataset) * val_p)
        num_train = len(train_dataset) - num_val
        train_dataset, val_dataset = random_split(train_dataset, [num_train, num_val])
    elif dataset == 'svhn':
        train_dataset = datasets.SVHN('./data', split='train', download=True, transform=train_transforms)
        test_dataset = datasets.SVHN('./data', split='test', download=True, transform=test_transforms)
        num_val = int(len(train_dataset) * val_p)
        num_train = len(train_dataset) - num_val
        train_dataset, val_dataset = random_split(train_dataset, [num_train, num_val])
    elif dataset == 'cifar100':
        train_dataset = datasets.CIFAR100('./data', train=True, download=True, transform=train_transforms)
        test_dataset = datasets.CIFAR100('./data', train=False, download=True, transform=test_transforms)
        num_val = int(len(train_dataset) * val_p)
        num_train = len(train_dataset) - num_val
        train_dataset, val_dataset = random_split(train_dataset, [num_train, num_val])
    elif dataset == 'food101':
        train_dataset, test_dataset = get_food101(config, train_transforms, test_transforms)
        num_val = int(len(train_dataset) * val_p)
        num_train = len(train_dataset) - num_val
        train_dataset, val_dataset = random_split(train_dataset, [num_train, num_val])
    elif dataset == 'caltech101':
        train_dataset, val_dataset, test_dataset = get_caltech101(config, train_transforms, train_transforms, train_transforms)
    else:
        raise Exception(f'Invalid linear evaluation dataset: {dataset}')

    return train_dataset, val_dataset, test_dataset
