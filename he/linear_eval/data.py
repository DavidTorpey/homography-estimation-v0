import torchvision
from torch.utils.data import random_split
from torchvision import transforms, datasets

from he.linear_eval.food101 import get_food101


def get_datasets(dataset, config, image_size, val_p=0.2):
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop((image_size, image_size)),
        transforms.ToTensor()
    ])

    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop((image_size, image_size)),
        transforms.ToTensor()
    ])

    if dataset == 'stl10':
        train_dataset = datasets.STL10('./data', split='train', download=True,
                                       transform=train_transforms)

        test_dataset = datasets.STL10('./data', split='test', download=True,
                                      transform=test_transforms)
    elif dataset == 'cifar10':
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=train_transforms)
        test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=test_transforms)
    elif dataset == 'svhn':
        train_dataset = datasets.SVHN('./data', split='train', download=True, transform=train_transforms)
        test_dataset = datasets.SVHN('./data', split='test', download=True, transform=test_transforms)
    elif dataset == 'cifar100':
        train_dataset = datasets.CIFAR100('./data', train=True, download=True, transform=train_transforms)
        test_dataset = datasets.CIFAR100('./data', train=False, download=True, transform=test_transforms)
    elif dataset == 'food101':
        train_dataset, test_dataset = get_food101(config, train_transforms, test_transforms)
    else:
        raise Exception(f'Invalid linear evaluation dataset: {dataset}')

    num_val = int(len(train_dataset) * val_p)
    num_train = len(train_dataset) - num_val
    train_dataset, val_dataset = random_split(train_dataset, [num_train, num_val])

    return train_dataset, val_dataset, test_dataset
