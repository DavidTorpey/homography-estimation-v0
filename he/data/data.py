from torch.utils.data import DataLoader

from he.cfg import Config
from he.data.augmentations import get_transform
from he.data.cifar10 import get_cifar10
from he.data.cifar100 import get_cifar100

def get_loaders(config: Config):
    dataset = config.data.dataset

    train_transform = get_transform(config.data.train_aug, config)
    val_transform = get_transform(config.data.val_aug, config)

    if dataset == 'cifar10':
        train_dataset, val_dataset = get_cifar10(config, train_transform, val_transform)
    elif dataset == 'cifar100':
        train_dataset, val_dataset = get_cifar100(config, train_transform, val_transform)
    else:
        raise NotImplementedError(f'Dataset {dataset} not supported')

    train_loader = DataLoader(
        train_dataset, batch_size=config.optim.batch_size, drop_last=True,
        shuffle=True, num_workers=config.optim.workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=config.optim.batch_size, drop_last=True,
        shuffle=False, num_workers=config.optim.workers, pin_memory=True
    )

    return train_loader, val_loader
