import sys
from argparse import ArgumentParser
from pathlib import Path

import dacite
import torch
import yaml
from torchvision import datasets

from he.configuration import Config
from he.data.data import get_data
from he.model.projection_head import MLPHead
from .data.augmentations import get_simclr_data_transforms
from .data.multiview_injector import MultiViewDataInjector
from .model.backbone import ResNetSimCLR
from .data.utils import get_train_validation_data_loaders
from .trainer import Trainer, AffineTrainer


def main():
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    args = parser.parse_args()

    config_dict = yaml.load(open(args.config_path, "r"), Loader=yaml.FullLoader)
    config: Config = dacite.from_dict(Config, config_dict)

    dataset = config.data.dataset

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")
    print('Using dataset:', dataset)

    train_loader, valid_loader = get_data(config)

    model = ResNetSimCLR(config).to(device)

    param_head = None
    if config.data.dataset_type == 'affine':
        param_head = MLPHead(
            in_channels=512,
            hidden_size=config.network.pred_head.hidden_size,
            proj_size=config.network.pred_head.proj_size
        ).to(device)

    if config.data.dataset_type == 'default':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.optimiser.lr,
            weight_decay=config.optimiser.weight_decay
        )
    elif config.data.dataset_type == 'affine':
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(param_head.parameters()),
            lr=config.optimiser.lr,
            weight_decay=config.optimiser.weight_decay
        )
    else:
        raise Exception(f'Dataset type not supported: {config.data.dataset_type}')

    batch_size = config.trainer.batch_size
    epochs = config.trainer.epochs

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1
    )

    warmup_steps = config.trainer.warmup_epochs
    run_folder = config.general.output_dir
    Path(run_folder).mkdir(parents=True, exist_ok=True)

    if config.data.dataset_type == 'default':
        trainer = Trainer(
            model, optimizer, scheduler, batch_size, epochs, device, dataset,
            run_folder, warmup_steps
        )
    elif config.data.dataset_type == 'affine':
        trainer = AffineTrainer(
            model, param_head, optimizer, scheduler, batch_size,
            epochs, device, dataset, run_folder, warmup_steps
        )
    else:
        raise Exception(f'Dataset type not supported: {config.data.dataset_type}')

    trainer.train(train_loader, valid_loader)


if __name__ == '__main__':
    main()
