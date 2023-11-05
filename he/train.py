import logging
import os
from argparse import ArgumentParser
from pathlib import Path

import dacite
import torch
import yaml

from he.configuration import Config
from he.data.data import get_data
from he.model.model import get_model
from he.model.projection_head import MLPHead
from he.trainer.trainer import get_trainer


def main():
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--run_num', type=str, default=None)
    args = parser.parse_args()

    config_dict = yaml.load(open(args.config_path, "r"), Loader=yaml.FullLoader)
    config: Config = dacite.from_dict(Config, config_dict)

    if args.run_num:
        config.general.output_dir = os.path.join(config.general.output_dir, args.run_num)

    run_folder = config.general.output_dir
    Path(run_folder).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(module)s:%(funcName)s:%(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.general.output_dir, 'out.log')),
            logging.StreamHandler()
        ]
    )

    dataset = config.data.dataset

    logging.info(f"Training with device: {config.trainer.device}")
    logging.info('Using dataset:', dataset)

    train_loader, valid_loader = get_data(config)

    model = get_model(config).to(config.trainer.device)

    param_head = None
    if config.data.dataset_type == 'affine':
        logging.info('Initialising param head')
        param_head = MLPHead(
            in_channels=512 if config.network.name == 'resnet18' else 2048,
            hidden_size=config.network.pred_head.hidden_size,
            proj_size=config.network.pred_head.proj_size
        ).to(config.trainer.device)

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

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1
    )

    trainer = get_trainer(config, model, param_head, optimizer, scheduler)

    trainer.train(train_loader, valid_loader)


if __name__ == '__main__':
    main()
