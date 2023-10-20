import json
import logging
import os
from argparse import ArgumentParser
from dataclasses import asdict

import torch
from torch.backends import cudnn

from he.cfg import load_config, Config
from he.constants import LOG_FILE_NAME, CONFIG_FILE_NAME
from he.data.data import get_loaders
from he.model.model import get_model
from he.trainer import Trainer
from he.utl import mkdir, cosine_scheduler


def train(config: Config):
    cudnn.benchmark = True

    train_loader, val_loader = get_loaders(config)

    ssl_model, homography_estimator = get_model(config)
    ssl_model = ssl_model.to(config.optim.device)
    homography_estimator = homography_estimator.to(config.optim.device)

    lr_schedule = cosine_scheduler(
        config.optim.lr, 0, config.optim.epochs, len(train_loader),
        config.optim.warmup_epochs
    )

    optimiser = torch.optim.Adam(
        list(ssl_model.parameters()) + list(homography_estimator.parameters()),
        lr=config.optim.lr,
        weight_decay=config.optim.lr
    )

    trainer = Trainer(ssl_model, homography_estimator, optimiser, lr_schedule, config)

    trainer.train(train_loader, val_loader)



def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config_path)

    mkdir(config.general.output_dir)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(module)s:%(funcName)s:%(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.general.output_dir, LOG_FILE_NAME)),
            logging.StreamHandler()
        ]
    )

    with open(os.path.join(config.general.output_dir, CONFIG_FILE_NAME), 'w') as handle:
        json.dump(asdict(config), handle, indent=2)

    train(config)


if __name__ == '__main__':
    main()
