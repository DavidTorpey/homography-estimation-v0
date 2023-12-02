import os
from argparse import ArgumentParser

import dacite
import torch
import yaml
from torch.utils.data import DataLoader

from he.configuration import Config
from he.linear_eval.data import get_datasets
from he.linear_eval.train import Trainer
from he.model.backbone import Encoder
from he.model.byol import BYOL


def get_model(config, args):
    device = 'cuda:0'

    if config.network.algo == 'simclr':
        encoder = Encoder(config)
        output_feature_dim = encoder.projection.net[0].in_features
        model_file_path = args.model_path
        load_params = torch.load(
            model_file_path,
            map_location=torch.device(device)
        )
        encoder.load_state_dict(load_params)
        encoder = encoder.encoder
        encoder = encoder.to(device)
    elif config.network.algo == 'byol':
        byol = BYOL(config)
        model_file_path = args.model_path
        load_params = torch.load(
            model_file_path,
            map_location=torch.device(device)
        )
        byol.load_state_dict(load_params)
        encoder = byol.online_network.encoder
        output_feature_dim = 2048 if config.network.name == 'resnet50' else 512
        encoder = encoder.to(device)
    elif config.network.algo == 'barlow_twins':
        encoder = Encoder(config)
        output_feature_dim = encoder.projection.net[0].in_features
        model_file_path = args.model_path
        load_params = torch.load(
            model_file_path,
            map_location=torch.device(device)
        )
        encoder.load_state_dict(load_params)
        encoder = encoder.encoder
        encoder = encoder.to(device)
    else:
        raise Exception(f'Invalid model: {config.network.algo}')

    return encoder


def train_le(config: Config, args):
    dataset = config.data.dataset
    print('Using dataset:', dataset)
    train_dataset, val_dataset, test_dataset = get_datasets(dataset, config, config.data.image_size)

    batch_size = config.trainer.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=0, drop_last=False, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             num_workers=0, drop_last=False, shuffle=False)

    encoder = get_model(config, args)

    trainer = Trainer(encoder, config)

    logistic_regression_model = trainer.train(train_loader, val_loader)

    trainer.test(test_loader, logistic_regression_model)


def main():
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--run_num', type=str, default=None)
    args = parser.parse_args()

    config_dict = yaml.load(open(args.config_path, "r"), Loader=yaml.FullLoader)
    config: Config = dacite.from_dict(Config, config_dict)

    if args.run_num:
        config.general.output_dir = os.path.join(config.general.output_dir, args.run_num)

    train_le(config, args)


if __name__ == '__main__':
    main()
