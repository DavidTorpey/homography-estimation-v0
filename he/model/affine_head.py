import logging

from he.configuration import Config
from he.model.projection_head import MLPHead


def get_affine_head(config: Config):
    param_head = None
    if config.data.dataset_type == 'affine':
        logging.info('Initialising param head')
        if config.network.aggregation_location == 'f':
            encoder_dim = 512 if config.network.name == 'resnet18' else 2048
        elif config.network.aggregation_location == 'g':
            encoder_dim = config.network.mlp_head.proj_size
        else:
            raise Exception(f'Invalid aggregation location: {config.network.aggregation_location}')

        if config.network.aggregation_strategy == 'diff':
            in_channel = encoder_dim
        elif config.network.aggregation_strategy == 'concat':
            in_channel = encoder_dim * 2
        else:
            raise Exception(f'Invalid aggregation strategy: {config.network.aggregation_strategy}')

        param_head = MLPHead(
            in_channels=in_channel,
            hidden_size=config.network.pred_head.hidden_size,
            proj_size=config.network.pred_head.proj_size
        ).to(config.trainer.device)

    return param_head