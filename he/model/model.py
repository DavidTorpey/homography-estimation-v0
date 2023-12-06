import logging

from he.configuration import Config
from he.model.backbone import Encoder
from he.model.byol import BYOL


def get_model(config: Config):
    model_name = config.network.algo

    logging.info(f'Initialising model: {model_name}')

    if model_name == 'simclr':
        model = Encoder(config)
    elif model_name == 'byol':
        model = BYOL(config)
    elif model_name == 'barlow_twins':
        model = Encoder(config)
    else:
        raise Exception(f'Model not supported: {model_name}')

    return model
