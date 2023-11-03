import logging

from he.configuration import Config
from he.trainer.simclr import SimCLRTrainer, SimCLRAffineTrainer


def get_simclr_trainer(config, model, param_head, optimizer, scheduler):
    logging.info('Initialising trainer for dataset_type=%s', config.data.dataset_type)
    if config.data.dataset_type == 'default':
        trainer = SimCLRTrainer(
            model, optimizer, scheduler, config
        )
    elif config.data.dataset_type == 'affine':
        trainer = SimCLRAffineTrainer(
            model, param_head, optimizer, scheduler, config
        )
    else:
        raise Exception(f'Dataset type not supported: {config.data.dataset_type}')

    return trainer


def get_trainer(config: Config, model, param_head, optimizer, scheduler):
    algo = config.network.algo

    if algo == 'simclr':
        trainer = get_simclr_trainer(config, model, param_head, optimizer, scheduler)
    elif algo == 'byol':
        ...
    else:
        raise Exception(f'Algorithm not supported: {algo}')

    return trainer
