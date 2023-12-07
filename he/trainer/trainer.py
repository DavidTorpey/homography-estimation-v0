import logging

from he.configuration import Config
from he.trainer.barlow_twins import BarlowTwinsAffineTrainer, BarlowTwinsTrainer, BarlowTwinsDoubleAffineTrainer
from he.trainer.byol import BYOLAffineTrainer, BYOLTrainer, BYOLDoubleAffineTrainer
from he.trainer.simclr import SimCLRTrainer, SimCLRAffineTrainer, SimCLRDoubleAffineTrainer


def get_simclr_trainer(config, model, param_head, optimizer, scheduler):
    logging.info('Initialising SimCLR trainer for dataset_type=%s', config.data.dataset_type)
    if config.data.dataset_type == 'default':
        trainer = SimCLRTrainer(
            model, optimizer, scheduler, config
        )
    elif config.data.dataset_type == 'affine':
        if config.data.affine_type == 'single':
            trainer = SimCLRAffineTrainer(
                model, param_head, optimizer, scheduler, config
            )
        elif config.data.affine_type == 'double':
            trainer = SimCLRDoubleAffineTrainer(
                model, param_head, optimizer, scheduler, config
            )
        elif config.data.affine_type == 'bounded':
            trainer = SimCLRAffineTrainer(
                model, param_head, optimizer, scheduler, config
            )
        else:
            raise Exception(f'Invalid affine type: {config.data.affine_type}')
    else:
        raise Exception(f'Dataset type not supported: {config.data.dataset_type}')

    return trainer


def get_byol_trainer(config, model, param_head, optimizer, scheduler):
    logging.info('Initialising BYOL trainer for dataset_type=%s', config.data.dataset_type)
    if config.data.dataset_type == 'default':
        trainer = BYOLTrainer(
            model, optimizer, scheduler, config
        )
    elif config.data.dataset_type == 'affine':
        if config.data.affine_type == 'single':
            trainer = BYOLAffineTrainer(
                model, param_head, optimizer, scheduler, config
            )
        elif config.data.affine_type == 'double':
            trainer = BYOLDoubleAffineTrainer(
                model, param_head, optimizer, scheduler, config
            )
        elif config.data.affine_type == 'bounded':
            trainer = BYOLAffineTrainer(
                model, param_head, optimizer, scheduler, config
            )
        else:
            raise Exception(f'Invalid affine type: {config.data.affine_type}')
    else:
        raise Exception(f'Dataset type not supported: {config.data.dataset_type}')

    return trainer


def get_barlow_twins_trainer(config, model, param_head, optimizer, scheduler):
    logging.info('Initialising BarlowTwins trainer for dataset_type=%s', config.data.dataset_type)
    if config.data.dataset_type == 'default':
        trainer = BarlowTwinsTrainer(
            model, optimizer, scheduler, config
        )
    elif config.data.dataset_type == 'affine':
        if config.data.affine_type == 'single':
            trainer = BarlowTwinsAffineTrainer(
                model, param_head, optimizer, scheduler, config
            )
        elif config.data.affine_type == 'double':
            trainer = BarlowTwinsDoubleAffineTrainer(
                model, param_head, optimizer, scheduler, config
            )
        elif config.data.affine_type == 'bounded':
            trainer = BarlowTwinsAffineTrainer(
                model, param_head, optimizer, scheduler, config
            )
        else:
            raise Exception(f'Invalid affine type: {config.data.affine_type}')
    else:
        raise Exception(f'Dataset type not supported: {config.data.dataset_type}')

    return trainer


def get_trainer(config: Config, model, param_head, optimizer, scheduler):
    algo = config.network.algo

    if algo == 'simclr':
        trainer = get_simclr_trainer(config, model, param_head, optimizer, scheduler)
    elif algo == 'byol':
        trainer = get_byol_trainer(config, model, param_head, optimizer, scheduler)
    elif algo == 'barlow_twins':
        trainer = get_barlow_twins_trainer(config, model, param_head, optimizer, scheduler)
    else:
        raise Exception(f'Algorithm not supported: {algo}')

    return trainer
