import logging
import os
from copy import deepcopy
from dataclasses import asdict

import wandb
import numpy as np
import torch
import torch.nn.functional as F
from lightly.loss import BarlowTwinsLoss
from torch import nn

from he.configuration import Config


class BarlowTwinsTrainer:
    def __init__(self, model, optimizer, scheduler, config: Config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_size = config.trainer.batch_size
        self.epochs = config.trainer.epochs
        self.device = config.trainer.device
        self.dataset = config.data.dataset
        self.run_folder = config.general.output_dir
        self.warmup_steps = config.trainer.warmup_epochs
        self.config = config

        self.model_name = 'model_{}.pth'.format(self.dataset)

        self.criterion = BarlowTwinsLoss()

        if config.general.log_to_wandb:
            wandb.init(
                project='phd', config=asdict(config),
                name=f'BarlowTwins : {config.data.dataset}',
                tags=[f'run_id: {config.general.run_id}'],
            )

    def _step(self, xis, xjs):
        _, zis = self.model(xis)  # [N,C]

        # get the representations and the projections
        _, zjs = self.model(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self.criterion(zis, zjs)

        return loss

    def _validate(self, val_loader):
        with torch.no_grad():
            self.model.eval()

            valid_loss = 0.0
            counter = 0
            for (xis, xjs), _ in val_loader:
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(xis, xjs)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        self.model.train()

        return valid_loss

    def train(self, train_loader, val_loader):
        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(self.epochs):
            logging.info('%s/%s', epoch_counter + 1, self.epochs)

            train_loss = 0.0
            for (xis, xjs), _ in train_loader:
                self.optimizer.zero_grad()

                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(xis, xjs)

                loss.backward()

                train_loss += float(loss.item())

                self.optimizer.step()
                n_iter += 1
            train_loss /= len(train_loader)

            valid_loss = self._validate(val_loader)

            if self.config.general.log_to_wandb:
                wandb.log({'train/loss': train_loss, 'val/loss': valid_loss})

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                model_file_path = 'model_{}_old.pth'.format(self.dataset)
                torch.save(self.model.state_dict(), model_file_path)

            if epoch_counter >= self.warmup_steps:
                self.scheduler.step()

            valid_n_iter += 1

            if epoch_counter % 10 == 0:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.run_folder, str(epoch_counter) + '_' + self.model_name)
                )


class BarlowTwinsAffineTrainer:
    def __init__(self, model, param_head, optimizer, scheduler, config: Config):
        self.model = model
        self.param_head = param_head
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_size = config.trainer.batch_size
        self.epochs = config.trainer.epochs
        self.device = config.trainer.device
        self.dataset = config.data.dataset
        self.run_folder = config.general.output_dir
        self.warmup_steps = config.trainer.warmup_epochs
        self.config = config

        self.criterion = BarlowTwinsLoss()

        self.mse_criterion = nn.MSELoss()

        self.model_name = 'model_{}.pth'.format(self.dataset)

        if config.general.log_to_wandb:
            wandb.init(
                project='phd', config=asdict(config),
                name=f'BarlowTwins Affine : {config.data.dataset}',
                tags=[f'run_id: {config.general.run_id}'],
            )

    def _step(self, xis, xjs, xits, gt_params):
        ris, zis = self.model(xis)  # [N,C]

        # get the representations and the projections
        rjs, zjs = self.model(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self.criterion(zis, zjs)

        rits, _ = self.model(xits)

        if self.config.network.aggregation_strategy == 'diff':
            transition_vector = ris - rits
        elif self.config.network.aggregation_strategy == 'concat':
            transition_vector = torch.cat([ris, rits], dim=-1)
        else:
            raise Exception(f'Invalid aggregation strategy: {self.config.network.aggregation_strategy}')

        params_dist = self.param_head(transition_vector)
        param_loss = self.mse_criterion(params_dist, gt_params)

        return loss, param_loss

    def _validate(self, val_loader):
        with torch.no_grad():
            self.model.eval()
            self.param_head.eval()

            valid_loss = 0.0
            valid_ssl_loss = 0.0
            valid_affine_loss = 0.0
            counter = 0
            for xis, xjs, xits, gt_params in val_loader:
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)
                xits = xits.to(self.device)
                gt_params = gt_params.to(self.device)

                ssl_loss, affine_loss = self._step(xis, xjs, xits, gt_params)
                loss = ssl_loss + affine_loss
                valid_loss += loss.item()
                valid_ssl_loss += ssl_loss.item()
                valid_affine_loss += affine_loss.item()
                counter += 1
            valid_loss /= counter
            valid_ssl_loss /= counter
            valid_affine_loss /= counter
        self.model.train()
        self.param_head.train()

        return {
            'val/loss': valid_loss,
            'val/ssl_loss': ssl_loss,
            'val/affine_loss': affine_loss,
        }

    def train(self, train_loader, val_loader):
        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(self.epochs):
            logging.info('%s/%s', epoch_counter + 1, self.epochs)

            train_loss = 0.0
            train_ssl_loss = 0.0
            train_affine_loss = 0.0
            for xis, xjs, xits, gt_params in train_loader:
                self.optimizer.zero_grad()

                xis = xis.to(self.device)
                xjs = xjs.to(self.device)
                xits = xits.to(self.device)
                gt_params = gt_params.to(self.device)

                ssl_loss, affine_loss = self._step(xis, xjs, xits, gt_params)
                loss = self.config.trainer.beta1 * ssl_loss + self.config.trainer.beta2 * affine_loss

                loss.backward()

                train_loss += float(loss.item())
                train_ssl_loss += float(ssl_loss.item())
                train_affine_loss += float(affine_loss.item())

                self.optimizer.step()
                n_iter += 1
            train_loss /= len(train_loader)
            train_ssl_loss /= len(train_loader)
            train_affine_loss /= len(train_loader)
            train_metrics = {
                'train/loss': train_loss, 'train/ssl_loss': train_ssl_loss, 'train/affine_loss': train_affine_loss
            }

            valid_metrics = self._validate(val_loader)
            valid_loss = valid_metrics['val/loss']

            metrics = deepcopy(train_metrics)
            metrics.update(valid_metrics)
            logging.info(metrics)
            if self.config.general.log_to_wandb:
                wandb.log(metrics)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.run_folder, self.model_name)
                )

            if epoch_counter >= self.warmup_steps:
                self.scheduler.step()

            valid_n_iter += 1

            if epoch_counter % 10 == 0:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.run_folder, str(epoch_counter) + '_' + self.model_name)
                )



class BarlowTwinsDoubleAffineTrainer:
    def __init__(self, model, param_head, optimizer, scheduler, config: Config):
        self.model = model
        self.param_head = param_head
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_size = config.trainer.batch_size
        self.epochs = config.trainer.epochs
        self.device = config.trainer.device
        self.dataset = config.data.dataset
        self.run_folder = config.general.output_dir
        self.warmup_steps = config.trainer.warmup_epochs
        self.config = config

        self.criterion = BarlowTwinsLoss()

        self.mse_criterion = nn.MSELoss()

        self.model_name = 'model_{}.pth'.format(self.dataset)

        if config.general.log_to_wandb:
            wandb.init(
                project='phd', config=asdict(config),
                name=f'BarlowTwins Affine : {config.data.dataset}',
                tags=[f'run_id: {config.general.run_id}'],
            )

    def compute_affine_loss(self, xts, rs, gt_params):
        rts, _ = self.model(xts)
        if self.config.network.aggregation_strategy == 'diff':
            transition_vector = rs - rts
        elif self.config.network.aggregation_strategy == 'concat':
            transition_vector = torch.cat([rs, rts], dim=-1)
        else:
            raise Exception(f'Invalid aggregation strategy: {self.config.network.aggregation_strategy}')
        params_dist = self.param_head(transition_vector)
        param_loss = self.mse_criterion(params_dist, gt_params)
        return param_loss

    def _step(self, xis, xjs, xits, gti_params, xjts, gtj_params):
        ris, zis = self.model(xis)  # [N,C]

        # get the representations and the projections
        rjs, zjs = self.model(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self.criterion(zis, zjs)

        param_loss_i = self.compute_affine_loss(xits, ris, gti_params)
        param_loss_j = self.compute_affine_loss(xjts, rjs, gtj_params)

        return loss + param_loss_i + param_loss_j

    def _validate(self, val_loader):
        with torch.no_grad():
            self.model.eval()
            self.param_head.eval()

            valid_loss = 0.0
            counter = 0
            for xis, xjs, xits, gti_params, xjts, gtj_params in val_loader:
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)
                xits = xits.to(self.device)
                gti_params = gti_params.to(self.device)
                xjts = xjts.to(self.device)
                gtj_params = gtj_params.to(self.device)

                loss = self._step(xis, xjs, xits, gti_params, xjts, gtj_params)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        self.model.train()
        self.param_head.train()

        return valid_loss

    def train(self, train_loader, val_loader):
        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(self.epochs):
            logging.info('%s/%s', epoch_counter + 1, self.epochs)

            train_loss = 0.0
            for xis, xjs, xits, gti_params, xjts, gtj_params in train_loader:
                self.optimizer.zero_grad()

                xis = xis.to(self.device)
                xjs = xjs.to(self.device)
                xits = xits.to(self.device)
                gti_params = gti_params.to(self.device)
                xjts = xjts.to(self.device)
                gtj_params = gtj_params.to(self.device)

                loss = self._step(xis, xjs, xits, gti_params, xjts, gtj_params)

                loss.backward()

                train_loss += float(loss.item())

                self.optimizer.step()
                n_iter += 1
            train_loss /= len(train_loader)

            valid_loss = self._validate(val_loader)

            logging.info({'train/loss': train_loss, 'val/loss': valid_loss})
            if self.config.general.log_to_wandb:
                wandb.log({'train/loss': train_loss, 'val/loss': valid_loss})

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.run_folder, self.model_name)
                )

            if epoch_counter >= self.warmup_steps:
                self.scheduler.step()

            valid_n_iter += 1

            if epoch_counter % 10 == 0:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.run_folder, str(epoch_counter) + '_' + self.model_name)
                )
