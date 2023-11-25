import logging
import os
from dataclasses import asdict

import wandb
import numpy as np
import torch
import torch.nn.functional as F
from lightly.loss import BarlowTwinsLoss
from torch import nn

from he.configuration import Config
from he.nt_xent import NTXentLoss


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

                train_loss += float(loss.item(0))

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
        transition_vector = ris - rits
        params_dist = self.param_head(transition_vector)
        param_loss = self.mse_criterion(params_dist, gt_params)

        return loss + param_loss

    def _validate(self, val_loader):
        with torch.no_grad():
            self.model.eval()
            self.param_head.eval()

            valid_loss = 0.0
            counter = 0
            for xis, xjs, xits, gt_params in val_loader:
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)
                xits = xits.to(self.device)
                gt_params = gt_params.to(self.device)

                loss = self._step(xis, xjs, xits, gt_params)
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
            for xis, xjs, xits, gt_params in train_loader:
                self.optimizer.zero_grad()

                xis = xis.to(self.device)
                xjs = xjs.to(self.device)
                xits = xits.to(self.device)
                gt_params = gt_params.to(self.device)

                loss = self._step(xis, xjs, xits, gt_params)

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
