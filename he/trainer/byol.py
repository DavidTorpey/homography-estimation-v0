import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from he.configuration import Config
from he.model.byol import BYOL


class BYOLTrainer:
    def __init__(self, model: BYOL, optimizer, scheduler, config: Config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_size = config.trainer.batch_size
        self.epochs = config.trainer.epochs
        self.device = config.trainer.device
        self.dataset = config.data.dataset
        self.run_folder = config.general.output_dir
        self.warmup_steps = config.trainer.warmup_epochs

        self.m_base = 0.996
        self.m = 0.996

        self.model_name = 'model_{}.pth'.format(self.dataset)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return -2 * (x * y).sum(dim=-1)

    def _step(self, batch_view_1, batch_view_2):
        predictions_from_view_1 = self.model.predictor(self.model.online_network(batch_view_1)[1])
        predictions_from_view_2 = self.model.predictor(self.model.online_network(batch_view_2)[1])

        with torch.no_grad():
            targets_to_view_2 = self.model.target_network(batch_view_1)[1]
            targets_to_view_1 = self.model.target_network(batch_view_2)[1]

        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
        loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)

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

    @torch.no_grad()
    def _update_target_network_parameters(self):
        for param_q, param_k in zip(self.model.online_network.parameters(), self.model.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def train(self, train_loader, val_loader):
        K = len(train_loader) * self.epochs
        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(self.epochs):
            logging.info('%s/%s', epoch_counter + 1, self.epochs)
            for (xis, xjs), _ in train_loader:
                self.optimizer.zero_grad()

                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(xis, xjs)

                loss.backward()

                self.optimizer.step()
                n_iter += 1

                self._update_target_network_parameters()

                self.m = 1 - (1 - self.m_base) * (np.cos(np.pi * n_iter / K) + 1) / 2

            valid_loss = self._validate(val_loader)
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


class BYOLAffineTrainer:
    def __init__(self, model: BYOL, param_head, optimizer, scheduler, config: Config):
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

        self.m_base = 0.996
        self.m = 0.996

        self.mse_criterion = nn.MSELoss()

        self.model_name = 'model_{}.pth'.format(self.dataset)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return -2 * (x * y).sum(dim=-1)

    def _step(self, xis, xjs, xits, gt_params):
        ris1, zis1 = self.model.online_network(xis)
        predictions_from_view_1 = self.model.predictor(zis1)

        ris2, zis2 = self.model.online_network(xjs)
        predictions_from_view_2 = self.model.predictor(zis2)

        with torch.no_grad():
            _, targets_to_view_2 = self.model.target_network(xis)
            _, targets_to_view_1 = self.model.target_network(xjs)

        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
        loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)

        rits, _ = self.model.online_network(xits)
        transition_vector = ris1 - rits
        params_dist = self.param_head(transition_vector)
        param_loss = self.mse_criterion(params_dist, gt_params)

        return loss.mean() + param_loss

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

    @torch.no_grad()
    def _update_target_network_parameters(self):
        for param_q, param_k in zip(self.model.online_network.parameters(), self.model.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def train(self, train_loader, val_loader):
        K = len(train_loader) * self.epochs

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(self.epochs):
            logging.info('%s/%s', epoch_counter + 1, self.epochs)
            for xis, xjs, xits, gt_params in train_loader:
                self.optimizer.zero_grad()

                xis = xis.to(self.device)
                xjs = xjs.to(self.device)
                xits = xits.to(self.device)
                gt_params = gt_params.to(self.device)

                loss = self._step(xis, xjs, xits, gt_params)

                loss.backward()

                self.optimizer.step()
                n_iter += 1

                self._update_target_network_parameters()

                self.m = 1 - (1 - self.m_base) * (np.cos(np.pi * n_iter / K) + 1) / 2

            valid_loss = self._validate(val_loader)
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
