import logging
import os
from dataclasses import asdict

import numpy as np
import torch
import wandb
from lightly.loss import NTXentLoss
from torch import nn
from torch.optim import Optimizer

from he.cfg import Config
from he.constants import LATEST_MODEL_FILE_NAME


class Trainer:
    def __init__(self, ssl_model, homography_estimator, optimiser: Optimizer, lr_schedule, config: Config):
        self.ssl_model = ssl_model
        self.homography_estimator = homography_estimator
        self.optimiser = optimiser
        self.lr_schedule = lr_schedule
        self.config = config

        self.nt_xent_loss = NTXentLoss()

        if config.model.affine_loss == 'mse':
            self.affine_loss_fn = nn.MSELoss()
        else:
            raise NotImplementedError(f'Affine loss not supported: {config.model.affine_loss}')

        if config.general.log_to_wandb:
            wandb.init(
                project='phd', config=asdict(config),
                name=f'SimCLR Homography Estimation : {config.data.dataset}',
                tags=[f'run_id: {config.general.run_id}'],
            )

    def aggregate_vectors(self, z, za):
        if self.config.model.aggregation_strategy == 'concat':
            affine_representation = torch.cat([z, za], dim=-1)
        elif self.config.model.aggregation_strategy == 'diff':
            affine_representation = z - za
        elif self.config.model.aggregation_strategy == 'sum':
            affine_representation = z + za
        elif self.config.model.aggregation_strategy == 'mean':
            affine_representation = (z + za) / 2.0
        else:
            raise NotImplementedError(
                f'Aggregation strategy not supported: {self.config.model.aggregation_strategy}'
            )

        return affine_representation

    def train_one_epoch(self, train_loader, epoch):
        train_loss = 0.0
        train_affine_loss = 0.0
        train_contrastive_loss = 0.0
        for batch_num, batch in enumerate(train_loader):
            global_iteration = len(train_loader) * epoch + batch_num

            self.optimiser.param_groups[0]['lr'] = self.lr_schedule[global_iteration]

            x1t, x1at, affine_params1, x2t = batch

            x1t = x1t.to(self.config.optim.device)
            x1at = x1at.to(self.config.optim.device)
            affine_params1 = affine_params1.to(self.config.optim.device)

            x2t = x2t.to(self.config.optim.device)

            z1, h1 = self.ssl_model(x1t)
            za1, ha1 = self.ssl_model(x1at)
            z2, h2 = self.ssl_model(x2t)

            contrastive_loss = self.nt_xent_loss(z1, z2)

            affine_representation = self.aggregate_vectors(h1, ha1)
            affine_params_pred = self.homography_estimator(affine_representation)
            affine_loss = self.affine_loss_fn(affine_params_pred, affine_params1)

            loss = (
                    self.config.optim.contrastive_loss_weight * contrastive_loss +
                    self.config.optim.affine_loss_weight * affine_loss
            )

            loss = loss / self.config.optim.grad_acc_steps

            train_contrastive_loss += float(contrastive_loss.item())
            train_affine_loss += float(affine_loss.item())
            train_loss += float(loss.item())

            loss.backward()

            if ((batch_num + 1) % self.config.optim.grad_acc_steps == 0) or ((batch_num + 1) == len(train_loader)):
                self.optimiser.step()
                self.optimiser.zero_grad()

        train_contrastive_loss /= len(train_loader)
        train_affine_loss /= len(train_loader)
        train_loss /= len(train_loader)

        return {
            'train/contrastive_loss': train_contrastive_loss,
            'train/affine_loss': train_affine_loss,
            'train/loss': train_loss,
        }

    def validate_one_epoch(self, val_loader):
        self.ssl_model.eval()
        self.homography_estimator.eval()

        val_loss = 0.0
        val_affine_loss = 0.0
        val_contrastive_loss = 0.0
        with torch.no_grad():
            for batch_num, batch in enumerate(val_loader):
                x1t, x1at, affine_params1, x2t = batch

                x1t = x1t.to(self.config.optim.device)
                x1at = x1at.to(self.config.optim.device)
                affine_params1 = affine_params1.to(self.config.optim.device)

                x2t = x2t.to(self.config.optim.device)

                z1, h1 = self.ssl_model(x1t)
                za1, ha1 = self.ssl_model(x1at)
                z2, h2 = self.ssl_model(x2t)

                contrastive_loss = self.nt_xent_loss(z1, z2)

                affine_representation = self.aggregate_vectors(h1, ha1)
                affine_params_pred = self.homography_estimator(affine_representation)
                affine_loss = self.affine_loss_fn(affine_params_pred, affine_params1)

                loss = (
                        self.config.optim.contrastive_loss_weight * contrastive_loss +
                        self.config.optim.affine_loss_weight * affine_loss
                )

                loss = loss / self.config.optim.grad_acc_steps

                val_contrastive_loss += float(contrastive_loss.item())
                val_affine_loss += float(affine_loss.item())
                val_loss += float(loss.item())

        val_contrastive_loss /= len(val_loader)
        val_affine_loss /= len(val_loader)
        val_loss /= len(val_loader)

        self.ssl_model.train()
        self.homography_estimator.train()

        return {
            'train/contrastive_loss': val_contrastive_loss,
            'train/affine_loss': val_affine_loss,
            'train/loss': val_loss,
        }

    def train(self, train_loader, val_loader):
        best_val_loss = np.inf

        for epoch in range(self.config.optim.epochs):
            logging.info('Epoch %s/%s', epoch + 1, self.config.optim.epochs)

            train_metrics = self.train_one_epoch(train_loader, epoch)

            val_metrics = self.validate_one_epoch(val_loader)

            if self.config.general.log_to_wandb:
                wandb.log({**train_metrics, **val_metrics})

            logging.info({**train_metrics, **val_metrics})

            val_loss = val_metrics['val/loss']

            state_dict = {
                'ssl_model': self.ssl_model.state_dict(),
                'homography_estimator': self.homography_estimator.state_dict(),
                'optimiser': self.optimiser.state_dict(),
                'epoch': epoch + 1,
            }

            torch.save(
                state_dict,
                os.path.join(self.config.general.output_dir, LATEST_MODEL_FILE_NAME)
            )

            if epoch % self.config.general.checkpoint_freq == 0 or (epoch + 1) == self.config.optim.epochs:
                torch.save(
                    state_dict,
                    os.path.join(self.config.general.output_dir, f'simclr-epoch-{epoch}.pth')
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    state_dict,
                    os.path.join(self.config.general.output_dir, 'best.pth')
                )

        return best_val_loss
