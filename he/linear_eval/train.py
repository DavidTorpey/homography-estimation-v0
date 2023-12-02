import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import wandb
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader

from he.configuration import Config


def train_and_test(config: Config, C, x_train, y_train, x_val, y_val):
    logging.info('Training logistic regression model: C=%s', C)

    classifier = LogisticRegression(
        solver='lbfgs', multi_class='multinomial', warm_start=True, C=C,
        n_jobs=config.trainer.n_jobs
    )

    test_score = classifier.fit(x_train, y_train).score(x_val, y_val)

    logging.info('Accuracy for logistic regression (C=%s): %s', C, test_score)

    return test_score, C


class Trainer:
    def __init__(
            self, model: nn.Module, config: Config
    ):
        self.model = model
        self.config = config

    def compute_features_for_loader(self, loader: DataLoader):
        with torch.no_grad():
            features = []
            targets = []
            for i, (x, y) in enumerate(loader):
                if i % 10 == 0:
                    logging.info('Computing features: %s/%s', i + 1, len(loader))
                x = x.to(self.config.trainer.device)
                x = self.model(x).detach().cpu().numpy()
                features.append(x)

                y = y.cpu().numpy()
                targets.append(y)

        return np.vstack(features), np.hstack(targets)

    def logreg_hpo(self, xtr, ytr, xval, yval, wd_range) -> float:
        with ThreadPoolExecutor(max_workers=self.config.trainer.models_in_parallel) as executor:
            futures = []
            for wd in wd_range:
                C = 1. / wd.item()

                future = executor.submit(
                    train_and_test,
                    self.config, C, xtr, ytr, xval, yval
                )
                futures.append(future)

            results = []
            for future in futures:
                results.append(future.result())

        test_scores, Cs = zip(*results)
        test_scores = np.array(test_scores)
        Cs = np.array(Cs)

        df = pd.DataFrame()
        df['Accuracy'] = test_scores
        df['C'] = Cs

        best_C = Cs[test_scores.argmax()]

        if self.config.general.log_to_wandb:
            table = wandb.Table(columns=['Accuracy', 'C'], data=df.values.tolist())
            wandb.log({'logistic_regression_sweep': table, 'best_C': best_C})

        return best_C

    def find_best_model(self, xtr, ytr, xval, yval):
        wd_range = torch.logspace(-6, 5, self.config.trainer.logreg_steps)

        logging.info('#Train=%s\t#Val=%s', len(xtr), len(xval))

        best_C = self.logreg_hpo(xtr, ytr, xval, yval, wd_range)

        logging.info('Best C: %s', best_C)

        logistic_regression = LogisticRegression(
            solver='lbfgs', multi_class='multinomial',
            warm_start=True, C=best_C,
            n_jobs=self.config.trainer.n_jobs
        )
        logistic_regression.fit(xtr, ytr)

        return logistic_regression

    def train(self, train_loader, val_loader):
        xtr, ytr = self.compute_features_for_loader(train_loader)
        xval, yval = self.compute_features_for_loader(val_loader)

        logistic_regression = self.find_best_model(xtr, ytr, xval, yval)

        return logistic_regression

    def test(self, test_loader, logistic_regression):
        xte, yte = self.compute_features_for_loader(test_loader)

        accuracy = logistic_regression.score(xte, yte)

        with open(os.path.join(self.config.general.output_dir, 'test_predictions.json'), 'w') as file:
            json.dump(
                {
                    'predictions': logistic_regression.predict(xte).tolist(),
                    'ground_truth': yte.tolist()
                },
                file
            )

        logging.info('Testing accuracy for best model: %s', accuracy)

        if self.config.general.log_to_wandb:
            wandb.log({'Test Accuracy': accuracy})

        metrics_file_name = f'metrics-{self.config.data.dataset}.json'

        result_file_path = os.path.join(
            self.config.general.output_dir,
            metrics_file_name
        )
        with open(result_file_path, 'w') as file:
            json.dump({'test_accuracy': accuracy}, file, indent=2)
