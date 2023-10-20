from torch import nn

from he.cfg import Config


class HomographyEstimator(nn.Module):
    def __init__(self, config: Config, encoder_dim):
        super().__init__()
        if config.model.aggregation_strategy == 'concat':
            self.homography_estimator = nn.Sequential(
                nn.Linear(2 * encoder_dim, config.model.hidden_dim),
                nn.BatchNorm1d(config.model.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(config.model.hidden_dim, 6)
            )
        elif config.model.aggregation_strategy in ['diff', 'sum', 'mean']:
            self.homography_estimator = nn.Sequential(
                nn.Linear(encoder_dim, config.model.hidden_dim),
                nn.BatchNorm1d(config.model.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(config.model.hidden_dim, 6)
            )

    def forward(self, x):
        return self.homography_estimator(x)
