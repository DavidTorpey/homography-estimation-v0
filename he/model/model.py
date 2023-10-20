from he.cfg import Config
from he.model.homography_estimator import HomographyEstimator
from he.model.simclr import SimCLR


def get_model(config: Config):
    ssl_algo = config.model.ssl_algo

    if ssl_algo == 'simclr':
        ssl_model = SimCLR(config)
    else:
        raise NotImplementedError(f'SSL algorithm not supported: {ssl_algo}')

    homography_estimator = HomographyEstimator(config, ssl_model.backbone.out_dim)

    return ssl_model, homography_estimator
