from he.cfg import Config


def get_model(config: Config):
    ssl_algo = config.model.ssl_algo

    if ssl_algo == 'simclr':
        ...
    else:
        raise NotImplementedError(f'SSL algorithm not supported: {ssl_algo}')

    return ssl_model, homography_estimator