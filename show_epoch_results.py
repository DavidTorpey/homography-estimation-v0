import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import scipy.stats


def ci(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return round(m * 100, 2), round(h * 100, 2)


def get_metric(p, N, use_ci):
    if not os.path.exists(p):
        return

    s = open(p).read().splitlines()[:N]
    print(len(s))
    s = [float(e) for e in s]
    mean, variability = ci(s)
    if use_ci:
        return f'{mean} \u00B1 {variability}'
    return mean


def main():
    p = ArgumentParser()
    p.add_argument('--N', type=int, default=5)
    p.add_argument('--ci', action='store_true')
    args = p.parse_args()

    template = 'results/resnet50/{model}/tiny_imagenet/metrics-{dataset}-{model_id}.txt'

    data = []
    for model in ['simclr', 'simclr-affine', 'byol', 'byol-affine', 'barlow_twins', 'barlow_twins-affine',
                  'barlow_twins-affine-investigate-losses']:
        for epoch in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
            for dataset in ['cifar10', 'cifar100', 'caltech101']:
                model_id = f'{epoch}_model_tiny_imagenet'
                metric = get_metric(template.format(model=model, dataset=dataset, model_id=model_id), args.N, args.ci)
                data.append([model, epoch, dataset, metric])
    df = pd.DataFrame(data, columns=['model', 'epoch', 'dataset', 'metric'])
    df.to_csv('epoch-results.csv', index=False)


if __name__ == '__main__':
    main()

