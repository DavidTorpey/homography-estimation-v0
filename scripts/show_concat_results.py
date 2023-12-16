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
        print('dne', p)
        return None

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

    template = 'results/resnet50/{model}/tiny_imagenet/concat/metrics-{dataset}.txt'

    data = [['model', 'cifar10', 'cifar100', 'caltech101']]
    for m in ['simclr-affine', 'byol-affine', 'barlow_twins-affine', 'barlow_twins-affine-updated-betas']:
        cifar10 = get_metric(template.format(model=m, dataset='cifar10'), args.N, args.ci)
        cifar100 = get_metric(template.format(model=m, dataset='cifar100'), args.N, args.ci)
        caltech101 = get_metric(template.format(model=m, dataset='caltech101'), args.N, args.ci)
        data.append([m, cifar10, cifar100, caltech101])

    df = pd.DataFrame(data[1:], columns=data[0])
    print(df.to_csv(index=False))


if __name__ == '__main__':
    main()

