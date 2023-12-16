import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import scipy.stats
from terminaltables import AsciiTable as AT


def ci(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return round(m * 100, 2), round(h * 100, 2)


def get_combo_results(combo, mm, dd, N, use_ci):
    f = f'results/resnet50/{mm}/tiny_imagenet/affine-components-ablations/{combo}/metrics-{dd}.txt'
    if not os.path.exists(f):
        print('dne', f)
        return
    s = open(f).read().splitlines()[:N]
    s = [float(e) for e in s]
    print(len(s))
    mean, variability = ci(s)
    if use_ci:
        return f'{mean} \u00B1 {variability}'
    return mean


def main():
    p = ArgumentParser()
    p.add_argument('--N', type=int, default=5)
    p.add_argument('--ci', action='store_true')
    args = p.parse_args()

    data = [['model', 'dataset', 'trans', 'shear', 'rot', 'scale']]
    for m in ['simclr-affine', 'byol-affine', 'barlow_twins-affine', 'barlow_twins-affine-investigate-losses']:
        for d in ['cifar10', 'cifar100', 'caltech101']:
            translation = get_combo_results('translation', m, d, args.N, args.ci)
            shear = get_combo_results('shear', m, d, args.N, args.ci)
            rotation = get_combo_results('rotation', m, d, args.N, args.ci)
            scale = get_combo_results('scale', m, d, args.N, args.ci)

            data.append([m, d, translation, shear, rotation, scale])

    print(AT(data).table)

    df = pd.DataFrame(data[1:], columns=data[0])
    print(df.to_csv(index=False))


if __name__ == '__main__':
    main()

