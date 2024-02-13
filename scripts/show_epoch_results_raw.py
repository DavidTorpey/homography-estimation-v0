import os

import pandas as pd

N = 5

def get_metric(p):
    if not os.path.exists(p):
        return

    s = open(p).read().splitlines()[:N]
    print(len(s))
    s = [float(e) for e in s]
    return s

template = 'results/resnet50/{model}/tiny_imagenet/metrics-{dataset}-{model_id}.txt'

data = []
for model in ['simclr', 'simclr-affine', 'byol', 'byol-affine', 'barlow_twins', 'barlow_twins-affine', 'barlow_twins-affine-investigate-losses']:
    for epoch in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
        for dataset in ['cifar10', 'cifar100', 'caltech101']:
            model_id = f'{epoch}_model_tiny_imagenet'
            metrics = get_metric(template.format(model=model, dataset=dataset, model_id=model_id))
            for metric in metrics:
                data.append([model, epoch, dataset, metric])
df = pd.DataFrame(data, columns=['model', 'epoch', 'dataset', 'metric'])
df.to_csv('epoch-results-raw.csv', index=False)


