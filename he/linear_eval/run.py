import os
from argparse import ArgumentParser

import dacite
import torch
import yaml
from torch.utils.data import DataLoader
import numpy as np

from ..configuration import Config
from ..model.backbone import Encoder
from .model import LogisticRegression
from .data import get_datasets
from .utils import get_numpy_data
from ..model.byol import BYOL

batch_size = 512
device = 'cuda'

parser = ArgumentParser()
parser.add_argument('--config_path', type=str, required=True)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--run_num', type=str, default=None)
args = parser.parse_args()

config_dict = yaml.load(open(args.config_path, "r"), Loader=yaml.FullLoader)
config: Config = dacite.from_dict(Config, config_dict)

if args.run_num:
    config.general.output_dir = os.path.join(config.general.output_dir, args.run_num)

run_folder = config.general.output_dir

dataset = config.data.dataset
print('Using dataset:', dataset)

num_classes = config.data.num_classes

train_dataset, val_dataset, test_dataset = get_datasets(dataset, config.data.image_size)

print("Input shape:", train_dataset[0][0].shape)

train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          num_workers=0, drop_last=False, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size,
                        num_workers=0, drop_last=False, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         num_workers=0, drop_last=False, shuffle=True)

if config.network.algo == 'simclr':
    encoder = Encoder(config)
    output_feature_dim = encoder.projection.net[0].in_features
    model_file_path = args.model_path
    load_params = torch.load(
        model_file_path,
        map_location=torch.device(device)
    )
    encoder.load_state_dict(load_params)
    encoder = encoder.encoder
    encoder = encoder.to(device)
elif config.network.algo == 'byol':
    byol = BYOL(config)
    model_file_path = args.model_path
    load_params = torch.load(
        model_file_path,
        map_location=torch.device(device)
    )
    byol.load_state_dict(load_params)
    encoder = byol.online_network.encoder
    output_feature_dim = 2048 if config.network.name == 'resnet50' else 512
    encoder = encoder.to(device)
else:
    raise Exception(f'Invalid model: {config.network.algo}')

x_train, y_train, x_val, y_val, x_test, y_test = \
    get_numpy_data(
        encoder, train_loader, val_loader, test_loader, device
    )

print("Training data shape:", x_train.shape, y_train.shape)
print("Val data shape:", x_val.shape, y_val.shape)
print("Testing data shape:", x_test.shape, y_test.shape)


def create_data_loaders_from_arrays(X_train, y_train, X_val, y_val, X_test, y_test):
    train = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)

    val = torch.utils.data.TensorDataset(X_val, y_val)
    val_loader = torch.utils.data.DataLoader(val, batch_size=64, shuffle=True)

    test = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test, batch_size=512, shuffle=False)
    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = create_data_loaders_from_arrays(
    torch.from_numpy(x_train), torch.from_numpy(y_train),
    torch.from_numpy(x_val), torch.from_numpy(y_val),
    torch.from_numpy(x_test), torch.from_numpy(y_test)
)


logreg = LogisticRegression(output_feature_dim, num_classes)
logreg = logreg.to(device)

optimizer = torch.optim.Adam(logreg.parameters(), lr=3e-4)
criterion = torch.nn.CrossEntropyLoss()
eval_every_n_epochs = 10

best_acc = 0
for epoch in range(100):
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = logreg(x)
        predictions = torch.argmax(logits, dim=1)

        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

    total = 0
    if epoch % eval_every_n_epochs == 0:
        correct = 0
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            logits = logreg(x)
            predictions = torch.argmax(logits, dim=1)

            total += y.size(0)
            correct += (predictions == y).sum().item()

        acc = correct / total
        print(f"Val accuracy: {100 * np.mean(acc)}")

        print(acc, best_acc)
        if acc > best_acc:
            best_acc = acc
            torch.save(logreg.state_dict(), os.path.join(run_folder, 'logreg.pth'))

logreg.load_state_dict(torch.load(os.path.join(run_folder, 'logreg.pth')))
logreg.eval()

total = 0
correct = 0
for x, y in test_loader:
    x = x.to(device)
    y = y.to(device)

    logits = logreg(x)
    predictions = torch.argmax(logits, dim=1)

    total += y.size(0)
    correct += (predictions == y).sum().item()

acc = correct / total
print(f"Testing accuracy: {100 * np.mean(acc)}")

model_id = os.path.basename(args.model_path).split('.')[0]

with open(os.path.join(run_folder, f'{model_id}_result.txt'), 'w') as f:
    f.write('{}\n'.format(acc))
