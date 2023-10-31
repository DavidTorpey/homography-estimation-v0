import sys

import torch
import torchvision
import yaml
from sklearn import preprocessing
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np

from .model.backbone import ResNetSimCLR

batch_size = 512
data_transforms = torchvision.transforms.Compose([transforms.ToTensor()])

config = yaml.load(open(sys.argv[-1], "r"), Loader=yaml.FullLoader)
dataset = config['data']['dataset']
print('Using dataset:', dataset)

num_classes = config['data']['num_classes']

if dataset == 'stl10':
    train_dataset = datasets.STL10('./data', split='train', download=True,
                                   transform=data_transforms)

    test_dataset = datasets.STL10('./data', split='test', download=True,
                                  transform=data_transforms)
elif dataset == 'cifar10':
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=data_transforms)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=data_transforms)
elif dataset == 'svhn':
    train_dataset = datasets.SVHN('./data', split='train', download=True, transform=data_transforms)
    test_dataset = datasets.SVHN('./data', split='test', download=True, transform=data_transforms)
elif dataset == 'cifar100':
    train_dataset = datasets.CIFAR100('./data', train=True, download=True, transform=data_transforms)
    test_dataset = datasets.CIFAR100('./data', train=False, download=True, transform=data_transforms)

print("Input shape:", train_dataset[0][0].shape)

train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          num_workers=0, drop_last=False, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         num_workers=0, drop_last=False, shuffle=True)

device = 'cpu'  # 'cuda' if torch.cuda.is_available() else 'cpu'
encoder = ResNetSimCLR(**config['network'])
output_feature_dim = encoder.projection.net[0].in_features

model_file_path = 'model_{}_old.pth'.format(dataset)
load_params = torch.load(
    model_file_path,
    map_location=torch.device(torch.device(device))
)
encoder.load_state_dict(load_params)

encoder = encoder.encoder
encoder = encoder.to(device)

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


logreg = LogisticRegression(output_feature_dim, num_classes)
logreg = logreg.to(device)


def get_features_from_encoder(encoder, loader):
    x_train = []
    y_train = []

    # get the features from the pre-trained model
    for i, (x, y) in enumerate(loader):
        with torch.no_grad():
            feature_vector = encoder(x)
            x_train.extend(feature_vector)
            y_train.extend(y.numpy())

    x_train = torch.stack(x_train)
    y_train = torch.tensor(y_train)
    return x_train, y_train


encoder.eval()
x_train, y_train = get_features_from_encoder(encoder, train_loader)
x_test, y_test = get_features_from_encoder(encoder, test_loader)

if len(x_train.shape) > 2:
    x_train = torch.mean(x_train, dim=[2, 3])
    x_test = torch.mean(x_test, dim=[2, 3])
print(x_train.shape, x_test.shape)

print("Training data shape:", x_train.shape, y_train.shape)
print("Testing data shape:", x_test.shape, y_test.shape)


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test):
    train = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)

    test = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test, batch_size=512, shuffle=False)
    return train_loader, test_loader


scaler = preprocessing.StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train).astype(np.float32)
x_test = scaler.transform(x_test).astype(np.float32)

train_loader, test_loader = create_data_loaders_from_arrays(torch.from_numpy(x_train), y_train,
                                                            torch.from_numpy(x_test), y_test)

optimizer = torch.optim.Adam(logreg.parameters(), lr=3e-4)
criterion = torch.nn.CrossEntropyLoss()
eval_every_n_epochs = 10

for epoch in range(200):
    #     train_acc = []
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        logits = logreg(x)
        predictions = torch.argmax(logits, dim=1)

        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

    total = 0
    if epoch % eval_every_n_epochs == 0:
        correct = 0
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            logits = logreg(x)
            predictions = torch.argmax(logits, dim=1)

            total += y.size(0)
            correct += (predictions == y).sum().item()

        acc = 100 * correct / total
        print(f"Testing accuracy: {np.mean(acc)}")

