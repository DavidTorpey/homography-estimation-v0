from dataclasses import dataclass, field
from uuid import uuid4

import dacite
import yaml


@dataclass
class General:
    output_dir: str
    log_to_wandb: bool = False
    run_id: str = str(uuid4())
    checkpoint_freq: int = 10


@dataclass
class Data:
    dataset: str
    train_aug: str = 'simclr'
    val_aug: str = 'simclr'
    max_images: int = None
    image_size: int = 224

    crop_after_transform: bool = False
    max_shear: int = 10
    max_degrees: int = 30
    max_translate: float = 0.1
    min_scale: float = 0.9
    max_scale: float = 1.1


@dataclass
class Optim:
    lr: float = 3e-4
    warmup_epochs: int = 1
    epochs: int = 1
    batch_size: int = 32
    workers: int = 1
    device: str = 'cuda'


@dataclass
class Backbone:
    name: str = 'resnet50'


@dataclass
class Model:
    ssl_algo: str = 'simclr'
    backbone: Backbone = field(default_factory=lambda: Backbone())
    hidden_dim: int = 2048
    proj_dim: int = 128
    aggregation_strategy: str = 'concat'


@dataclass
class Config:
    data: Data
    optim: Optim = field(default_factory=lambda: Optim())
    model: Model = field(default_factory=lambda: Model())


def load_config(path):
    with open(path) as file:
        data = yaml.safe_load(file)

    return dacite.from_dict(Config, data)
