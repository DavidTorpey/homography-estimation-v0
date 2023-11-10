from dataclasses import dataclass, field


@dataclass
class General:
    output_dir: str

@dataclass
class MLP:
    hidden_size: int = 512
    proj_size: int = 128


@dataclass
class Network:
    algo: str = 'simclr'
    name: str = 'resnet18'
    mlp_head: MLP = field(default_factory=lambda: MLP())
    pred_head: MLP = field(default_factory=lambda: MLP())


@dataclass
class Data:
    dataset: str
    root: str
    num_classes: int
    dataset_type: str
    s: float = 1.0
    image_size: int = 32


@dataclass
class Trainer:
    batch_size: int = 256
    epochs: int = 100
    warmup_epochs: int = 10
    num_workers: int = 4
    device: str = 'cpu'


@dataclass
class Optimiser:
    lr: float = 3e-4
    weight_decay: float = 10e-6


@dataclass
class Config:
    general: General
    data: Data
    network: Network = field(default_factory=lambda: Network())
    trainer: Trainer = field(default_factory=lambda: Trainer())
    optimiser: Optimiser = field(default_factory=lambda: Optimiser())
