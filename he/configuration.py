from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class General:
    output_dir: str
    log_to_wandb: bool = False
    run_id: str = str(uuid4())


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
    aggregation_strategy: str = 'diff'


@dataclass
class Data:
    dataset: str
    root: str
    num_classes: int
    dataset_type: str
    s: float = 1.0
    image_size: int = 32

    rotation: bool = True
    translation: bool = True
    scale: bool = True
    shear: bool = True


@dataclass
class Trainer:
    batch_size: int = 256
    epochs: int = 100
    warmup_epochs: int = 10
    num_workers: int = 4
    device: str = 'cpu'

    logreg_steps: int = 10
    models_in_parallel: int = 4
    n_jobs: int = 2


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
