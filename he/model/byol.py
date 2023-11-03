from torch import nn

from he.configuration import Config
from he.model.backbone import Encoder
from he.model.projection_head import MLPHead


class BYOL(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.online_network = Encoder(config)

        self.target_network = Encoder(config)

        self.initializes_target_network()

        self.predictor = MLPHead(
            in_channels=512 if config.network.name == 'resnet18' else 2048,
            hidden_size=config.network.mlp_head.hidden_size,
            proj_size=config.network.mlp_head.proj_size
        )

    def initializes_target_network(self):
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
