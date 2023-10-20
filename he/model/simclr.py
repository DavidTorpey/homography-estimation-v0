from lightly.models.modules import SimCLRProjectionHead
from torch import nn

from he.cfg import Config
from he.model.backbone import get_backbone


class SimCLR(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.backbone = get_backbone(config)

        self.projection_head = SimCLRProjectionHead(
            self.backbone.out_dim,
            config.model.hidden_dim,
            config.model.proj_dim
        )

    def forward(self, x):
        h = self.backbone(x)
        z = self.projection_head(h)
        return z, h
