import torch
from torchvision import models

from .projection_head import MLPHead
from ..configuration import Config


class ResNetSimCLR(torch.nn.Module):
    def __init__(self, config: Config):
        super(ResNetSimCLR, self).__init__()
        name = config.network.name
        if name == 'resnet18':
            resnet = models.resnet18(pretrained=False)
        elif name == 'resnet50':
            resnet = models.resnet50(pretrained=False)
        elif name == 'resnet50_2':
            resnet = models.wide_resnet50_2(pretrained=False)

        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.projection = MLPHead(
            in_channels=resnet.fc.in_features,
            hidden_size=config.network.mlp_head.hidden_size,
            proj_size=config.network.mlp_head.proj_size
        )

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return h, self.projection(h)
