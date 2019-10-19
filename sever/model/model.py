import torch
import torch.nn as nn

from sever.base import BaseModel

from segmentation_models_pytorch import *  # noqa
from efficientnet_pytorch import EfficientNet  # noqa
from efficientnet_pytorch.utils import get_same_padding_conv2d


class EffNet(BaseModel):
    """
    https://github.com/lukemelas/EfficientNet-PyTorch
    """
    def __init__(self, encoder_name, classes, pretrained, in_channels='rgb', verbose=0):
        super().__init__(verbose)
        if pretrained:
            self.encoder = EfficientNet.from_pretrained(encoder_name, num_classes=classes)
        else:
            self.encoder = EfficientNet.from_name(
                encoder_name,
                override_params={'num_classes': classes})

        # modify input channels
        Conv2d = get_same_padding_conv2d(image_size=self.encoder._global_params.image_size)
        conv_stem = Conv2d(
            len(in_channels),
            self.encoder._conv_stem.out_channels,
            kernel_size=3,
            stride=2,
            bias=False
        )
        transfer_weights(self.encoder._conv_stem, conv_stem, in_channels)
        self.encoder._conv_stem = conv_stem

        self.decoder = Noop()  # for compatibility with segmentation models
        self.logger.info(f'<init>: \n{self}')

    def forward(self, x):
        return self.encoder(x)

    def __str__(self):
        return str(self.encoder)

    def __repr__(self):
        return self.__str__()


class Noop(nn.Module):

    def forward(self, x):
        return x


def select_rgb_weights(weights, rgb_str):
    """Repeat RGB weights given a str eg. RRGGBB would repeat each weight twice"""
    rgb_str = rgb_str.lower()
    rgb_map = {'r': 0, 'g': 1, 'b': 2}
    slices = [(rgb_map[c] % 3, rgb_map[c] % 3 + 1) for c in rgb_str]  # slice a:a+1 to keep dims
    new_weights = torch.cat([
        weights[:, a:b, :, :] for a, b in slices
    ], dim=1)
    return new_weights


def transfer_weights(pretrained_layer, replacement_layer, rgb_str='rgb'):
    """
    Transform pretrained weights to be used for a layer with a different number of channels.
    """
    weights = select_rgb_weights(pretrained_layer.weight, rgb_str)
    replacement_layer.weight = nn.Parameter(weights)
    return replacement_layer
