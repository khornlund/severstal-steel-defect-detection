from efficientnet_pytorch import EfficientNet
from segmentation_models_pytorch import *

from sever.base import BaseModel


class EffNet(BaseModel):
    """
    https://github.com/lukemelas/EfficientNet-PyTorch
    """
    def __init__(self, num_classes, pretrained, model='b2', verbose=0):
        super().__init__(verbose)
        model_name = f'efficientnet-{model}'
        if pretrained:
            self.model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
        else:
            self.model = EfficientNet.from_name(
                model_name,
                override_params={'num_classes': num_classes})

        self.logger.info(f'<init>: \n{self}')

    def forward(self, x):
        return self.model(x)

    def __str__(self):
        return str(self.model)

    def __repr__(self):
        return self.__str__()
