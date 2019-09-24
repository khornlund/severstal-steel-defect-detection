import torch.nn as nn
import numpy as np

from sever.utils import setup_logger


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    def __init__(self, verbose=0):
        super().__init__()
        self.logger = setup_logger(self, verbose=verbose)

    def forward(self, *input):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + f'\nTrainable parameters: {params}'
