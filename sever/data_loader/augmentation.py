import abc
from copy import deepcopy

from albumentations import (HorizontalFlip, VerticalFlip, Normalize, Compose, RandomContrast,
                            RandomBrightness)
from albumentations.torch import ToTensor


class AugmentationBase(abc.ABC):

    def __init__(self):
        self.transform = self.notimplemented

    def build_transforms(self, train):
        if train:
            self.transform = self.build_train()
        else:
            self.transform = self.build_test()

    @abc.abstractmethod
    def build_train(self):
        pass

    @abc.abstractmethod
    def build_test(self):
        pass

    def notimplemented(self, *args, **kwargs):
        raise Exception('You must call `build_transforms()` before using me!')

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    def copy(self):
        return deepcopy(self)


class LightTransforms(AugmentationBase):

    MEAN = [0.3439]
    STD  = [0.0383]

    def __init__(self):
        super().__init__()

    def build_train(self):
        return Compose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            Normalize(mean=self.MEAN, std=self.STD),
            ToTensor(),
        ])

    def build_test(self):
        return Compose([
            Normalize(mean=self.MEAN, std=self.STD),
            ToTensor(),
        ])


class MediumTransforms(AugmentationBase):

    MEAN = [0.3439]
    STD  = [0.0383]

    def __init__(self):
        super().__init__()

    def build_train(self):
        return Compose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            Normalize(mean=self.MEAN, std=self.STD),
            RandomContrast(p=0.2),
            RandomBrightness(p=0.2),
            ToTensor(),
        ])

    def build_test(self):
        return Compose([
            Normalize(mean=self.MEAN, std=self.STD),
            ToTensor(),
        ])
