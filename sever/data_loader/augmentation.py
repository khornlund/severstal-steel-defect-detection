import abc
from copy import deepcopy

from albumentations.torch import ToTensor
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Normalize,
    Compose,
    RandomContrast,
    RandomBrightness,
    RandomSizedCrop,
    Cutout,
    RandomCrop,
)


class AugmentationBase(abc.ABC):

    MEAN = [0.3439]
    STD  = [0.0383]

    H = 256
    W = 1600

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

    def build_test(self):
        return Compose([
            Normalize(mean=self.MEAN, std=self.STD),
            ToTensor(),
        ])

    def notimplemented(self, *args, **kwargs):
        raise Exception('You must call `build_transforms()` before using me!')

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    def copy(self):
        return deepcopy(self)


class LightTransforms(AugmentationBase):

    def __init__(self):
        super().__init__()

    def build_train(self):
        return Compose([
            HorizontalFlip(p=0.5),
            Normalize(mean=self.MEAN, std=self.STD),
            ToTensor(),
        ])


class MediumTransforms(AugmentationBase):

    def __init__(self):
        super().__init__()

    def build_train(self):
        return Compose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            Normalize(mean=self.MEAN, std=self.STD),
            RandomContrast(p=0.1),
            RandomBrightness(p=0.1),
            ToTensor(),
        ])


class HeavyTransforms(AugmentationBase):

    def __init__(self):
        super().__init__()

    def build_train(self):
        return Compose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            Normalize(mean=self.MEAN, std=self.STD),
            RandomContrast(p=0.2),
            RandomBrightness(p=0.2),
            RandomSizedCrop((240, 256), self.H, self.W, w2h_ratio=1600 / 256),
            Cutout(max_h_size=32, max_w_size=32),
            ToTensor(),
        ])


class RandomCropTransforms(AugmentationBase):

    def __init__(self):
        super().__init__()

    def build_train(self):
        return Compose([
            RandomCrop(self.H, self.H * 2),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            Normalize(mean=self.MEAN, std=self.STD),
            ToTensor(),
        ])

    def build_test(self):
        return Compose([
            RandomCrop(self.H, self.H * 2),
            Normalize(mean=self.MEAN, std=self.STD),
            ToTensor(),
        ])
