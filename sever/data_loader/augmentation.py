import abc
import random
from copy import deepcopy

import numpy as np
import cv2

from albumentations.pytorch import ToTensor
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Flip,
    Normalize,
    Compose,
    RandomContrast,
    RandomBrightness,
    RandomSizedCrop,
    Cutout,
    RandomCrop,
    RandomRotate90,
    CropNonEmptyMaskIfExists,
    OneOf,
    ImageOnlyTransform,
    GaussianBlur,
    IAASharpen,
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
            RandomCrop(self.H, self.H),
            HorizontalFlip(p=0.5),
            Normalize(mean=self.MEAN, std=self.STD),
            ToTensor(),
        ])


class RandomCropMediumTransforms(AugmentationBase):

    def __init__(self):
        super().__init__()

    def build_train(self):
        return Compose([
            RandomCrop(self.H, self.H),
            Flip(p=0.5),
            RandomRotate90(p=0.5),
            Normalize(mean=self.MEAN, std=self.STD),
            ToTensor(),
        ])


class RandomCrop256x400Transforms(AugmentationBase):

    def __init__(self):
        super().__init__()

    def build_train(self):
        return Compose([
            RandomCrop(self.H, 416),
            Flip(p=0.5),
            Normalize(mean=self.MEAN, std=self.STD),
            ToTensor(),
        ])


class HeavyCropTransforms(AugmentationBase):

    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width

    def build_train(self):
        return Compose([
            OneOf([
                CropNonEmptyMaskIfExists(self.height, self.width),
                RandomCrop(self.height, self.width)
            ], p=1),
            OneOf([
                CLAHE(p=0.5),  # modified source to get this to work
                GaussianBlur(3, p=0.3),
                IAASharpen(alpha=(0.2, 0.3), p=0.3),
            ], p=1),
            Flip(p=0.5),
            Normalize(mean=self.MEAN, std=self.STD),
            ToTensor(),
        ])


class HeavyCropClasTransforms(AugmentationBase):

    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width

    def build_train(self):
        return Compose([
            OneOf([
                CropNonEmptyMaskIfExists(self.height, self.width),
                RandomCrop(self.height, self.width)
            ], p=1),
            OneOf([
                CLAHE(p=0.5),  # modified source to get this to work
                GaussianBlur(3, p=0.3),
                IAASharpen(alpha=(0.2, 0.3), p=0.3),
            ], p=1),
            Flip(p=0.5),
            Normalize(mean=self.MEAN, std=self.STD),
            ToTensor(),
        ])

    def build_test(self):
        return Compose([
            RandomCrop(self.height, self.width),  # not fully conv, so need to limit img size
            Normalize(mean=self.MEAN, std=self.STD),
            ToTensor(),
        ])


class MaskCropTransforms(AugmentationBase):

    def __init__(self):
        super().__init__()

    def build_train(self):
        return Compose([
            CropNonEmptyMaskIfExists(self.H, self.H),
            Flip(p=0.5),
            RandomRotate90(p=0.5),
            Normalize(mean=self.MEAN, std=self.STD),
            ToTensor(),
        ])


# -- custom --

class CLAHE(ImageOnlyTransform):
    """Apply Contrast Limited Adaptive Histogram Equalization to the input image.

    Args:
        clip_limit (float or (float, float)): upper threshold value for contrast limiting.
            If clip_limit is a single float value, the range will be (1, clip_limit).
        tile_grid_size ((int, int)): size of grid for histogram equalization. Default: (8, 8).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8
    """

    def __init__(self, clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5):
        super(CLAHE, self).__init__(always_apply, p)
        self.clip_limit = to_tuple(clip_limit, 1)
        self.tile_grid_size = tuple(tile_grid_size)

    def apply(self, img, clip_limit=2, **params):
        return clahe(img, clip_limit, self.tile_grid_size)

    def get_params(self):
        return {"clip_limit": random.uniform(self.clip_limit[0], self.clip_limit[1])}

    def get_transform_init_args_names(self):
        return ("clip_limit", "tile_grid_size")


def clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    if img.dtype != np.uint8:
        raise TypeError("clahe supports only uint8 inputs")

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(img.shape) == 2:
        img = clahe.apply(img)
    else:
        img = clahe.apply(img[:, :, 0])
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        # img[:, :, 0] = clahe.apply(img[:, :, 0])
        # img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    return img[:, :, np.newaxis]


def to_tuple(param, low=None, bias=None):
    """Convert input argument to min-max tuple
    Args:
        param (scalar, tuple or list of 2+ elements): Input value.
            If value is scalar, return value would be (offset - value, offset + value).
            If value is tuple, return value would be value + offset (broadcasted).
        low:  Second element of tuple can be passed as optional argument
        bias: An offset factor added to each element
    """
    if low is not None and bias is not None:
        raise ValueError("Arguments low and bias are mutually exclusive")

    if param is None:
        return param

    if isinstance(param, (int, float)):
        if low is None:
            param = -param, +param
        else:
            param = (low, param) if low < param else (param, low)
    elif isinstance(param, (list, tuple)):
        param = tuple(param)
    else:
        raise ValueError("Argument param must be either scalar (int, float) or tuple")

    if bias is not None:
        return tuple([bias + x for x in param])

    return tuple(param)
