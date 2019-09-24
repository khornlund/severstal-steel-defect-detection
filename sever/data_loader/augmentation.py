import abc

from albumentations import HorizontalFlip, Normalize, Compose
from albumentations.torch import ToTensor
import torchvision.transforms as T

MEAN = [0.485, 0.456, 0.406],
STD = [0.229, 0.224, 0.225],


def get_transforms(train):
    list_transforms = []
    if train == "train":
        list_transforms.extend(
            [
                HorizontalFlip(p=0.5),  # only horizontal flip as of now
            ]
        )
    list_transforms.extend(
        [
            Normalize(mean=MEAN, std=STD),
            ToTensor(),
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms


class AugmentationBase(abc.ABC):

    def __init__(self, train):
        self.train = train
        self.transform = self.build_transforms()

    @abc.abstractmethod
    def build_transforms(self):
        pass

    def __call__(self, images):
        return self.transform(images)


class NormalizeTransforms(AugmentationBase):

    MEANS = (0.1307,)
    STDS  = (0.3081,)

    def __init__(self, train):
        super().__init__(train)

    def build_transforms(self):
        return T.Compose([
            T.ToTensor(),
            T.Normalize(self.MEANS, self.STDS)
        ])
