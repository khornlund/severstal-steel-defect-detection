import cv2
from torch.utils.data import Dataset

from .process import make_mask
from .augmentation import get_transforms


# TODO: using cv2 means mean/std for imagenet will be different

class SteelDatasetTrainVal(Dataset):

    def __init__(self, df, data_dir, train):
        self.df = df
        self.data_dir = data_dir / 'train_images'
        self.train = train
        self.transforms = get_transforms(train)
        self.fnames = self.df.index.tolist()

    def __getitem__(self, idx):
        image_id, mask = make_mask(idx, self.df)
        image_path = str(self.data_dir / image_id)
        img = cv2.imread(image_path)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']  # 1x256x1600x4
        mask = mask[0].permute(2, 0, 1)  # 1x4x256x1600
        return img, mask

    def __len__(self):
        return len(self.fnames)


class SteelDatasetTest(Dataset):

    def __init__(self, df, data_dir):
        self.df = df
        self.data_dir = data_dir / 'test_images'
        self.transforms = get_transforms(False)
        self.fnames = self.df.index.tolist()

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = str(self.data_dir / fname)
        image = cv2.imread(path)
        images = self.transform(image=image)["image"]
        return fname, images

    def __len__(self):
        return len(self.fnames)
