import os

import cv2
from torch.utils.data import Dataset

from .process import make_mask
from .augmentation import get_transforms


class SteelDataset(Dataset):

    def __init__(self, df, data_folder, train):
        self.df = df
        self.root = data_folder
        self.train = train
        self.transforms = get_transforms(train)
        self.fnames = self.df.index.tolist()

    def __getitem__(self, idx):
        image_id, mask = make_mask(idx, self.df)
        image_path = os.path.join(self.root, "train_images", image_id)
        img = cv2.imread(image_path)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']  # 1x256x1600x4
        mask = mask[0].permute(2, 0, 1)  # 1x4x256x1600
        return img, mask

    def __len__(self):
        return len(self.fnames)
