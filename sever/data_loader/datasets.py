import cv2
from torch.utils.data import Dataset

from .process import make_mask


class SteelDataset(Dataset):

    img_folder = 'implement me!'
    N_CLASSES = 4
    rle_cols = [f'rle{i}' for i in range(N_CLASSES)]

    def __init__(self, df, data_dir, transforms):
        self.df = df
        self.data_dir = data_dir / self.img_folder
        self.transforms = transforms
        self.fnames = self.df.index.tolist()

    def read_greyscale(self, idx):
        f = self.fnames[idx]
        return f, cv2.imread(str(self.data_dir / f))[:, :, 0:1]  # select one channel

    def rle(self, idx):
        return self.df.iloc[idx][self.rle_cols]

    def __len__(self):
        return len(self.fnames)


class SteelDatasetPseudo(SteelDataset):

    img_folder = 'joined_images'

    def __init__(self, df, data_dir, transforms, train):
        super().__init__(df, data_dir, transforms)
        self.transforms.build_transforms(train=train)

    def __getitem__(self, idx):
        mask = make_mask(self.rle(idx))
        _, img = self.read_greyscale(idx)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']  # 1x256x1600x4
        mask = mask[0].permute(2, 0, 1)  # 1x4x256x1600
        return img, mask


class SteelDatasetTrainVal(SteelDataset):

    img_folder = 'train_images'

    def __init__(self, df, data_dir, transforms, train):
        super().__init__(df, data_dir, transforms)
        self.transforms.build_transforms(train=train)

    def __getitem__(self, idx):
        mask = make_mask(self.rle(idx))
        _, img = self.read_greyscale(idx)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']  # 1x256x1600x4
        mask = mask[0].permute(2, 0, 1)  # 1x4x256x1600
        return img, mask


class SteelDatasetTest(SteelDataset):

    img_folder = 'test_images'

    def __init__(self, df, data_dir, transforms):
        super().__init__(df, data_dir, transforms)
        self.transforms.build_transforms(train=False)

    def __getitem__(self, idx):
        f, image = self.read_greyscale(idx)
        images = self.transforms(image=image)["image"]
        return f, images
