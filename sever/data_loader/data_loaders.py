from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .datasets import (
    SteelSegDatasetTrainVal,
    SteelSegDatasetTest,
    SteelClasDatasetTrainVal,
    SteelClasDatasetTest
)


class SteelSegDataLoader(DataLoader):

    train_csv = 'train.csv'
    test_csv  = 'sample_submission.csv'

    def __init__(self, transforms, data_dir, batch_size, shuffle,
                 validation_split, nworkers, pin_memory=True, train=True
    ):  # noqa
        self.transforms, self.shuffle = transforms, shuffle
        self.batch_size, self.nworkers, self.pin_memory = batch_size, nworkers, pin_memory
        self.data_dir = Path(data_dir)

        self.train_df, self.val_df = self.load_df(train, validation_split)

        if train:
            dataset = SteelSegDatasetTrainVal(self.train_df, self.data_dir, transforms.copy(), True)
        else:
            dataset = SteelSegDatasetTest(self.df, self.data_dir, transforms.copy())

        super().__init__(dataset, batch_size, shuffle=shuffle,
                         num_workers=nworkers, pin_memory=pin_memory)

    def load_df(self, train, validation_split):
        csv_filename = self.train_csv if train else self.test_csv
        df = pd.read_csv(self.data_dir / csv_filename)
        df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
        df['ClassId'] = df['ClassId'].astype(int)
        df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
        df.columns = [f'rle{c}' for c in range(4)]
        df['defects'] = df.count(axis=1)

        if train and validation_split > 0:
            return train_test_split(df, test_size=validation_split, stratify=df["defects"])

        return df, pd.DataFrame({})

    def split_validation(self):
        if self.val_df.empty:
            return None
        else:
            dataset = SteelSegDatasetTrainVal(
                self.val_df, self.data_dir, self.transforms.copy(), False)
            return DataLoader(dataset, self.batch_size,
                              num_workers=self.nworkers, pin_memory=self.pin_memory)


class SteelClasDataLoader(DataLoader):

    train_csv = 'train.csv'
    test_csv  = 'sample_submission.csv'

    def __init__(self, transforms, data_dir, batch_size, shuffle,
                 validation_split, nworkers, pin_memory=True, train=True
    ):  # noqa
        self.transforms, self.shuffle = transforms, shuffle
        self.batch_size, self.nworkers, self.pin_memory = batch_size, nworkers, pin_memory
        self.data_dir = Path(data_dir)

        self.train_df, self.val_df = self.load_df(train, validation_split)

        if train:
            dataset = SteelClasDatasetTrainVal(
                self.train_df, self.data_dir, transforms.copy(), True)
        else:
            dataset = SteelClasDatasetTest(self.df, self.data_dir, transforms.copy())

        super().__init__(dataset, batch_size, shuffle=shuffle,
                         num_workers=nworkers, pin_memory=pin_memory)

    def load_df(self, train, validation_split):
        csv_filename = self.train_csv if train else self.test_csv
        df = pd.read_csv(self.data_dir / csv_filename)
        df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
        df['ClassId'] = df['ClassId'].astype(int)
        df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
        df.columns = [f'rle{c}' for c in range(4)]
        df['defects'] = df.count(axis=1)

        # add classification columns
        for c in range(4):
            df[f'c{c}'] = df[f'rle{c}'].apply(lambda rle: not pd.isnull(rle))

        if train and validation_split > 0:
            return train_test_split(df, test_size=validation_split, stratify=df["defects"])

        return df, pd.DataFrame({})

    def split_validation(self):
        if self.val_df.empty:
            return None
        else:
            dataset = SteelClasDatasetTrainVal(
                self.val_df, self.data_dir, self.transforms.copy(), False)
            return DataLoader(dataset, self.batch_size,
                              num_workers=self.nworkers, pin_memory=self.pin_memory)
