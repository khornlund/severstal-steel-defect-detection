from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .datasets import SteelDatasetTrainVal, SteelDatasetTest


class SteelDataLoader(DataLoader):

    train_csv = 'train.csv'
    test_csv  = 'sample_submission.csv'

    def __init__(self, data_dir, batch_size, shuffle, validation_split, nworkers, train=True):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.nworkers = nworkers

        self.train_df, self.val_df = self.load_df(train, validation_split)

        if train:
            dataset = SteelDatasetTrainVal(self.train_df, self.data_dir, True)
        else:
            dataset = SteelDatasetTest(self.df, self.data_dir)
        self.n_samples = len(dataset)
        super().__init__(dataset, batch_size, shuffle, num_workers=nworkers)

    def load_df(self, train, validation_split):
        csv_filename = self.train_csv if train else self.test_csv
        df = pd.read_csv(self.data_dir / csv_filename)
        df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
        df['ClassId'] = df['ClassId'].astype(int)
        df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
        df['defects'] = df.count(axis=1)

        if train and validation_split > 0:
            # df = df.loc[df.defects > 0, :]  # only train on images with defects
            return train_test_split(df, test_size=validation_split, stratify=df["defects"])

        return df, pd.DataFrame({})

    def split_validation(self):
        if self.val_df.empty:
            return None
        else:
            dataset = SteelDatasetTrainVal(self.val_df, self.data_dir, False)
            return DataLoader(dataset, self.batch_size, num_workers=self.nworkers)
