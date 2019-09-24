from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .datasets import SteelDataset


class SteelDataLoader(DataLoader):

    train_csv = 'train.csv'

    def __init__(self, data_dir, batch_size, shuffle, validation_split, nworkers, train=True):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.nworkers = nworkers

        df = pd.read_csv(self.data_dir / self.train_csv)
        df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
        df['ClassId'] = df['ClassId'].astype(int)
        df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
        df['defects'] = df.count(axis=1)

        self.train_df, self.val_df = train_test_split(
            df, test_size=validation_split, stratify=df["defects"])

        dataset = SteelDataset(df, self.data_dir, train)
        self.n_samples = len(dataset)
        super().__init__(dataset, batch_size, shuffle, num_workers=nworkers)

    def split_validation(self):
        if self.val_df.empty:
            return None
        else:
            dataset = SteelDataset(self.val_df, self.data_dir, False)
            return DataLoader(dataset, self.batch_size, num_workers=self.nworkers)
