from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .datasets import SteelDatasetTrainVal, SteelDatasetTest, SteelDatasetPseudo
from .sampling import SamplerFactory


class SteelDataLoader(DataLoader):

    train_csv = 'train.csv'
    test_csv  = 'sample_submission.csv'

    def __init__(self, transforms, data_dir, batch_size, shuffle,
                 validation_split, nworkers, pin_memory=True, train=True, alpha=None, balance=None
    ):  # noqa
        self.transforms, self.shuffle = transforms, shuffle
        self.bs, self.nworkers, self.pin_memory = batch_size, nworkers, pin_memory
        self.data_dir = Path(data_dir)

        self.train_df, self.val_df = self.load_df(train, validation_split)

        if train:
            dataset = SteelDatasetTrainVal(self.train_df, self.data_dir, transforms.copy(), True)
        else:
            dataset = SteelDatasetTest(self.train_df, self.data_dir, transforms.copy())

        if train and balance is not None and alpha is not None:
            class_idxs = self.sort_classes(self.train_df)
            n_batches = self.train_df.shape[0] // batch_size
            sampler = SamplerFactory(2).get(class_idxs, batch_size, n_batches, alpha, balance)
            super().__init__(dataset, batch_sampler=sampler,
                            num_workers=nworkers, pin_memory=pin_memory)
        else:
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

    def sort_classes(self, df):
        counts = {c: df[f'c{c}'].sum() for c in range(4)}
        sorted_classes = sorted(counts.items(), key=lambda kv: kv[1])

        def assign_min_sample_class(row, sorted_classes):
            for c, _ in sorted_classes:
                if row[f'c{c}']:
                    return c
            return -1

        df['sample_class'] = df.apply(
            lambda row: assign_min_sample_class(row, sorted_classes), axis=1)
        class_idxs = [list(np.where(df['sample_class'] == c)[0]) for c in range(-1, 4)]
        return class_idxs

    def split_validation(self):
        if self.val_df.empty:
            return None
        else:
            dataset = SteelDatasetTrainVal(
                self.val_df, self.data_dir, self.transforms.copy(), False)
            return DataLoader(dataset, self.bs,
                              num_workers=self.nworkers, pin_memory=self.pin_memory)


class SteelPseudoDataLoader(DataLoader):

    train_csv  = 'train.csv'
    pseudo_csv = 'pseudo.csv'

    def __init__(self, transforms, data_dir, batch_size, shuffle,
                 validation_split, nworkers, pin_memory=True, alpha=0
    ):  # noqa
        self.transforms, self.shuffle = transforms, shuffle
        self.bs, self.nworkers, self.pin_memory = batch_size, nworkers, pin_memory
        self.data_dir = Path(data_dir)

        train_df, self.val_df = self.load_df(True, validation_split)
        pseudo_df, _          = self.load_df(False, 0)
        self.train_df = pd.concat([train_df, pseudo_df])

        dataset = SteelDatasetPseudo(self.train_df, self.data_dir, transforms.copy(), True)

        n_train  = train_df.shape[0]
        n_pseudo = pseudo_df.shape[0]
        train_idxs = [idx for idx in range(n_train)]
        pseudo_idxs = [idx + n_train for idx in range(n_pseudo)]
        class_idxs = [train_idxs, pseudo_idxs]

        n_batches = (n_train + n_pseudo) // batch_size
        sampler = SamplerFactory(2).get(class_idxs, batch_size, n_batches, alpha=alpha, kind='fixed')
        super().__init__(dataset, batch_sampler=sampler,
                         num_workers=nworkers, pin_memory=pin_memory)

    def load_df(self, train, validation_split):
        csv_filename = self.train_csv if train else self.pseudo_csv
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
            dataset = SteelDatasetPseudo(
                self.val_df, self.data_dir, self.transforms.copy(), False)
            return DataLoader(dataset, self.bs,
                              num_workers=self.nworkers, pin_memory=self.pin_memory)
