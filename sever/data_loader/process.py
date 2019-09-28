from collections import Sequence

import numpy as np
import cv2


# https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
def mask2rle(img):
    '''
    img: numpy array, 1 -> mask, 0 -> background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def make_mask(row_id, df):
    '''Given a row index, return image_id and mask (256, 1600, 4) from the dataframe `df`'''
    labels = df.iloc[row_id][:4]
    masks = np.zeros((256, 1600, 4), dtype=np.float32)  # float32 is V.Imp

    for idx, label in enumerate(labels.values):
        if label == label:  # NaN check
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            mask = np.zeros(256 * 1600, dtype=np.uint8)
            for pos, le in zip(positions, length):
                mask[pos:(pos + le)] = 1
            masks[:, :, idx] = mask.reshape(256, 1600, order='F')
    return masks


class PostProcessor:

    N_CLASSES = 4

    def __init__(self, thresholds=None, min_sizes=None):
        self.thresholds = self._setup_thresholds(thresholds)
        self.min_sizes  = self._setup_min_sizes(thresholds)

    def process(self, class_, probability):
        '''Post processing of each predicted mask, components with lesser number of pixels
        than `min_size` are ignored'''
        pr_th = self.thresholds[class_]
        sz_th = self.min_sizes[class_]

        mask = cv2.threshold(probability, pr_th, 1, cv2.THRESH_BINARY)[1]
        n_components, labels = cv2.connectedComponents(mask.astype(np.uint8))
        predictions = np.zeros((256, 1600), np.float32)
        num = 0
        for component in range(1, n_components):
            p = (labels == component)
            if p.sum() > sz_th:
                predictions[p] = 1
                num += 1
        return predictions, num

    # -- default value handlers -------------------------------------------------------------------

    def _setup_thresholds(self, thresholds):
        if thresholds is None:
            return [0.5, 0.5, 0.5, 0.5]
        if isinstance(thresholds, str):
            return [str(thresholds)] * self.N_CLASSES
        if isinstance(thresholds, Sequence):
            if len(thresholds) != self.N_CLASSES:
                raise Exception(f'Threshold length must be {self.N_CLASSES}. Received {thresholds}')
            return thresholds
        return [thresholds] * self.N_CLASSES

    def _setup_min_sizes(self, min_sizes):
        if min_sizes is None:
            return [3500, 3500, 3500, 3500]
        if isinstance(min_sizes, str):
            return [str(min_sizes)] * self.N_CLASSES
        if isinstance(min_sizes, Sequence):
            if len(min_sizes) != self.N_CLASSES:
                raise Exception(f'Threshold length must be {self.N_CLASSES}. Received {min_sizes}')
            return min_sizes
        return [min_sizes] * self.N_CLASSES
