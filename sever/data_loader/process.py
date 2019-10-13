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


def rle2mask(mask_rle, shape):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (width,height) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in
                       (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


def make_mask(labels):
    '''Given a row index, return image_id and mask (256, 1600, 4) from the dataframe `df`'''
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
    MIN_COMPONENT_SIZE = 200

    def __init__(self, p_thresh=None, min_class_sizes=None):
        self.p_thresh = self._setup_p_thresh(p_thresh)
        self.min_class_sizes = self.min_class_sizes(min_class_sizes)

    def _component_domination(self, preds):
        """
        Ensure that no predictions in the multi-channel mask overlap. Larger predicted components
        will overwrite overlapping predictions.
        """
        components, component_sizes = self._find_components(preds)
        mask = self._write_preds(components, component_sizes)
        return mask

    def _find_components(self, preds):
        C, H, W = preds.shape
        total_components = 0
        channel_components = []
        for c in range(C):
            max_label, labelled_components = cv2.connectedComponents(preds[c].astype(np.uint8))
            n_components = max_label - 1
            labelled_components[labelled_components > 0] += total_components  # offset labels
            total_components += n_components
            channel_components.append(labelled_components)
        components = np.stack(channel_components, axis=0)
        component_sizes = [(label, (components == label).sum())
                           for label in range(1, total_components + 1)]
        component_sizes = sorted(component_sizes, key=lambda item: item[1])  # sort by size
        return components, component_sizes

    def _write_preds(self, components, component_sizes):
        C, H, W = components.shape
        mask = np.zeros((C, H, W), dtype=np.uint8)
        for label, size in component_sizes:
            if size < self.MIN_COMPONENT_SIZE:
                continue
            component_mask_3d = components == label
            component_mask_flatten = component_mask_3d.any(axis=0)
            component_mask_expand = np.repeat(component_mask_flatten[np.newaxis, :, :], C, axis=0)

            # set the mask region to zero across all channels
            mask[component_mask_expand] = 0

            # set just the channel applicable to the mask to 1
            mask[component_mask_3d] = 1
        return mask

    def process(self, probabilities):
        preds = probabilities > self.p_thresh[:, np.newaxis, np.newaxis]
        mask = self._component_domination(preds)
        mask = self._component_domination(mask)
        for c in range(self.N_CLASSES):
            if mask[c, :, :].sum() < self.min_class_sizes[c]:
                mask[c, :, :] = 0  # wipe the predictions
        return mask

    # -- default value handlers -------------------------------------------------------------------

    def _setup_p_thresh(self, p_thresh):
        if p_thresh is None:
            return np.array([0.5, 0.5, 0.5, 0.5])
        if isinstance(p_thresh, str):
            return np.array([str(p_thresh)] * self.N_CLASSES)
        if isinstance(p_thresh, Sequence):
            if len(p_thresh) != self.N_CLASSES:
                raise Exception(f'Threshold length must be {self.N_CLASSES}. Received {p_thresh}')
            return np.array(p_thresh)
        return np.array([p_thresh] * self.N_CLASSES)

    def min_class_sizes(self, min_sizes):
        if min_sizes is None:
            return np.array([3500, 3500, 3500, 3500])
        if isinstance(min_sizes, str):
            return np.array([str(min_sizes)] * self.N_CLASSES)
        if isinstance(min_sizes, Sequence):
            if len(min_sizes) != self.N_CLASSES:
                raise Exception(f'Threshold length must be {self.N_CLASSES}. Received {min_sizes}')
            return np.array(min_sizes)
        return np.array([min_sizes] * self.N_CLASSES)


class RLE:
    """
    Encapsulates run-length-encoding functionality.
    """

    MASK_H = 256
    MASK_W = 1600

    @classmethod
    def from_str(cls, s):
        if s != s:
            return cls()
        list_ = [int(n) for n in s.split(' ')]
        return cls.from_list(list_)

    @classmethod
    def from_mask(cls, mask):
        pixels = mask.T.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return cls.from_list(runs)

    @classmethod
    def from_list(cls, list_):
        n_items = int(len(list_) / 2)
        items = np.zeros((n_items, 2), dtype=np.uint64)
        for i in range(n_items):
            items[i, 0] = list_[i * 2]
            items[i, 1] = list_[i * 2 + 1]
        return cls(items)

    def __init__(self, items=np.zeros((0, 0))):
        self._items = items

    @property
    def items(self):
        return self._items

    def __iter__(self):
        for idx, item in enumerate(self.items):
            yield (item[0], item[1])  # run, length

    def __len__(self):
        return self.items.shape[0]

    def to_mask(self):
        mask = np.zeros(self.MASK_H * self.MASK_W, dtype=np.uint8)
        for run, length in self:
            run = int(run - 1)
            end = int(run + length)
            mask[run:end] = 1
        return mask.reshape(self.MASK_H, self.MASK_W, order='F')

    def to_str_list(self):
        list_ = []
        for run, length in self:
            list_.append(str(run))
            list_.append(str(length))
        return list_

    def __str__(self):
        if len(self) == 0:
            return ''
        return ' '.join(self.to_str_list())

    def __repr__(self):
        return self.__str__()
