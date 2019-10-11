import numpy as np
from torch.utils.data.sampler import BatchSampler, WeightedRandomSampler

from sever.utils import setup_logger


class SamplerFactory:
    """
    Factory class to create balanced samplers.
    """

    def __init__(self, verbose=0):
        self.logger = setup_logger(self, verbose)

    def get(self, class_idxs, batch_size, n_batches, alpha, kind):
        """
        Parameters
        ----------
        class_idxs : 2D list of ints
            List of sample indices for each class. Eg. [[0, 1], [2, 3]] implies indices 0, 1
            belong to class 0, and indices 2, 3 belong to class 1.
        batch_size : int
            The batch size to use.
        n_batches : int
            The number of batches per epoch.
        alpha : numeric in range [0, 1]
            Weighting term used to determine weights of each class in each batch.
            When `alpha` == 0, the batch class distribution will approximate the training population
            class distribution.
            When `alpha` == 1, the batch class distribution will approximate a uniform distribution,
            with equal number of samples from each class.
        kind : str ['fixed' | 'random']
            The kind of sampler. `Fixed` will ensure each batch contains a constant proportion of
            samples from each class. `Random` will simply sample with replacement according to the
            calculated weights.
        """
        if kind == 'random':
            return self.random(class_idxs, batch_size, n_batches, alpha)
        if kind == 'fixed':
            return self.fixed(class_idxs, batch_size, n_batches, alpha)
        raise Exception(f'Received kind {kind}, must be `random` or `fixed`')

    def random(self, class_idxs, batch_size, n_batches, alpha):
        self.logger.info(f'Creating `{WeightedRandomBatchSampler.__class__.__name__}`...')
        class_sizes, weights = self._weight_classes(class_idxs, alpha)
        sample_rates = self._sample_rates(weights, class_sizes)
        return WeightedRandomBatchSampler(sample_rates, class_idxs, batch_size, n_batches)

    def fixed(self, class_idxs, batch_size, n_batches, alpha):
        self.logger.info(f'Creating `{WeightedFixedBatchSampler.__class__.__name__}`...')
        class_sizes, weights = self._weight_classes(class_idxs, alpha)
        class_samples_per_batch = self._fix_batches(weights, class_sizes, batch_size, n_batches)
        return WeightedFixedBatchSampler(class_samples_per_batch, class_idxs, n_batches)

    def _weight_classes(self, class_idxs, alpha):
        class_sizes = np.asarray([len(idxs) for idxs in class_idxs])
        n_samples = class_sizes.sum()
        n_classes = len(class_idxs)

        original_weights = np.asarray([size / n_samples for size in class_sizes])
        uniform_weights = np.repeat(1 / n_classes, n_classes)

        self.logger.info(f'Sample population absolute class sizes: {class_sizes}')
        self.logger.info(f'Sample population relative class sizes: {original_weights}')

        weights = self._balance_weights(uniform_weights, original_weights, alpha)
        return class_sizes, weights

    def _balance_weights(self, weight_a, weight_b, alpha):
        assert alpha >= 0 and alpha <= 1, f'invalid alpha {alpha}, must be 0 <= alpha <= 1'
        beta = 1 - alpha
        weights = (alpha * weight_a) + (beta * weight_b)
        self.logger.info(f'Target batch class distribution {weights} using alpha={alpha}')
        return weights

    def _sample_rates(self, weights, class_sizes):
        return weights / class_sizes

    def _fix_batches(self, weights, class_sizes, batch_size, n_batches):
        """
        Calculates the number of samples of each class to include in each batch, and the number
        of batches required to use all the data in an epoch.
        """
        class_samples_per_batch = np.round((weights * batch_size)).astype(int)

        # cleanup rounding edge-cases
        remainder = batch_size - class_samples_per_batch.sum()
        largest_class = np.argmax(class_samples_per_batch)
        class_samples_per_batch[largest_class] += remainder

        assert class_samples_per_batch.sum() == batch_size

        proportions_of_class_per_batch = class_samples_per_batch / batch_size
        self.logger.info(f'Rounded batch class distribution {proportions_of_class_per_batch}')

        proportions_of_samples_per_batch = class_samples_per_batch / class_sizes

        self.logger.info(f'Expecting {class_samples_per_batch} samples of each class per batch, '
                         f'over {n_batches} batches of size {batch_size}')

        oversample_rates = proportions_of_samples_per_batch * n_batches
        self.logger.info(f'Sampling rates: {oversample_rates}')

        return class_samples_per_batch


class WeightedRandomBatchSampler(BatchSampler):
    """
    Samples with replacement according to the provided weights.
    Parameters
    ----------
    class_weights : `numpy.array(int)`
        The number of samples of each class to include in each batch.
    class_idxs : 2D list of ints
        The indices that correspond to samples of each class.
    batch_size : int
        The size of each batch yielded.
    n_batches : int
        The number of batches to yield.
    """

    def __init__(self, class_weights, class_idxs, batch_size, n_batches):
        self.sample_idxs = []
        for idxs in class_idxs:
            self.sample_idxs.extend(idxs)

        sample_weights = []
        for c, weight in enumerate(class_weights):
            sample_weights.extend([weight] * len(class_idxs[c]))

        self.sampler = WeightedRandomSampler(sample_weights, batch_size, replacement=True)
        self.n_batches = n_batches

    def __iter__(self):
        for bidx in range(self.n_batches):
            selected = []
            for idx in self.sampler:
                selected.append(self.sample_idxs[idx])
            yield selected

    def __len__(self):
        return self.n_batches


class WeightedFixedBatchSampler(BatchSampler):
    """
    Ensures each batch contains a given class distribution.
    The lists of indices for each class are shuffled at the start of each call to `__iter__`.
    Parameters
    ----------
    class_samples_per_batch : `numpy.array(int)`
        The number of samples of each class to include in each batch.
    class_idxs : 2D list of ints
        The indices that correspond to samples of each class.
    n_batches : int
        The number of batches to yield.
    """

    def __init__(self, class_samples_per_batch, class_idxs, n_batches):
        self.class_samples_per_batch = class_samples_per_batch
        self.class_idxs = [CircularList(idx) for idx in class_idxs]
        self.n_batches = n_batches

        self.n_classes = len(self.class_samples_per_batch)
        self.batch_size = self.class_samples_per_batch.sum()

        assert len(self.class_samples_per_batch) == len(self.class_idxs)
        assert isinstance(self.n_batches, int)

    def _get_batch(self, start_idxs):
        selected = []
        for c, size in enumerate(self.class_samples_per_batch):
            selected.extend(self.class_idxs[c][start_idxs[c]:start_idxs[c] + size])
        np.random.shuffle(selected)
        return selected

    def __iter__(self):
        [cidx.shuffle() for cidx in self.class_idxs]
        start_idxs = np.zeros(self.n_classes, dtype=int)
        for bidx in range(self.n_batches):
            yield self._get_batch(start_idxs)
            start_idxs += self.class_samples_per_batch

    def __len__(self):
        return self.n_batches


class CircularList:
    """
    Applies modulo function to indexing.
    """
    def __init__(self, items):
        self._items = items
        self._mod = len(self._items)
        self.shuffle()

    def shuffle(self):
        np.random.shuffle(self._items)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self[i] for i in range(key.start, key.stop)]
        return self._items[key % self._mod]
