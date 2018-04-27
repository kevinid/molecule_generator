from mxnet.gluon.data import Dataset
import itertools
from abc import ABCMeta


__all__ = ['IterableDataset', 'KFold', 'Filter', 'Lambda']

class IterableDataset(Dataset):

    __metaclass__ = ABCMeta

    def __iter__(self):
        return (self[i] for i in range(len(self)))


class KFold(IterableDataset):
    """
    Perform k-fold split of a given dataset
    """

    def __init__(self, dataset, k=5, fold_id=0, is_train=False):
        self.dataset = dataset
        self.k = k
        self.fold_id = fold_id
        self.is_train = is_train

        index = range(len(self.dataset))
        chunk_size = int(float(len(self.dataset))/self.k)
        chunks = []
        for i in range(self.k):
            if i < self.k - 1:
                chunks.append(index[i * chunk_size:(i + 1)*chunk_size])
            else:
                chunks.append(index[i * chunk_size:])

        if self.is_train:
            del chunks[fold_id]
            self.index = list(itertools.chain(*chunks))
        else:
            self.index = chunks[fold_id]

    def __getitem__(self, index):
        return self.dataset[self.index[index]]

    def __len__(self):
        return len(self.index)


class Filter(IterableDataset):
    """
    Perform filtering of a given dataset
    """

    def __init__(self, dataset, fn=lambda _x: True):
        self.fn = fn
        self.records = []

        for i in range(len(dataset)):
            if fn(dataset[i]):
                self.records.append(dataset[i])

    def __getitem__(self, index):
        return self.records[index]

    def __len__(self):
        return len(self.records)


class Lambda(IterableDataset):
    """
    Preprocessing fn
    """

    def __init__(self, dataset, fn=lambda _x: _x):
        self.dataset = dataset
        self.fn = fn

    def __getitem__(self, index):
        return self.fn(self.dataset[index])

    def __len__(self):
        return len(self.dataset)
