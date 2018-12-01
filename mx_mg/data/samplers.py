import numpy as np
import random
from mxnet.gluon.data.sampler import Sampler


__all__ = ['BalancedSampler']

class BalancedSampler(Sampler):

    def __init__(self, cost, batch_size):
        # random.shuffle(cost)
        index = np.argsort(cost).tolist()
        chunk_size = int(float(len(cost))/batch_size)
        self.index = []
        for i in range(batch_size):
            self.index.append(index[i*chunk_size:(i + 1)*chunk_size])

    def _g(self):
        # shuffle data
        for index_i in self.index:
            random.shuffle(index_i)

        for batch_index in zip(*self.index):
            yield batch_index

    def __iter__(self):
        return self._g()

    def __len__(self):
        return len(self.index[0])
