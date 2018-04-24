import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from . import utils

class GraphLoader(DataLoader):
    """Load graph based molecule representation from SMILES"""
    def __init__(self, dataset, batch_size=10, num_workers=0,
                 k=10, p=0.9, conditional=None, shuffle=False, sampler=None):
        self.k = k
        self.p = p
        self.conditional = conditional

        super(GraphLoader, self).__init__(dataset, batch_size, shuffle=shuffle, sampler=sampler, batch_sampler=None,
                                          num_workers=num_workers, collate_fn=self._collate_fn,
                                          pin_memory=False, drop_last=True, worker_init_fn=None)


    def _collate_fn(self, batch):
        # names = X, A,
        #         NX, NA,
        #         mol_ids, rep_ids, iw_ids,
        #         action_0, actions,
        #         last_append_mask,
        #         log_p

        shapes = [[0], [0, 3],
                  [0], [0],
                  [0], [0], [0],
                  [0], [0, 5],
                  [0],
                  [0]]
        dtypes = [np.int32, np.int32,
                  np.int32, np.int32,
                  np.int32, np.int32, np.int32,
                  np.int32, np.int32,
                  np.int32,
                  np.float32]

        _build = lambda: [np.zeros(shape=s, dtype=d) for s, d in zip(shapes, dtypes)]
        _append = lambda _r0, _r1: [np.concatenate([__r0, __r1], axis=0)
                                    for __r0, __r1 in zip(_r0, _r1)]

        X, A, \
        NX, NA, \
        mol_ids, rep_ids, iw_ids, \
        action_0, actions, \
        last_append_mask, \
        log_p = _build()

        if self.conditional is not None:
            c = []
        else:
            c = None

        for i, record_in in enumerate(batch):
            if self.conditional is not None:
                assert callable(self.conditional)
                record_in = self.conditional(record_in)
                smiles = record_in[0]
                c_i = [float(val) for val in record_in[1:]]
                c.append(c_i)
            else:
                smiles = record_in

            X_i, A_i, \
            NX_i, NA_i, \
            mol_ids_i, rep_ids_i, iw_ids_i, \
            action_0_i, actions_i, \
            last_append_mask_i, log_p_i = utils.process_single(smiles, self.k, self.p)

            if i != 0:
                mol_ids_i += mol_ids[-1] + 1
                iw_ids_i += iw_ids[-1] + 1

            X, A, \
            NX, NA, \
            mol_ids, rep_ids, iw_ids, \
            action_0, actions, \
            last_append_mask, \
            log_p = _append([X, A,
                             NX, NA,
                             mol_ids, rep_ids, iw_ids,
                             action_0, actions,
                             last_append_mask,
                             log_p],
                            [X_i, A_i,
                             NX_i, NA_i,
                             mol_ids_i, rep_ids_i, iw_ids_i,
                             action_0_i, actions_i,
                             last_append_mask_i,
                             log_p_i])

        X, A, \
        mol_ids_rep, rep_ids_rep, iw_ids, \
        last_append_mask, \
        NX, NX_rep, \
        action_0, actions, \
        log_p = utils.merge_single(X, A,
                                   NX, NA,
                                   mol_ids, rep_ids, iw_ids,
                                   action_0, actions,
                                   last_append_mask,
                                   log_p)

        result_out = [X, A,
                      mol_ids_rep, rep_ids_rep, iw_ids,
                      last_append_mask,
                      NX, NX_rep,
                      action_0, actions,
                      log_p]

        if c is not None:
            c = np.array(c, dtype=np.float32)
            result_out.append(c)

        return result_out

    @staticmethod
    def from_numpy_to_tensor(record, device_id):
        """Convert numpy to tensor and place it to a specific device"""
        [X, A,
         mol_ids_rep, rep_ids_rep, iw_ids,
         last_append_mask,
         NX, NX_rep,
         action_0, actions,
         log_p] = record

        X = Variable(torch.from_numpy(X).long()).cuda(device_id)
        A_sparse = []
        for A_i in A:
            if A_i.shape[0] == 0:
                # pytorch does not support zero dimensions
                # should be reflected in GraphConvFn
                A_sparse_i = None
            else:
                A_sparse_i = torch.cuda.sparse.FloatTensor(torch.from_numpy(A_i).long().t().cuda(device_id),
                                                           torch.ones([A_i.shape[0]]).cuda(device_id),
                                                           torch.Size([X.size(0), ]*2))
                A_sparse_i = A_sparse_i + A_sparse_i.t()
            A_sparse.append(A_sparse_i)

        batch_size, iw_size = np.asscalar(mol_ids_rep.max() + 1), \
                              np.asscalar(rep_ids_rep.max() + 1)

        mol_ids_rep, rep_ids_rep, iw_ids, \
        last_append_mask, \
        NX, NX_rep, action_0, actions = [Variable(torch.from_numpy(_x).long()).cuda(device_id)
                                         for _x in [mol_ids_rep, rep_ids_rep, iw_ids,
                                                    last_append_mask,
                                                    NX, NX_rep, action_0, actions]]

        log_p = Variable(torch.from_numpy(log_p).float()).cuda(device_id)

        record = [X, A_sparse, iw_ids, last_append_mask,
                  NX, NX_rep, action_0, actions, log_p,
                  batch_size, iw_size, mol_ids_rep, rep_ids_rep]

        return record
