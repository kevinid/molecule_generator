import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet.gluon.data import DataLoader
from rdkit import Chem

import utils
from mx_mg.data import data_struct


__all__ = ['MolLoader', 'MolRNNLoader', 'CMolRNNLoader']

class MolLoader(DataLoader):
    """Load graph based molecule representation from SMILES"""
    def __init__(self, dataset, batch_size=10, num_workers=0,
                 k=10, p=0.9, shuffle=False, sampler=None, batch_sampler=None):
        self.k = k
        self.p = p

        # batch_sampler, sampler and shuffle are mutually exclusive
        if batch_sampler is not None:
            super(MolLoader, self).__init__(dataset, batch_sampler=batch_sampler,
                                            num_workers=num_workers, batchify_fn=self._collate_fn)
        elif sampler is not None:
            super(MolLoader, self).__init__(dataset, sampler=sampler,
                                            num_workers=num_workers, batchify_fn=self._collate_fn,
                                            last_batch='rollover')
        else:
            super(MolLoader, self).__init__(dataset, batch_size, shuffle=shuffle,
                                            num_workers=num_workers, batchify_fn=self._collate_fn,
                                            last_batch='rollover')


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


        for i, record_in in enumerate(batch):
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

        X = nd.array(X, ctx=mx.gpu(device_id), dtype='int32')
        A_sparse = []
        for A_i in A:
            if A_i.shape[0] == 0:
                A_sparse.append(None)
            else:
                # transpose may not be supported in gpu
                A_i = np.concatenate([A_i, A_i[:, [1, 0]]], axis=0)

                # construct csr matrix ...
                data = np.ones((A_i.shape[0], ), dtype=np.float32)
                row, col = A_i[:, 0], A_i[:, 1]
                A_sparse_i = nd.sparse.csr_matrix((data, (row, col)),
                                                  shape=tuple([int(X.shape[0]), ]*2),
                                                  ctx=mx.gpu(device_id),
                                                  dtype='float32')

                # append to list
                A_sparse.append(A_sparse_i)

        batch_size, iw_size = np.asscalar(mol_ids_rep.max() + 1), \
                              np.asscalar(rep_ids_rep.max() + 1)

        mol_ids_rep, rep_ids_rep, iw_ids, \
        last_append_mask, \
        NX, NX_rep, action_0, actions = [nd.array(_x, ctx=mx.gpu(device_id), dtype='int32')
                                         for _x in [mol_ids_rep, rep_ids_rep, iw_ids,
                                                    last_append_mask,
                                                    NX, NX_rep, action_0, actions]]

        log_p = nd.array(log_p, ctx=mx.gpu(device_id), dtype='float32')

        record = [X, A_sparse, iw_ids, last_append_mask,
                  NX, NX_rep, action_0, actions, log_p,
                  batch_size, iw_size]


        return record


class MolRNNLoader(MolLoader):

    def _collate_fn(self, batch):
        result_out = super(MolRNNLoader, self)._collate_fn(batch)

        # things ready for rnn
        mol_list = [Chem.MolFromSmiles(batch_i) for batch_i in batch]
        # preparing mapping
        graph_to_rnn = np.zeros((len(batch), self.k, data_struct.get_mol_spec().max_iter), dtype=np.int32)
        rnn_to_graph = []
        cum_sum = 0
        for i, mol_i in enumerate(mol_list):
            num_iter = mol_i.GetNumBonds() + 1
            for k in range(self.k):
                graph_to_rnn[i, k, :num_iter] = (np.arange(num_iter) + cum_sum)

                rnn_to_graph_0 = np.ones([num_iter,], dtype=np.int32) * i
                rnn_to_graph_1 = np.ones_like(rnn_to_graph_0) * k
                rnn_to_graph_2 = np.arange(num_iter)
                rnn_to_graph.append(np.stack([rnn_to_graph_0, rnn_to_graph_1, rnn_to_graph_2], axis=0))

                cum_sum += num_iter
        rnn_to_graph = np.concatenate(rnn_to_graph, axis=1)
        NX_cum = np.cumsum(result_out[6])

        result_out = result_out + [graph_to_rnn, rnn_to_graph, NX_cum]

        return result_out

    @staticmethod
    def from_numpy_to_tensor(record, device_id):
        [X, A,
         mol_ids_rep, rep_ids_rep, iw_ids,
         last_append_mask,
         NX, NX_rep,
         action_0, actions,
         log_p,
         graph_to_rnn, rnn_to_graph, NX_cum] = record

        output = MolLoader.from_numpy_to_tensor([X, A,
                                                 mol_ids_rep, rep_ids_rep, iw_ids,
                                                 last_append_mask,
                                                 NX, NX_rep,
                                                 action_0, actions,
                                                 log_p],
                                                device_id)

        graph_to_rnn, rnn_to_graph, NX_cum =\
            nd.array(graph_to_rnn, ctx=mx.gpu(device_id), dtype='int32'),\
            nd.array(rnn_to_graph, ctx=mx.gpu(device_id), dtype='int32'), \
            nd.array(NX_cum, ctx=mx.gpu(device_id), dtype='int32')

        output = output + [graph_to_rnn, rnn_to_graph, NX_cum]

        return output


class CMolRNNLoader(MolRNNLoader):

    def __init__(self, dataset, batch_size=10, num_workers=0,
                 k=10, p=0.9, shuffle=False, sampler=None, batch_sampler=None,
                 conditional=None):
        if conditional is None:
            raise ValueError('Conditional function is not set, '
                             'use unconditional version instead')
        if not callable(conditional):
            raise TypeError('Provided condition is not callable')

        self.conditional = conditional

        super(CMolRNNLoader, self).__init__(dataset, batch_size, num_workers,
                                            k, p, shuffle, sampler, batch_sampler)

    def _collate_fn(self, batch):
        smiles_list, c = [], []
        for record_i in batch:
            smiles_i, c_i = self.conditional(record_i)
            smiles_list.append(smiles_i)
            c.append(c_i)
        c = np.stack(c, axis=0)

        output = super(CMolRNNLoader, self)._collate_fn(smiles_list)
        output.append(c)

        return output

    @staticmethod
    def from_numpy_to_tensor(record, device_id):
        [X, A,
         mol_ids_rep, rep_ids_rep, iw_ids,
         last_append_mask,
         NX, NX_rep,
         action_0, actions,
         log_p,
         graph_to_rnn, rnn_to_graph, NX_cum,
         c] = record

        output = MolRNNLoader.from_numpy_to_tensor([X, A,
                                                    mol_ids_rep, rep_ids_rep, iw_ids,
                                                    last_append_mask,
                                                    NX, NX_rep,
                                                    action_0, actions,
                                                    log_p,
                                                    graph_to_rnn, rnn_to_graph, NX_cum],
                                                   device_id)
        ids = nd.array(mol_ids_rep, ctx=mx.gpu(device_id), dtype='int32')

        c = nd.array(c, ctx=mx.gpu(device_id), dtype='float32')

        output = output + [c, ids]

        return output

