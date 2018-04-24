import numpy as np
import json
import os
import mxnet as mx
from mxnet import autograd, nd
from abc import ABCMeta, abstractmethod
from rdkit import Chem

from mx_mg.data import get_mol_spec
from mx_mg import models, data


def _decode_step(X, A, NX, NA, last_action, finished,
                 get_init, get_action,
                 random=True, n_node_types=get_mol_spec().num_atom_types,
                 n_edge_types=get_mol_spec().num_bond_types):
    if X is None:
        init = get_init()

        if random:
            X = []
            for i in range(init.shape[0]):
                p = init[i, :]
                selected_atom = np.random.choice(np.arange(init.shape[1]), 1, p=p)[0]
                X.append(selected_atom)
            X = np.array(X, dtype=np.int32)
        else:
            X = np.argmax(init, axis=1)
        A = np.zeros((0, 3), dtype=np.int32)
        NX = last_action = np.ones([X.shape[0]], dtype=np.int32)
        NA = np.zeros([X.shape[0]], dtype=np.int32)
        finished = np.array([False, ] * X.shape[0], dtype=np.bool)

        return X, A, NX, NA, last_action, finished
    else:
        X_u = X[np.repeat(np.logical_not(finished), NX)]
        A_u = A[np.repeat(np.logical_not(finished), NA), :]
        NX_u = NX[np.logical_not(finished)]
        NA_u = NA[np.logical_not(finished)]
        last_action_u = last_action[np.logical_not(finished)]

        # conv
        mol_ids_rep = NX_rep = np.repeat(np.arange(NX_u.shape[0]), NX_u)
        rep_ids_rep = np.zeros_like(mol_ids_rep)

        if A.shape[0] == 0:
            D_2 = D_3 = np.zeros((0, 2), dtype=np.int32)
            A_u = [np.zeros((0, 2), dtype=np.int32) for _ in range(get_mol_spec().num_bond_types)]
            A_u += [D_2, D_3]
        else:
            cumsum = np.cumsum(np.pad(NX_u, [[1, 0]], mode='constant')[:-1])
            shift = np.repeat(cumsum, NA_u)
            A_u[:, :2] += np.stack([shift, ] * 2, axis=1)
            D_2, D_3 = data.get_d(A_u, X_u)
            A_u = [A_u[A_u[:, 2] == _i, :2] for _i in range(n_edge_types)]
            A_u += [D_2, D_3]

        mask = np.zeros([X_u.shape[0]], dtype=np.int32)
        last_append_index = np.cumsum(NX_u) - 1
        mask[last_append_index] = np.where(last_action_u == 1,
                                           np.ones_like(last_append_index, dtype=np.int32),
                                           np.ones_like(last_append_index, dtype=np.int32) * 2)

        decode_input = [X_u, A_u, NX_u, NX_rep, mask, mol_ids_rep, rep_ids_rep]
        append, connect, end = get_action(decode_input)

        if A.shape[0] == 0:
            max_index = np.argmax(np.reshape(append, [-1, n_node_types * n_edge_types]), axis=1)
            atom_type, bond_type = np.unravel_index(max_index, [n_node_types, n_edge_types])
            X = np.reshape(np.stack([X, atom_type], axis=1), [-1])
            NX = np.array([2, ] * len(finished), dtype=np.int32)
            A = np.stack([np.zeros([len(finished), ], dtype=np.int32),
                          np.ones([len(finished), ], dtype=np.int32),
                          bond_type], axis=1)
            NA = np.ones([len(finished), ], dtype=np.int32)
            last_action = np.ones_like(NX, dtype=np.int32)

        else:
            # process for each molecule
            append, connect = np.split(append, np.cumsum(NX_u)), np.split(connect, np.cumsum(NX_u))
            end = end.tolist()

            unfinished_ids = np.where(np.logical_not(finished))[0].tolist()
            cumsum = np.cumsum(NX)
            cumsum_a = np.cumsum(NA)

            X_insert = []
            X_insert_ids = []
            A_insert = []
            A_insert_ids = []
            finished_ids = []

            for i, (unfinished_id, append_i, connect_i, end_i) \
                    in enumerate(zip(unfinished_ids, append, connect, end)):
                if random:
                    def _rand_id(*_x):
                        _x_reshaped = [np.reshape(_xi, [-1]) for _xi in _x]
                        _x_length = np.array([_x_reshape_i.shape[0] for _x_reshape_i in _x_reshaped],
                                             dtype=np.int32)
                        _begin = np.cumsum(np.pad(_x_length, [[1, 0]], mode='constant')[:-1])
                        _end = np.cumsum(_x_length) - 1
                        _p = np.concatenate(_x_reshaped)
                        _p = _p / np.sum(_p)
                        _rand_index = np.random.choice(np.arange(_p.shape[0]), 1, p=_p)[0]
                        _p_step = _p[_rand_index]
                        _x_index = np.where(np.logical_and(_begin <= _rand_index, _end >= _rand_index))[0][0]
                        _rand_index = _rand_index - _begin[_x_index]
                        _rand_index = np.unravel_index(_rand_index, _x[_x_index].shape)
                        return _x_index, _rand_index, _p_step

                    action_type, action_index, p_step = _rand_id(append_i, connect_i, np.array([end_i]))
                else:
                    _argmax = lambda _x: np.unravel_index(np.argmax(_x), _x.shape)
                    append_id, append_val = _argmax(append_i), np.max(append_i)
                    connect_id, connect_val = _argmax(connect_i), np.max(connect_i)
                    end_val = end_i
                    if end_val >= append_val and end_val >= connect_val:
                        action_type = 2
                        action_index = None
                    elif append_val >= connect_val and append_val >= end_val:
                        action_type = 0
                        action_index = append_id
                    else:
                        action_type = 1
                        action_index = connect_id
                if action_type == 2:
                    # finish growth
                    finished_ids.append(unfinished_id)
                elif action_type == 0:
                    # append action
                    append_pos, atom_type, bond_type = action_index
                    X_insert.append(atom_type)
                    X_insert_ids.append(unfinished_id)
                    A_insert.append([append_pos, NX[unfinished_id], bond_type])
                    A_insert_ids.append(unfinished_id)
                else:
                    # connect
                    connect_ps, bond_type = action_index
                    A_insert.append([NX[unfinished_id] - 1, connect_ps, bond_type])
                    A_insert_ids.append(unfinished_id)
            if len(A_insert_ids) > 0:
                A = np.insert(A, cumsum_a[A_insert_ids], A_insert, axis=0)
                NA[A_insert_ids] += 1
                last_action[A_insert_ids] = 0
            if len(X_insert_ids) > 0:
                X = np.insert(X, cumsum[X_insert_ids], X_insert, axis=0)
                NX[X_insert_ids] += 1
                last_action[X_insert_ids] = 1
            if len(finished_ids) > 0:
                finished[finished_ids] = True
            # print finished

        return X, A, NX, NA, last_action, finished

class Builder(object):

    __metaclass__ = ABCMeta

    def __init__(self, model_loc, gpu_id=0):
        with open(os.path.join(model_loc, 'configs.json')) as f:
            configs = json.load(f)

        self.mdl = self.__class__._get_model(configs)

        self.ctx = mx.gpu(gpu_id) if gpu_id is not None else mx.cpu()
        self.mdl.load_params(os.path.join(model_loc, 'ckpt.params'), ctx=self.ctx)

    @staticmethod
    def _get_model(configs):
        raise NotImplementedError

    @abstractmethod
    def sample(self, num_samples, *args, **kwargs):
        raise NotImplementedError


class Vanilla_Builder(Builder):

    @staticmethod
    def _get_model(configs):
        return models.VanillaMolGen(get_mol_spec().num_atom_types, get_mol_spec().num_bond_types, D=2, **configs)

    def sample(self, num_samples, output_type='mol', sanitize=True, random=True):
        with autograd.predict_mode():
            # step one
            finished = [False, ] * num_samples

            def get_init():
                self.mdl.mode = 'decode_0'
                init = self.mdl(self.ctx).asnumpy()
                init = np.stack([init, ] * num_samples, axis=0)
                return init

            outputs = _decode_step(X=None, A=None, NX=None, NA=None, last_action=None, finished=finished,
                                   get_init=get_init, get_action=None,
                                   n_node_types=self.mdl.N_A, n_edge_types=self.mdl.N_B,
                                   random=random)
            X, A, NX, NA, last_action, finished = outputs

            count = 1
            while not np.all(finished) and count < 100:
                def get_action(inputs):
                    self.mdl.mode = 'decode_step'

                    _append, _connect, _end = self.mdl(*self.to_nd(inputs))
                    return _append.asnumpy(), _connect.asnumpy(), _end.asnumpy()

                outputs = _decode_step(X, A, NX, NA, last_action, finished,
                                       get_init=None, get_action=get_action,
                                       n_node_types=self.mdl.N_A, n_edge_types=self.mdl.N_B,
                                       random=random)
                X, A, NX, NA, last_action, finished = outputs

                count += 1

            graph_list = []

            cumsum_X_ = np.cumsum(np.pad(NX, [[1, 0]], mode='constant')).tolist()
            cumsum_A_ = np.cumsum(np.pad(NA, [[1, 0]], mode='constant')).tolist()

            for cumsum_A_pre, cumsum_A_post, \
                cumsum_X_pre, cumsum_X_post in zip(cumsum_A_[:-1], cumsum_A_[1:],
                                                   cumsum_X_[:-1], cumsum_X_[1:]):
                graph_list.append([X[cumsum_X_pre:cumsum_X_post], A[cumsum_A_pre:cumsum_A_post, :]])

            if output_type=='graph':
                return graph_list
            elif output_type == 'mol':
                return data.get_mol_from_graph_list(graph_list, sanitize)
            elif output_type == 'smiles':
                mol_list = data.get_mol_from_graph_list(graph_list, sanitize=True)
                smiles_list = [Chem.MolToSmiles(m) if m is not None else None for m in mol_list]
                return smiles_list
            else:
                raise ValueError('Unrecognized output type')

    def to_nd(self, inputs):
        X, A, NX, NX_rep, mask = inputs[:-2]

        # convert to ndarray
        _to_ndarray = lambda _x: nd.array(_x, self.ctx, 'int32')
        X, NX, NX_rep, mask = \
            _to_ndarray(X), _to_ndarray(NX), _to_ndarray(NX_rep), _to_ndarray(mask)
        A_sparse = []
        for _A_i in A:
            if _A_i.shape[0] == 0:
                A_sparse.append(None)
            else:
                # transpose may not be supported in gpu
                _A_i = np.concatenate([_A_i, _A_i[:, [1, 0]]], axis=0)

                # construct csr matrix ...
                _data = np.ones((_A_i.shape[0],), dtype=np.float32)
                _row, _col = _A_i[:, 0], _A_i[:, 1]
                _A_sparse_i = nd.sparse.csr_matrix((_data, (_row, _col)),
                                                   shape=tuple([int(X.shape[0]), ] * 2),
                                                   ctx=self.ctx, dtype='float32')

                # append to list
                A_sparse.append(_A_sparse_i)
        return X, A_sparse, NX, NX_rep, mask


class Vanilla_RNN_Builder(Builder):

    @staticmethod
    def _get_model(configs):
        return models.VanillaMolGen_RNN(get_mol_spec().num_atom_types, get_mol_spec().num_bond_types, D=2, **configs)


    def sample(self, num_samples, output_type='mol', sanitize=True, random=True):
        with autograd.predict_mode():
            # step one
            finished = [False, ] * num_samples

            def get_init():
                self.mdl.mode = 'decode_0'
                init = self.mdl(self.ctx).asnumpy()
                init = np.stack([init, ] * num_samples, axis=0)
                return init

            outputs = _decode_step(X=None, A=None, NX=None, NA=None, last_action=None, finished=finished,
                                   get_init=get_init, get_action=None,
                                   n_node_types=self.mdl.N_A, n_edge_types=self.mdl.N_B,
                                   random=random)
            X, A, NX, NA, last_action, finished = outputs

            count = 1
            h = np.zeros([self.mdl.N_rnn, num_samples, self.mdl.F_c[-1]], dtype=np.float32)
            while not np.all(finished) and count < 100:
                def get_action(inputs):
                    self.mdl.mode = 'decode_step'
                    _h = nd.array(h[:, np.logical_not(finished), :], ctx=self.ctx, dtype='float32')
                    _X, _A_sparse, _NX, _NX_rep, _mask, _NX_cum = self.to_nd(inputs)
                    _append, _connect, _end, _h = self.mdl(_X, _A_sparse, _NX, _NX_rep, _mask, _NX_cum, _h)
                    h[:, np.logical_not(finished), :] = _h[0].asnumpy()
                    return _append.asnumpy(), _connect.asnumpy(), _end.asnumpy()

                outputs = _decode_step(X, A, NX, NA, last_action, finished,
                                       get_init=None, get_action=get_action,
                                       n_node_types=self.mdl.N_A, n_edge_types=self.mdl.N_B,
                                       random=random)
                X, A, NX, NA, last_action, finished = outputs

                count += 1

            graph_list = []

            cumsum_X_ = np.cumsum(np.pad(NX, [[1, 0]], mode='constant')).tolist()
            cumsum_A_ = np.cumsum(np.pad(NA, [[1, 0]], mode='constant')).tolist()

            for cumsum_A_pre, cumsum_A_post, \
                cumsum_X_pre, cumsum_X_post in zip(cumsum_A_[:-1], cumsum_A_[1:],
                                                   cumsum_X_[:-1], cumsum_X_[1:]):
                graph_list.append([X[cumsum_X_pre:cumsum_X_post], A[cumsum_A_pre:cumsum_A_post, :]])

            if output_type=='graph':
                return graph_list
            elif output_type == 'mol':
                return data.get_mol_from_graph_list(graph_list, sanitize)
            elif output_type == 'smiles':
                mol_list = data.get_mol_from_graph_list(graph_list, sanitize=True)
                smiles_list = [Chem.MolToSmiles(m) if m is not None else None for m in mol_list]
                return smiles_list
            else:
                raise ValueError('Unrecognized output type')

    def to_nd(self, inputs):
        X, A, NX, NX_rep, mask = inputs[:-2]
        NX_cum = np.cumsum(NX)

        # convert to ndarray
        _to_ndarray = lambda _x: nd.array(_x, self.ctx, 'int32')
        X, NX, NX_rep, mask, NX_cum = \
            _to_ndarray(X), _to_ndarray(NX), _to_ndarray(NX_rep), _to_ndarray(mask), _to_ndarray(NX_cum)
        A_sparse = []
        for _A_i in A:
            if _A_i.shape[0] == 0:
                A_sparse.append(None)
            else:
                # transpose may not be supported in gpu
                _A_i = np.concatenate([_A_i, _A_i[:, [1, 0]]], axis=0)

                # construct csr matrix ...
                _data = np.ones((_A_i.shape[0],), dtype=np.float32)
                _row, _col = _A_i[:, 0], _A_i[:, 1]
                _A_sparse_i = nd.sparse.csr_matrix((_data, (_row, _col)),
                                                   shape=tuple([int(X.shape[0]), ] * 2),
                                                   ctx=self.ctx, dtype='float32')

                # append to list
                A_sparse.append(_A_sparse_i)
        return X, A_sparse, NX, NX_rep, mask, NX_cum


class CVanilla_RNN_Builder(Builder):

    @staticmethod
    def _get_model(configs):
        return models.CVanillaMolGen_RNN(get_mol_spec().num_atom_types, get_mol_spec().num_bond_types, D=2, **configs)


    def sample(self, num_samples, c, output_type='mol', sanitize=True, random=True):
        if len(c.shape) == 1:
            c = np.stack([c, ]*num_samples, axis=0)

        with autograd.predict_mode():
            # step one
            finished = [False, ] * num_samples

            def get_init():
                self.mdl.mode = 'decode_0'
                _c = nd.array(c, dtype='float32', ctx=self.ctx)
                init = self.mdl(_c).asnumpy()
                return init

            outputs = _decode_step(X=None, A=None, NX=None, NA=None, last_action=None, finished=finished,
                                   get_init=get_init, get_action=None,
                                   n_node_types=self.mdl.N_A, n_edge_types=self.mdl.N_B,
                                   random=random)
            X, A, NX, NA, last_action, finished = outputs

            count = 1
            h = np.zeros([self.mdl.N_rnn, num_samples, self.mdl.F_c[-1]], dtype=np.float32)
            while not np.all(finished) and count < 100:
                def get_action(inputs):
                    self.mdl.mode = 'decode_step'
                    _h = nd.array(h[:, np.logical_not(finished), :], ctx=self.ctx, dtype='float32')
                    _c = nd.array(c[np.logical_not(finished), :], ctx=self.ctx, dtype='float32')
                    _X, _A_sparse, _NX, _NX_rep, _mask, _NX_cum = self.to_nd(inputs)
                    _append, _connect, _end, _h = self.mdl(_X, _A_sparse, _NX, _NX_rep, _mask, _NX_cum, _h, _c, _NX_rep)
                    h[:, np.logical_not(finished), :] = _h[0].asnumpy()
                    return _append.asnumpy(), _connect.asnumpy(), _end.asnumpy()

                outputs = _decode_step(X, A, NX, NA, last_action, finished,
                                       get_init=None, get_action=get_action,
                                       n_node_types=self.mdl.N_A, n_edge_types=self.mdl.N_B,
                                       random=random)
                X, A, NX, NA, last_action, finished = outputs

                count += 1

            graph_list = []

            cumsum_X_ = np.cumsum(np.pad(NX, [[1, 0]], mode='constant')).tolist()
            cumsum_A_ = np.cumsum(np.pad(NA, [[1, 0]], mode='constant')).tolist()

            for cumsum_A_pre, cumsum_A_post, \
                cumsum_X_pre, cumsum_X_post in zip(cumsum_A_[:-1], cumsum_A_[1:],
                                                   cumsum_X_[:-1], cumsum_X_[1:]):
                graph_list.append([X[cumsum_X_pre:cumsum_X_post], A[cumsum_A_pre:cumsum_A_post, :]])

            if output_type=='graph':
                return graph_list
            elif output_type == 'mol':
                return data.get_mol_from_graph_list(graph_list, sanitize)
            elif output_type == 'smiles':
                mol_list = data.get_mol_from_graph_list(graph_list, sanitize=True)
                smiles_list = [Chem.MolToSmiles(m) if m is not None else None for m in mol_list]
                return smiles_list
            else:
                raise ValueError('Unrecognized output type')

    def to_nd(self, inputs):
        X, A, NX, NX_rep, mask = inputs[:-2]
        NX_cum = np.cumsum(NX)

        # convert to ndarray
        _to_ndarray = lambda _x: nd.array(_x, self.ctx, 'int32')
        X, NX, NX_rep, mask, NX_cum = \
            _to_ndarray(X), _to_ndarray(NX), _to_ndarray(NX_rep), _to_ndarray(mask), _to_ndarray(NX_cum)
        A_sparse = []
        for _A_i in A:
            if _A_i.shape[0] == 0:
                A_sparse.append(None)
            else:
                # transpose may not be supported in gpu
                _A_i = np.concatenate([_A_i, _A_i[:, [1, 0]]], axis=0)

                # construct csr matrix ...
                _data = np.ones((_A_i.shape[0],), dtype=np.float32)
                _row, _col = _A_i[:, 0], _A_i[:, 1]
                _A_sparse_i = nd.sparse.csr_matrix((_data, (_row, _col)),
                                                   shape=tuple([int(X.shape[0]), ] * 2),
                                                   ctx=self.ctx, dtype='float32')

                # append to list
                A_sparse.append(_A_sparse_i)
        return X, A_sparse, NX, NX_rep, mask, NX_cum

