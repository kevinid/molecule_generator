import tensorflow as tf
from abc import ABCMeta, abstractmethod
from os import path
import json
import numpy as np

import network
import data
import meta


def _decode_step(X, A, NX, NA, last_action, finished,
                 get_init, get_action,
                 random=False, n_node_types=len(meta.ATOM_TYPES), n_edge_types=len(meta.BOND_TYPES)):
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
            A_u = [np.zeros((0, 2), dtype=np.int32) for _ in range(meta)]
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
            A = np.stack([np.zeros(finished, dtype=np.int32),
                          np.ones(finished, dtype=np.int32),
                          bond_type], axis=1)
            NA = np.ones([finished], dtype=np.int32)
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


class Visualizer(object):

    __metaclass__ = ABCMeta

    def __init__(self, model_dir):
        self.g = tf.Graph()
        with open(path.join(model_dir, 'model.json')) as f:
            self.params = json.load(f)
        with self.g.as_default():
            self._build()
            self.sess = tf.Session()
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, path.join(model_dir, 'ckpt'))
            self.g.finalize()

    @abstractmethod
    def _build(self):
        raise NotImplementedError


class GraphLM_Visualizer(Visualizer):

    def _build(self):
        self.mdl = network.GraphLM(self.params)
        self._build_in_out()

    def _build_in_out(self):
        _, self.decode_0_output = self.mdl('decode_0')
        self.decode_step_input, self.decode_step_output = self.mdl('decode_step')
        self.loss_input, self.loss_output = self.mdl('val')

    def sample(self, num_samples):
        with self.g.as_default():
            # step one
            finished = [False,]*num_samples
            def get_init():
                init = self.sess.run(self.decode_0_output)[0]
                init = np.tile(init, [num_samples, 1])
                return init
            outputs = _decode_step(X=None, A=None, NX=None, NA=None, last_action=None, finished=finished,
                                   get_init=get_init, get_action=None,
                                   n_node_types=self.mdl.n_node_types, n_edge_types=self.mdl.n_edge_types,
                                   random=False)
            X, A, NX, NA, last_action, finished = outputs

            count = 1
            while not np.all(finished) and count < 100:
                def get_action(inputs):
                    return self.sess.run(self.decode_step_output,
                                         network.get_feed_dict(self.decode_step_input,
                                                               inputs[:-2]))

                outputs = _decode_step(X, A, NX, NA, last_action, finished,
                                       get_init=None, get_action=get_action,
                                       n_node_types=self.mdl.n_node_types, n_edge_types=self.mdl.n_edge_types,
                                       random=False)
                X, A, NX, NA, last_action, finished = outputs

                count += 1

            graph_list = []

            cumsum_X_ = np.cumsum(np.pad(NX, [[1, 0]], mode='constant')).tolist()
            cumsum_A_ = np.cumsum(np.pad(NA, [[1, 0]], mode='constant')).tolist()

            for cumsum_A_pre, cumsum_A_post, \
                cumsum_X_pre, cumsum_X_post in zip(cumsum_A_[:-1], cumsum_A_[1:],
                           cumsum_X_[:-1], cumsum_X_[1:]):
                graph_list.append([X[cumsum_X_pre:cumsum_X_post], A[cumsum_A_pre:cumsum_A_post, :]])

            return graph_list



class CGraphLM_Visualizer(Visualizer):

    def _build(self):
        self.mdl = network.CGraphLM(self.params)
        self._build_in_out()

    def _build_in_out(self):
        self.decode_0_input, self.decode_0_output = self.mdl('decode_0')
        self.decode_step_input, self.decode_step_output = self.mdl('decode_step')
        self.loss_input, self.loss_output = self.mdl('val')

    def sample(self, num_samples, c):
        if len(c.shape) < 2:
            c = c[np.newaxis, :]
            c = np.tile(c, [num_samples, 1])

        with self.g.as_default():
            # step one
            finished = [False,]*num_samples
            def get_init():
                init = self.sess.run(self.decode_0_output,
                                     {self.decode_0_input[0]:c})[0]
                return init
            outputs = _decode_step(X=None, A=None, NX=None, NA=None, last_action=None, finished=finished,
                                   get_init=get_init, get_action=None,
                                   n_node_types=self.mdl.n_node_types, n_edge_types=self.mdl.n_edge_types,
                                   random=False)
            X, A, NX, NA, last_action, finished = outputs

            count = 1
            while not np.all(finished) and count < 100:
                def get_action(inputs):
                    c_u = c[np.logical_not(finished), :]
                    return self.sess.run(self.decode_step_output,
                                         network.get_feed_dict(self.decode_step_input,
                                                               inputs + [c_u, ]))

                outputs = _decode_step(X, A, NX, NA, last_action, finished,
                                       get_init=None, get_action=get_action,
                                       n_node_types=self.mdl.n_node_types, n_edge_types=self.mdl.n_edge_types,
                                       random=False)
                X, A, NX, NA, last_action, finished = outputs

                count += 1

            graph_list = []

            cumsum_X_ = np.cumsum(np.pad(NX, [[1, 0]], mode='constant')).tolist()
            cumsum_A_ = np.cumsum(np.pad(NA, [[1, 0]], mode='constant')).tolist()

            for cumsum_A_pre, cumsum_A_post, \
                cumsum_X_pre, cumsum_X_post in zip(cumsum_A_[:-1], cumsum_A_[1:],
                           cumsum_X_[:-1], cumsum_X_[1:]):
                graph_list.append([X[cumsum_X_pre:cumsum_X_post], A[cumsum_A_pre:cumsum_A_post, :]])

            return graph_list