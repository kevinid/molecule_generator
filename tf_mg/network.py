import tensorflow as tf
from abc import ABCMeta, abstractmethod
import numpy as np

import modules
import ops


# describe single input
class InputDesc(object):

    def __init__(self, name, dtype, shape):
        self.name = name
        self.dtype = dtype
        self.shape = shape

    def get_placeholder(self):
        return tf.placeholder(self.dtype, self.shape, self.name)


# describe multiple inputs
class InputDescList(object):

    def __init__(self, input_desc_list):
        self.input_desc_list = input_desc_list

    def check_inputs(self, inputs):
        def _check(input_desc, input_item):
            if isinstance(input_desc, list):
                if not isinstance(input_item, list):
                    return False, 'Expect list'
                else:
                    if len(input_desc) != len(input_item):
                        return False, 'Number of input do not match'
                    else:
                        for _input_desc, _input_item in zip(input_desc, input_item):
                            is_match, msg = _check(_input_desc, _input_item)
                            if not is_match:
                                return is_match, msg
                            else:
                                continue
                    return True, None
            else:
                if not isinstance(input_item, np.ndarray):
                    return False, 'Item should be ndarray'
                else:
                    # check shape
                    if len(input_desc.shape) != len(input_item.shape):
                        return False, 'Dimension do not match for {} and {}'.format(input_desc.shape, input_item.shape)
                    else:
                        for _input_desc_shape, _input_item_shape in zip(input_desc.shape, input_item.shape):
                            if _input_desc_shape is not None and _input_desc_shape != _input_item_shape:
                                return False, 'Shape do not match for {} and {}'.format(input_desc.shape, input_item.shape)
                            else:
                                continue
                        return True, None
        return _check(self.input_desc_list, inputs)

    def get_placeholder(self):
        def _get_placeholder(input_desc):
            if isinstance(input_desc, list):
                placeholders = []
                for _input_desc in input_desc:
                    placeholders.append(_get_placeholder(_input_desc))
                return placeholders
            else:
                return input_desc.get_placeholder()
        return _get_placeholder(self.input_desc_list)

    def get_queue(self, queue_size=1, num_dequeue_ops=1, shared_name='queue'):
        inputs = self.get_placeholder()
        inputs_flat = flatten(inputs)
        inputs_dtype = [_input.dtype for _input in inputs_flat]
        inputs_shape = [[shape_i if shape_i is not None else -1
                         for shape_i in _input.get_shape().as_list()]
                        for _input in inputs_flat]
        queue = tf.FIFOQueue(queue_size, dtypes=inputs_dtype, shared_name=shared_name)
        enqueue = queue.enqueue(inputs_flat)

        dequeue_ops = []
        _shaper = lambda _s: [-1 if _s_i is None else _s_i for _s_i in _s]
        for _ in range(num_dequeue_ops):
            dequeue = queue.dequeue()
            dequeue = [tf.reshape(_item, _shaper(_shape)) for _item, _shape in zip(dequeue, inputs_shape)]
            dequeue = condense(inputs, dequeue)
            dequeue_ops.append(dequeue)

        return inputs, enqueue, dequeue_ops


def get_feed_dict(placeholders, inputs):
    def _get_feed_dict(input_desc, input_item, feed_dict=None):
        if feed_dict is None:
            feed_dict = {}
        if isinstance(input_desc, list):
            for _input_desc, _input_item in zip(input_desc, input_item):
                _get_feed_dict(_input_desc, _input_item, feed_dict)
        else:
            feed_dict[input_desc] = input_item
        return feed_dict
    return _get_feed_dict(placeholders, inputs)


def flatten(inputs):
    def _flatten(_input):
        outputs = []
        if isinstance(_input, list):
            for __input in _input:
                outputs.extend(_flatten(__input))
        else:
            outputs.append(_input)
        return outputs
    return _flatten(inputs)


def condense(inputs, flat_inputs):
    def _condense(_inputs, _flat_inputs):
        _condense_inputs = []
        while len(_inputs) > 0:
            if isinstance(_inputs[0], list):
                _new_condense_inputs, _flat_inputs = _condense(_inputs[0], _flat_inputs)
            else:
                _new_condense_inputs = _flat_inputs[0]
                _flat_inputs = _flat_inputs[1:]
            _condense_inputs.append(_new_condense_inputs)
            _inputs = _inputs[1:]
        return _condense_inputs, _flat_inputs
    result, _ = _condense(inputs, flat_inputs)
    return result


class Network(modules.Module):

    __metaclass__ = ABCMeta

    def __init__(self, params, name='network'):
        self._parse_params(params)
        super(Network, self).__init__(name)

    @abstractmethod
    def _parse_params(self, params):
        raise NotImplementedError

    @staticmethod
    def get_modes():
        return ['train', 'val', 'summarize']

    @abstractmethod
    def get_input_desc(self, mode):
        raise NotImplementedError

    def _call(self, mode, inputs=None):
        assert mode in self.__class__.get_modes()
        if inputs is None:
            input_desc = self.get_input_desc(mode)
            if input_desc is not None:
                inputs = self.get_input_desc(mode).get_placeholder()
            else:
                inputs = None
            outputs = self._build_subnet(mode, inputs)
            return inputs, outputs
        else:
            return self._build_subnet(mode, inputs)

    @abstractmethod
    def _train(self, inputs, name='train'):
        raise NotImplementedError

    @abstractmethod
    def _val(self, inputs, name='val'):
        raise NotImplementedError

    @abstractmethod
    def _summarize(self, inputs):
        raise NotImplementedError

    def _build_subnet(self, mode, inputs):
        if mode == 'train':
            self.training(True)
            return self._train(inputs)
        elif mode == 'val':
            self.training(False)
            return self._val(inputs)
        elif mode == 'summarize':
            self.training(False)
            return self._summarize(inputs)
        else:
            raise ValueError


class GraphLM(Network):
    @staticmethod
    def get_modes():
        return ['train', 'val', 'summarize',
                'decode_0', 'decode_step']

    def get_input_desc(self, mode):
        if mode == 'train' or mode == 'val' or mode == 'summarize':
            return InputDescList(
                [InputDesc('X', tf.int32, [None]), [InputDesc('A_{}'.format(i), tf.int32, [None, 2])
                                                    for i in range(self.n_edge_types + self.n_d)],
                 InputDesc('mol_ids_rep', tf.int32, [None]), InputDesc('rep_ids_rep', tf.int32, [None]),
                 InputDesc('iw_ids', tf.int32, [None]),
                 InputDesc('last_append_mask', tf.int32, [None]),
                 InputDesc('NX', tf.int32, [None]), InputDesc('NX_rep', tf.int32, [None]),
                 InputDesc('action_0', tf.int32, [None]), InputDesc('actions', tf.int32, [None, 5]),
                 InputDesc('log_p', tf.float32, [None])])
        elif mode == 'decode_0':
            return None
        elif mode == 'decode_step':
            return InputDescList([InputDesc('X', tf.int32, [None]), [InputDesc('A_{}'.format(i), tf.int32, [None, 2])
                                                                     for i in range(self.n_edge_types + self.n_d)],
                                  InputDesc('NX', tf.int32, [None]), InputDesc('NX_rep', tf.int32, [None]),
                                  InputDesc('last_append_mask', tf.int32, [None])])
        else:
            raise Exception('Unrecognized mode')

    def _parse_params(self, params):
        # meta
        self.n_node_types = params['n_node_types']
        self.n_edge_types = params['n_edge_types']
        self.n_d = params['n_d']

        # architecture
        self.Fe = params['Fe']
        self.Fh = params['Fh']
        self.Fskip = params['Fskip']
        self.Fh_policy = params['Fh_policy']

        # optimization
        self.starter_learning_rate = params['learning_rate']
        self.decay_step = params['decay_step']
        self.decay_rate = params['decay_rate']
        self.beta1 = params['beta1']
        self.beta2 = params['beta2']
        self.epsilon = params['epsilon']
        self.clip_val = params['clip_val']

        # activations and batch_norm
        self.activation = params['activation']
        self.batch_norm = params['batch_norm']

    def _build(self):

        # embedding
        self.embedding_node = modules.Embedding(Fin=self.n_node_types,
                                                Fout=self.Fe, name='embedding_nodes')
        self.embedding_mask = modules.Embedding(Fin=3, Fout=self.Fe,
                                                name='embedding_mask')

        # decoder modules
        self.decoder_conv = []
        for i, (f_in, f_out) in enumerate(zip([self.Fe] + list(self.Fh[: -1]), self.Fh)):
            self.decoder_conv.append(modules.MolConv(f_in, f_out, D=self.n_edge_types + self.n_d,
                                                     activation=self.activation, batch_norm=self.batch_norm,
                                                     init_bias=0.0, name='decoder_conv_{}'.format(i)))
        self.decoder_skip = modules.Linear(Fin=sum(self.Fh), Fout=self.Fskip,
                                           activation=self.activation,
                                           batch_norm=self.batch_norm,
                                           name='decoder_skip')

        # policy
        self.policy_0 = modules.create_variable('policy_0', [1, self.n_node_types])
        self.policy_h = modules.Linear(Fin=self.Fskip, Fout=self.Fh_policy,
                                       activation=None, batch_norm=False,
                                       name='policy_h')
        # end
        self.end = modules.Linear(Fin=self.Fh_policy, Fout=1, activation='exp', batch_norm=False, name='action')
        # append: where to append && what to append
        self.append = modules.Linear(Fin=self.Fh_policy, Fout=(self.n_node_types * self.n_edge_types),
                                     activation='exp', batch_norm=False, name='append')
        # connect: connect to where, how to connect
        self.connect = modules.Linear(Fin=self.Fh_policy, Fout=self.n_edge_types,
                                      activation='exp', batch_norm=False, name='connect')

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step,
                                                        self.decay_step, self.decay_rate)

    def _decode_policy_0(self, name='decode_policy_0'):
        with tf.name_scope(name):
            init = tf.exp(self.policy_0)/tf.reduce_sum(tf.exp(self.policy_0))
        return init

    def _decode_conv(self, inputs, name='decode_conv'):
        X, A, last_append_mask = inputs
        with tf.name_scope(name):
            # process A
            _to_sparse = lambda _i: tf.SparseTensor(indices=tf.cast(_i, dtype=tf.int64),
                                                    dense_shape=tf.stack([ops.get_len(X, out_type=tf.int64),
                                                                          ops.get_len(X, out_type=tf.int64)], axis=0),
                                                    values=tf.ones_like(_i[:, 0], dtype=tf.float32))
            _symmetric = lambda _s: tf.sparse_add(_s, tf.sparse_transpose(_s))
            A_sparse = [_symmetric(_to_sparse(_A_i)) for _A_i in A]

            # embedding
            X = self.embedding_node(X) + self.embedding_mask(last_append_mask)

            # convolution
            X_out = [X]
            for conv in self.decoder_conv:
                X_out_i = conv(X_out[-1], A_sparse)
                X_out.append(X_out_i)
            X_out = tf.concat(X_out[1:], axis=1)
            X_out = self.decoder_skip(X_out)
        return X_out

    def _decode_policy(self, inputs, name='decode_policy'):
        X, NX, NX_rep = inputs
        with tf.name_scope(name):
            Xh = self.policy_h(X)

            X_end = ops.segment_mean_gpu(Xh, NX_rep, NX)
            X_end = self.end(X_end)

            X_append = self.append(ops.ACTIVATIONS[self.activation](Xh))

            X_connect = self.connect(ops.ACTIVATIONS[self.activation](Xh))

            X_sum = X_end + \
                    ops.segment_sum_gpu(tf.reduce_sum(X_append, axis=1, keep_dims=True), NX_rep) + \
                    ops.segment_sum_gpu(tf.reduce_sum(X_connect, axis=1, keep_dims=True), NX_rep)

            X_sum_gathered = ops.gather_gpu(X_sum, NX_rep)
            append, connect, end = X_append / X_sum_gathered, X_connect / X_sum_gathered, X_end / X_sum

            append = tf.reshape(append, [-1, self.n_node_types, self.n_edge_types])

        return append, connect, end

    def _loss(self, inputs, name='loss'):
        init, append, connect, end, \
        mol_ids_rep, rep_ids_rep, \
        action_0, actions, \
        iw_ids, \
        log_p_sigma = inputs
        with tf.name_scope(name):
            batch_size, iw_size = tf.reduce_max(mol_ids_rep) + 1, tf.reduce_max(rep_ids_rep) + 1

            # decompose action:
            action_type, node_type, edge_type, append_pos, connect_pos = \
                actions[:, 0], actions[:, 1], actions[:, 2], actions[:, 3], actions[:, 4]
            _log_mask = lambda _x, _mask: tf.squeeze(tf.where(_mask,
                                                              tf.log(_x + 1e-10),
                                                              tf.zeros_like(_x, dtype=tf.float32)))

            # init
            init = tf.reshape(init, tf.stack([batch_size * iw_size, self.n_node_types]))
            action_0 = tf.stack([tf.range(batch_size * iw_size, dtype=tf.int32), action_0], axis=1)
            loss_init = tf.squeeze(tf.log(tf.gather_nd(init, action_0) + 1e-10))

            # end
            loss_end = _log_mask(end, tf.equal(action_type, 2))

            # append
            append_indices = tf.stack([append_pos, node_type, edge_type], axis=1)
            append = tf.gather_nd(append, append_indices)
            loss_append = _log_mask(append, tf.equal(action_type, 0))

            # connect
            connect_indices = tf.stack([connect_pos, edge_type], axis=1)
            connect = tf.gather_nd(connect, connect_indices)
            loss_connect = _log_mask(connect, tf.equal(action_type, 1))

            # sum up results
            log_p_x = loss_end + loss_append + loss_connect
            log_p_x = tf.squeeze(ops.segment_sum_gpu(log_p_x[:, tf.newaxis], iw_ids))
            log_p_x += loss_init

            # reshape
            log_p_x = tf.reshape(log_p_x, tf.stack([batch_size, iw_size]))
            log_p_sigma = tf.reshape(log_p_sigma, tf.stack([batch_size, iw_size]))
            loss_rec = log_p_x - log_p_sigma
            loss_rec = tf.reduce_logsumexp(loss_rec, axis=1) - tf.log(tf.cast(iw_size, dtype=tf.float32))
            total_loss = -tf.reduce_mean(loss_rec)

        return total_loss


    def _train(self, inputs, name='train'):
        with tf.name_scope(name):
            total_loss, = self._val(inputs)

            optimizer = tf.train.AdamOptimizer(self.learning_rate, self.beta1, self.beta2, self.epsilon)

            grad = optimizer.compute_gradients(total_loss)

            grad = [(tf.clip_by_value(g, -self.clip_val, self.clip_val), v) for g, v in grad]
            step_op = optimizer.apply_gradients(grad, global_step=self.global_step)
        return step_op, self.global_step, total_loss

    def _val(self, inputs, name='val', loss_iw=False):
        X, A, \
        mol_ids_rep, rep_ids_rep, iw_ids, \
        last_append_mask, \
        NX, NX_rep, \
        action_0, actions, \
        log_p = inputs
        with tf.name_scope(name):
            batch_size, iw_size = tf.reduce_max(mol_ids_rep) + 1, tf.reduce_max(rep_ids_rep) + 1

            init = self._decode_policy_0()
            init = tf.tile(init[tf.newaxis, :, :], tf.stack([batch_size, iw_size, 1]))

            inputs = [X, A, last_append_mask]
            X_out = self._decode_conv(inputs)
            inputs = [X_out, NX, NX_rep]
            append, connect, end = self._decode_policy(inputs)

            # get loss
            inputs = [init, append, connect, end,
                      mol_ids_rep, rep_ids_rep,
                      action_0, actions,
                      iw_ids,
                      log_p]
            total_loss = self._loss(inputs)
        return total_loss,

    def _summarize(self, inputs):
        total_loss, = self._val(inputs)
        with tf.name_scope('Loss'):
            tf.summary.scalar('Total', total_loss)
        with tf.name_scope('Variables'):
            tf.summary.scalar('learning_rate', self.learning_rate)
        summary_op = tf.summary.merge_all()
        return summary_op, self.global_step

    def _build_subnet(self, mode, inputs):
        if mode == 'decode_0':
            self.training(False)
            init = self._decode_policy_0()
            return init,
        elif mode == 'decode_step':
            self.training(False)
            X, A, NX, NX_rep, last_append_mask = inputs
            inputs = [X, A, last_append_mask]
            X = self._decode_conv(inputs)
            inputs = [X, NX, NX_rep]
            return self._decode_policy(inputs)
        else:
            return super(GraphLM, self)._build_subnet(mode, inputs)


class CGraphLM(GraphLM):

    def get_input_desc(self, mode):
        if mode == 'decode_step':
            return InputDescList([InputDesc('X', tf.int32, [None]), [InputDesc('A_{}'.format(i), tf.int32, [None, 2])
                                                                     for i in range(self.n_edge_types + self.n_d)],
                                  InputDesc('NX', tf.int32, [None]), InputDesc('NX_rep', tf.int32, [None]),
                                  InputDesc('last_append_mask', tf.int32, [None]),
                                  InputDesc('mol_ids_rep', tf.int32, [None]),
                                  InputDesc('rep_ids_rep', tf.int32, [None]),
                                  InputDesc('c', tf.float32, [None, self.Fc])])
        else:
            input_desc_list = super(CGraphLM, self).get_input_desc(mode)
            input_desc_list.input_desc_list.append(InputDesc('c', tf.float32, [None, self.Fc]))
            return input_desc_list

    def _parse_params(self, params):
        super(CGraphLM, self)._parse_params(params)

        self.Fc = params['Fc']
        self.Fch = params['Fch']

    def _build(self):

        # embedding
        self.embedding_node = modules.Embedding(Fin=self.n_node_types,
                                                Fout=self.Fe, name='embedding_nodes')
        self.embedding_mask = modules.Embedding(Fin=3, Fout=self.Fe,
                                                name='embedding_mask')

        # conditional code
        self.linear_c = modules.Linear(Fin=self.Fc, Fout=self.Fch, activation=self.activation,
                                       batch_norm=self.batch_norm, name='linear_c')

        # decoder modules
        self.decoder_conv = []
        for i, (f_in, f_out) in enumerate(zip([self.Fe] + list(self.Fh[: -1]), self.Fh)):
            self.decoder_conv.append(modules.MolConv(f_in, f_out, D=self.n_edge_types + self.n_d, Fc=self.Fch,
                                                     activation=self.activation, batch_norm=self.batch_norm,
                                                     init_bias=0.0, name='decoder_conv_{}'.format(i)))

        self.decoder_skip = modules.Linear(Fin=sum(self.Fh), Fout=self.Fskip,
                                           activation=self.activation,
                                           batch_norm=self.batch_norm,
                                           name='decoder_skip')

        # policy
        self.policy_0 = modules.Linear(self.Fch, self.n_node_types, activation='softmax',
                                       batch_norm=False, name='policy_0')

        self.policy_h = modules.Linear(Fin=self.Fskip, Fout=self.Fh_policy,
                                       activation=None, batch_norm=False,
                                       name='policy_h')
        # end
        self.end = modules.Linear(Fin=self.Fh_policy, Fout=1, activation='exp', batch_norm=False, name='action')
        # append: where to append && what to append
        self.append = modules.Linear(Fin=self.Fh_policy, Fout=(self.n_node_types * self.n_edge_types),
                                     activation='exp', batch_norm=False, name='append')
        # connect: connect to where, how to connect
        self.connect = modules.Linear(Fin=self.Fh_policy, Fout=self.n_edge_types,
                                      activation='exp', batch_norm=False, name='connect')

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step,
                                                        self.decay_step, self.decay_rate)

    def _mlp_c(self, inputs, name='mlp_c'):
        c, = inputs
        with tf.name_scope(name):
            hc = self.linear_c(c)
        return hc

    def _decode_policy_0(self, inputs, name='policy_0'):
        hc, = inputs
        with tf.name_scope(name):
            init = self.policy_0(hc)
        return init

    def _decode_conv(self, inputs, name='decode_conv'):
        X, A, last_append_mask, mol_ids_rep, rep_ids_rep, hc = inputs
        with tf.name_scope(name):
            # process A
            _to_sparse = lambda _i: tf.SparseTensor(indices=tf.cast(_i, dtype=tf.int64),
                                                    dense_shape=tf.stack([ops.get_len(X, out_type=tf.int64),
                                                                          ops.get_len(X, out_type=tf.int64)], axis=0),
                                                    values=tf.ones_like(_i[:, 0], dtype=tf.float32))
            _symmetric = lambda _s: tf.sparse_add(_s, tf.sparse_transpose(_s))
            A_sparse = [_symmetric(_to_sparse(_A_i)) for _A_i in A]

            # embedding
            X = self.embedding_node(X) + self.embedding_mask(last_append_mask)

            # preparing c
            batch_size, iw_size = tf.reduce_max(mol_ids_rep) + 1, tf.reduce_max(rep_ids_rep) + 1
            hc_expand = tf.tile(hc[:, tf.newaxis, :], tf.stack([1, iw_size, 1]))
            indices = tf.stack([mol_ids_rep, rep_ids_rep], axis=1)

            # convolution
            X_out = [X]

            for conv in self.decoder_conv:
                X_out_i = conv(X_out[-1], A_sparse, hc_expand, indices)
                X_out.append(X_out_i)
            X_out = tf.concat(X_out[1:], axis=1)
            X_out = self.decoder_skip(X_out)
        return X_out


    def _train(self, inputs, name='train'):
        with tf.name_scope(name):
            total_loss, = self._val(inputs)

            optimizer = tf.train.AdamOptimizer(self.learning_rate, self.beta1, self.beta2, self.epsilon)

            grad = optimizer.compute_gradients(total_loss)

            grad = [(tf.clip_by_value(g, -self.clip_val, self.clip_val), v) for g, v in grad]
            step_op = optimizer.apply_gradients(grad, global_step=self.global_step)
        return step_op, self.global_step, total_loss

    def _val(self, inputs, name='val', loss_iw=False):
        X, A, \
        mol_ids_rep, rep_ids_rep, iw_ids, \
        last_append_mask, \
        NX, NX_rep, \
        action_0, actions, \
        log_p, c = inputs
        with tf.name_scope(name):
            batch_size, iw_size = tf.reduce_max(mol_ids_rep) + 1, tf.reduce_max(rep_ids_rep) + 1

            inputs = [c,]
            hc = self._mlp_c(inputs)

            inputs = [hc,]
            init = self._decode_policy_0(inputs)
            init = tf.tile(init[:, tf.newaxis, :], tf.stack([1, iw_size, 1]))

            inputs = [X, A, last_append_mask, mol_ids_rep, rep_ids_rep, hc]
            X_out = self._decode_conv(inputs)

            inputs = [X_out, NX, NX_rep]
            append, connect, end = self._decode_policy(inputs)

            # get loss
            inputs = [init, append, connect, end,
                      mol_ids_rep, rep_ids_rep,
                      action_0, actions,
                      iw_ids,
                      log_p]
            total_loss = self._loss(inputs)
        return total_loss,

    def _summarize(self, inputs):
        total_loss, = self._val(inputs)
        with tf.name_scope('Loss'):
            tf.summary.scalar('Total', total_loss)
        with tf.name_scope('Variables'):
            tf.summary.scalar('learning_rate', self.learning_rate)
        summary_op = tf.summary.merge_all()
        return summary_op, self.global_step

    def _build_subnet(self, mode, inputs):
        if mode == 'decode_0':
            self.training(False)
            c, = inputs
            init = self._decode_policy_0(c)
            return init,
        elif mode == 'decode_step':
            self.training(False)
            X, A, NX, NX_rep, last_append_mask, mol_ids_rep, c = inputs
            rep_ids_rep = tf.zeros_like(mol_ids_rep)

            inputs = [c, ]
            hc = self._mlp_c(inputs)

            inputs = [X, A, last_append_mask, mol_ids_rep, rep_ids_rep, hc]
            X = self._decode_conv(inputs)

            inputs = [X, NX, NX_rep]
            return self._decode_policy(inputs)
        else:
            return super(CGraphLM, self)._build_subnet(mode, inputs)


NETWORKS = {'graph_lm':GraphLM,
            'c_graph_lm':CGraphLM}
