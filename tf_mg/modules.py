"""
Contains interface for modules, as well as some commonly used modules inside the project
"""
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
import tensorflow as tf

import ops

_module_stack = []


@contextmanager
def module_scope(current_module):
    _module_stack.append(current_module)
    yield
    _module_stack.pop()


def get_current_module():
    if len(_module_stack) == 0:
        return None
    else:
        current_module = _module_stack[-1]
        return current_module


def create_variable(name, shape):
    """Create a convolution filter variable with the specified name and shape,
    and initialize it using Xavier initialisation."""
    current_module = get_current_module()
    if current_module is None:
        raise Exception('Root level variable not allowed !')
    initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)
    variable = tf.get_variable(initializer=initializer(shape=shape), name=name)
    current_module.variables[name] = variable
    return variable


def create_bias_variable(name, shape, trainable=True, init_value=0.0):
    """Create a bias variable with the specified name and shape and initialize
    it to zero."""
    current_module = get_current_module()
    initializer = tf.constant_initializer(value=init_value, dtype=tf.float32)
    variable = tf.get_variable(initializer=initializer(shape=shape), name=name, trainable=trainable)
    if current_module is not None:
        current_module.variables[name] = variable
    return variable


def create_global_step(name='global_step'):
    current_module = get_current_module()
    if current_module is None:
        raise Exception('Root level variable not allowed !')
    global_step = tf.Variable(0, trainable=False, name=name)
    current_module.variables[name] = global_step
    return global_step


class Module(object):
    __metaclass__ = ABCMeta

    def __init__(self, name):
        self._name = name
        self._training = True

        with tf.variable_scope('{}'.format(self.name)), tf.name_scope('{}_variables'.format(self.name)):
            self._variables = {}
            self._submodules = {}

            # add self to current module's child list
            current_module = get_current_module()
            if current_module is not None:
                current_module.submodules[self.name] = self
            self._parent = current_module

            # build
            with module_scope(self):
                self._build()

    def __call__(self, *args, **kwargs):
        with tf.name_scope(self.name):
            return self._call(*args, **kwargs)

    def _build(self):
        pass

    def training(self, is_training=True):
        self._training = is_training
        for m in self._submodules:
            self.submodules[m].training(is_training)

    @property
    def l1(self):
        with tf.name_scope('{}_l1'.format(self.name)):
            l1 = 0
            for v in self._variables:
                l1 += tf.reduce_sum(tf.abs(self._variables[v]))
            for m in self._submodules:
                l1 += self.submodules[m].l1
            return l1

    @property
    def l2(self):
        with tf.name_scope('{}_l2'.format(self.name)):
            l2 = 0
            for v in self._variables:
                l2 += tf.reduce_sum(self._variables[v] ** 2)
            for m in self._submodules:
                l2 += self._submodules[m].l2
            return l2

    @property
    def variables(self):
        return self._variables

    @property
    def submodules(self):
        return self._submodules

    @property
    def name(self):
        return self._name

    @property
    def parent(self):
        return self._parent

    @abstractmethod
    def _call(self, *args, **kwargs):
        raise NotImplementedError

    def get_all_variables(self):
        variables = []
        for k in self.variables:
            variables.append(self.variables[k])

        for m in self.submodules:
            variables.extend(self.submodules[m].get_all_variables())

        return variables


class BatchNorm(Module):

    def __init__(self, F,
                 init_offset=0,
                 init_scale=1.0,
                 decay=0.9,
                 name='batch_norm'):
        self.F = F
        self.init_offset = init_offset
        self.init_scale = init_scale
        self.decay = decay
        super(BatchNorm, self).__init__(name)

    def _build(self):
        if self.init_offset is not None:
            self.w = create_bias_variable('w', [1, self.F])
            self.b = create_bias_variable('b', [1, self.F])
        else:
            self.w = create_bias_variable('w', [1, self.F])
        self.mean = create_bias_variable('mean', [1, self.F], trainable=False)
        self.var = create_bias_variable('var', [1, self.F], trainable=False, init_value=1.0)

    def _call(self, X):
        # input checking
        assert X.get_shape().as_list()[1] == self.F

        if self._training:
            mean, var = tf.nn.moments(X, axes=[0], keep_dims=True)
            train_mean = tf.assign(self.mean,
                                   self.decay * self.mean + (1 - self.decay) * mean)
            train_val = tf.assign(self.var,
                                  self.decay * self.var + (1 - self.decay) * var)
            with tf.control_dependencies([train_mean, train_val]):
                X = tf.nn.batch_normalization(X, mean, var,
                                              self.b + self.init_offset if self.init_offset is not None else 0,
                                              self.w + 1, variance_epsilon=0.01)
        else:
            X = tf.nn.batch_normalization(X, self.mean, self.var,
                                          self.b + self.init_offset if self.init_offset is not None else 0,
                                          self.w + 1, variance_epsilon=0.01)
        return X


class Linear(Module):

    def __init__(self, Fin, Fout,
                 activation='relu',
                 init_bias=0,
                 batch_norm=True,
                 name='linear'):
        self.Fin = Fin
        self.Fout = Fout
        self.init_bias = init_bias
        self.activation = ops.ACTIVATIONS[activation]
        self.use_batch_norm = batch_norm

        super(Linear, self).__init__(name)

    def _build(self):
        self.w = create_variable('w', [self.Fin, self.Fout])
        if self.use_batch_norm:
            self.bn = BatchNorm(F=self.Fout, init_offset=self.init_bias, name='bn')
        else:
            if self.init_bias is not None:
                self.b = create_bias_variable('b', [1, self.Fout])

    def _call(self, X):
        # input checking
        assert X.get_shape().as_list()[1] == self.Fin

        X_out = tf.matmul(X, self.w)
        if self.use_batch_norm:
            X_out = self.bn(X_out)
        else:
            if self.init_bias is not None:
                X_out += self.b + self.init_bias

        X_out = self.activation(X_out)

        return X_out


class Embedding(Module):

    def __init__(self, Fin, Fout, name='MolEmbedding'):
        self.Fin = Fin
        self.Fout = Fout
        super(Embedding, self).__init__(name)

    def _build(self):
        self.w = create_variable('w', [self.Fin, self.Fout])

    def _call(self, X):
        X = ops.gather_gpu(self.w, X)
        return X


class MolConv(Module):

    def __init__(self, Fin, Fout, D, Fc=None,
                 activation='relu',
                 init_bias=0,
                 batch_norm=True,
                 name='MolConv'):
        self.Fin = Fin
        self.Fout = Fout
        self.D = D
        self.Fc = Fc
        self.activation = ops.ACTIVATIONS[activation]
        self.init_bias = init_bias
        self.batch_norm = batch_norm
        super(MolConv, self).__init__(name)

    def _build(self):
        self.w = create_variable('w', [self.Fin * (self.D + 1), self.Fout])
        if self.batch_norm:
            self.bn = BatchNorm(F=self.Fout, init_offset=self.init_bias, name='bn')
        else:
            if self.init_bias is not None:
                self.b = create_bias_variable('b', [1, self.Fout])
        if self.Fc is not None:
            self.wc = create_variable('wc', [self.Fc, self.Fout])

    def _call(self, X, A, c=None, ids=None):
        # input checking
        assert X.get_shape().as_list()[1] == self.Fin
        assert len(A) == self.D

        X = ops.mol_conv(X, A)
        X = tf.matmul(X, self.w)

        if self.Fc is not None:
            # input checking
            assert c is not None and c.get_shape().as_list()[-1] == self.Fc

            # for computational efficiency
            shape_old = tf.shape(c)
            c = tf.reshape(c, [-1, self.Fc])
            X_c = tf.matmul(c, self.wc)
            X_c = tf.reshape(X_c, tf.concat([shape_old[:-1],
                                             tf.constant(self.Fout,
                                                         dtype=tf.int32)[tf.newaxis]],
                                            axis=0))

            if ids is not None:
                X_c = tf.gather_nd(X_c, ids)
            X += X_c


        if self.batch_norm:
            X = self.bn(X)
        else:
            if self.init_bias is not None:
                X += self.b + self.init_bias

        X = self.activation(X)

        return X

