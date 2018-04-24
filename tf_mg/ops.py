"""
Define commonly used operations
"""
import tensorflow as tf
import numpy as np


def get_len(x, i=0, out_type=tf.int32, name='get_len'):
    with tf.name_scope(name):
        val = tf.shape(x)[i]
        val = tf.cast(val, out_type)
        return val


def bool_to_index(x, name='bool_to_index'):
    # a modified version of where for 1-D tensor
    # get the index of elements that is True
    with tf.name_scope(name):
        return tf.reshape(tf.where(x), [-1])


def gather_gpu(params, indices, name='gather_gpu'):
    with tf.name_scope(name):
        sparse_helper = tf.SparseTensor(indices=tf.stack([tf.range(get_len(indices, out_type=tf.int64), dtype=tf.int64),
                                                          tf.cast(indices, dtype=tf.int64)], axis=1),
                                        values=tf.ones_like(indices, dtype=params.dtype),
                                        dense_shape=tf.stack([get_len(indices, out_type=tf.int64),
                                                              get_len(params, out_type=tf.int64)], axis=0))
        result = tf.sparse_tensor_dense_matmul(sparse_helper, params)
    return result


def segment_sum_gpu(x, ids, name='segment_sum_gpu'):
    with tf.name_scope(name):
        sparse_helper = tf.SparseTensor(indices=tf.stack([tf.cast(ids, dtype=tf.int64),
                                                          tf.range(get_len(ids, out_type=tf.int64), dtype=tf.int64)], axis=1),
                                        values=tf.ones_like(ids, dtype=x.dtype),
                                        dense_shape=tf.stack([tf.cast(tf.reduce_max(ids) + 1, dtype=tf.int64),
                                                              get_len(ids, out_type=tf.int64)], axis=0))
        result = tf.sparse_tensor_dense_matmul(sparse_helper, x)
    return result


def segment_mean_gpu(x, ids, nx, name='segment_mean_gpu'):
    with tf.name_scope(name):
        result = segment_sum_gpu(x, ids) # num_segments * num_features
        result /= tf.expand_dims(tf.cast(nx, x.dtype), axis=1)
    return result


def mol_conv(X, A, name='mol_conv'):
    with tf.name_scope(name):
        X_new = [X]
        for A_i in A:
            X_new_i = tf.sparse_tensor_dense_matmul(A_i, X)
            X_new.append(X_new_i)
        X_new = tf.concat(X_new, axis=1)

    return X_new


def log_prob(mu, var, x):
    return - 0.5 * np.log(2 * np.pi) - 0.5 * tf.log(var) - (x - mu) ** 2 / (2 * var)


ACTIVATIONS = {
    'relu':tf.nn.relu,
    'elu':tf.nn.elu,
    'softmax':tf.nn.softmax,
    'tanh':tf.nn.tanh,
    'selu':lambda _x: 1.0507 * tf.where(tf.greater(_x, 0), _x, 1.6733 * (tf.exp(_x) - 1.0)),
    'exp':tf.exp,
    'sigmoid':tf.nn.sigmoid,
    None:lambda _x:_x
}