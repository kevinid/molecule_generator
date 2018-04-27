from mxnet import nd
from mxnet.autograd import Function
import math


__all__ = ['GraphConvFn', 'EfficientGraphConvFn', 'SegmentSumFn',
           'squeeze', 'unsqueeze', 'logsumexp', 'kl_divergence', 'get_activation', 'log_prob']

class GraphConvFn(Function):

    def __init__(self, A):
        self.A = A # type: nd.sparse.CSRNDArray
        self.A_T = self.A # assume symmetric
        super(GraphConvFn, self).__init__()

    def forward(self, X):
        if self.A is not None:
            if len(X.shape) > 2:
                X_resized = X.reshape((X.shape[0], -1))
                output = nd.sparse.dot(self.A, X_resized)
                output = output.reshape([-1, ] + [X.shape[i] for i in range(1, len(X.shape))])
            else:
                output = nd.sparse.dot(self.A, X)
            return output
        else:
            return nd.zeros_like(X)

    def backward(self, grad_output):

        if self.A is not None:
            if len(grad_output.shape) > 2:
                grad_output_resized = grad_output.reshape((grad_output.shape[0], -1))
                grad_input = nd.sparse.dot(self.A_T, grad_output_resized)
                grad_input = grad_input.reshape([-1] + [grad_output.shape[i]
                                                        for i in range(1, len(grad_output.shape))])
            else:
                grad_input = nd.sparse.dot(self.A_T, grad_output)
            return grad_input
        else:
            return nd.zeros_like(grad_output)


class EfficientGraphConvFn(Function):
    """Save memory by re-computation"""

    def __init__(self, A_list):
        self.A_list = A_list
        super(EfficientGraphConvFn, self).__init__()

    def forward(self, X, W):
        X_list = [X]
        for A in self.A_list:
            if A is not None:
                X_list.append(nd.sparse.dot(A, X))
            else:
                X_list.append(nd.zeros_like(X))
        X_out = nd.concat(*X_list, dim=1)
        self.save_for_backward(X, W)

        return nd.dot(X_out, W)

    def backward(self, grad_output):
        X, W = self.saved_tensors

        # recompute X_out
        X_list = [X, ]
        for A in self.A_list:
            if A is not None:
                X_list.append(nd.sparse.dot(A, X))
            else:
                X_list.append(nd.zeros_like(X))
        X_out = nd.concat(*X_list, dim=1)

        grad_W = nd.dot(X_out.T, grad_output)

        grad_X_out = nd.dot(grad_output, W.T)
        grad_X_out_list = nd.split(grad_X_out, num_outputs=len(self.A_list) + 1)


        grad_X = [grad_X_out_list[0], ]
        for A, grad_X_out in zip(self.A_list, grad_X_out_list[1:]):
            if A is not None:
                grad_X.append(nd.sparse.dot(A, grad_X_out))
            else:
                grad_X.append(nd.zeros_like(grad_X_out))

        grad_X = sum(grad_X)

        return grad_X, grad_W


class SegmentSumFn(GraphConvFn):

    def __init__(self, idx, num_seg):
        # build A
        # construct coo
        data = nd.ones(idx.shape[0], ctx=idx.context, dtype='int32')
        row, col = idx, nd.arange(idx.shape[0], ctx=idx.context, dtype='int32')
        shape = (num_seg, int(idx.shape[0]))
        sparse = nd.sparse.csr_matrix((data, (row, col)), shape=shape,
                                      ctx=idx.context, dtype='float32')
        super(SegmentSumFn, self).__init__(sparse)

        sparse = nd.sparse.csr_matrix((data, (col, row)), shape=(shape[1], shape[0]),
                                      ctx=idx.context, dtype='float32')
        self.A_T = sparse


def squeeze(input, axis):
    assert input.shape[axis] == 1

    new_shape = list(input.shape)
    del new_shape[axis]

    return input.reshape(new_shape)


def unsqueeze(input, axis):
    return nd.expand_dims(input, axis=axis)


def logsumexp(inputs, axis=None, keepdims=False):
    """Numerically stable logsumexp.

    Args:
        inputs: A Variable with any shape.
        axis: An integer.
        keepdims: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).

    Adopted from: https://github.com/pytorch/pytorch/issues/2591
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if axis is None:
        inputs = inputs.reshape([-1])
        axis = 0
    s = nd.max(inputs, axis=axis, keepdims=True)
    outputs = s + (inputs - s).exp().sum(axis=axis, keepdims=True).log()
    if not keepdims:
        outputs = nd.sum(outputs, axis=axis, keepdims=False)
    return outputs


def kl_divergence(mu, var, nits=0.0):
    num_mols = mu.shape[0]
    loss_kl = - 0.5 * nd.sum(1 + nd.log(var + 1e-10) - mu ** 2 - var, axis=0)
    loss_kl = loss_kl / num_mols
    loss_kl = nd.maximum(loss_kl, nits)
    return loss_kl.sum()


def get_activation(name):
    activation_dict = {
        'relu':nd.relu,
        'tanh':nd.tanh
    }
    return activation_dict[name]


def log_prob(mu, var, x):
    if not isinstance(var, nd.NDArray):
        log_var = math.log(var)
    else:
        log_var = nd.log(var)

    return - 0.5 * math.log(2 * math.pi) - 0.5 * log_var - (x - mu) ** 2 / (2 * var)