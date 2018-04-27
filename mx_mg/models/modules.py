import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn

import functions as fn


__all__ = ['Linear_BN', 'GraphConv', 'Policy', 'BatchNorm']

class Linear_BN(nn.Sequential):
    def __init__(self, F_in, F_out):
        super(Linear_BN, self).__init__()
        self.add(nn.Dense(F_out, in_units=F_in, use_bias=False))
        self.add(BatchNorm(in_channels=F_out))


class GraphConv(nn.Block):

    def __init__(self, Fin, Fout, D):
        super(GraphConv, self).__init__()

        # model settings
        self.Fin = Fin
        self.Fout = Fout
        self.D = D

        # model parameters
        self.W = self.params.get('w', shape=(self.Fin * (self.D + 1), self.Fout),
                                 init=None, allow_deferred_init=False)

    def forward(self, X, A_list):
        try:
            assert len(A_list) == self.D
        except AssertionError as e:
            print self.D, len(A_list)
            raise e
        return fn.EfficientGraphConvFn(A_list)(X, self.W.data(X.context))


class Policy(nn.Block):

    def __init__(self, F_in, F_h, N_A, N_B, k=1):
        super(Policy, self).__init__()
        self.F_in = F_in # number of input features for each atom
        self.F_h = F_h # number of context variables
        self.N_A = N_A # number of atom types
        self.N_B = N_B # number of bond types
        self.k = k # number of softmax used in the mixture


        with self.name_scope():
            self.linear_h = Linear_BN(F_in * 2, self.F_h * k)
            self.linear_h_t = Linear_BN(F_in, self.F_h * k)

            self.linear_x = nn.Dense(self.N_B + self.N_B*self.N_A, in_units=self.F_h)
            self.linear_x_t = nn.Dense(1, in_units=self.F_h)

            if self.k > 1:
                self.linear_pi = nn.Dense(self.k, in_units=self.F_in)
            else:
                self.linear_pi = None

    def forward(self, X, NX, NX_rep, X_end=None):
        # segment mean for X
        if X_end is None:
            X_end = fn.SegmentSumFn(NX_rep, NX.shape[0])(X)/nd.cast(fn.unsqueeze(NX, 1), 'float32')
        X = nd.concat(X, X_end[NX_rep, :], dim=1)

        X_h = nd.relu(self.linear_h(X)).reshape([-1, self.F_h])
        X_h_end = nd.relu(self.linear_h_t(X_end)).reshape([-1, self.F_h])

        X_x = nd.exp(self.linear_x(X_h)).reshape([-1, self.k, self.N_B + self.N_B*self.N_A])
        X_x_end = nd.exp(self.linear_x_t(X_h_end)).reshape([-1, self.k, 1])

        X_sum = nd.sum(fn.SegmentSumFn(NX_rep, NX.shape[0])(X_x), -1, keepdims=True) + X_x_end
        X_sum_gathered = X_sum[NX_rep, :, :]

        X_softmax = X_x / X_sum_gathered
        X_softmax_end = X_x_end/ X_sum

        if self.k > 1:
            pi = fn.unsqueeze(nd.softmax(self.linear_pi(X_end), axis=1), -1)
            pi_gathered = pi[NX_rep, :, :]

            X_softmax = nd.sum(X_softmax * pi_gathered, axis=1)
            X_softmax_end = nd.sum(X_softmax_end * pi, axis=1)
        else:
            X_softmax = fn.squeeze(X_softmax, 1)
            X_softmax_end = fn.squeeze(X_softmax_end, 1)

        # generate output
        connect, append = X_softmax[:, :self.N_B], X_softmax[:, self.N_B:]
        append = append.reshape([-1, self.N_A, self.N_B])
        end = fn.squeeze(X_softmax_end, -1)

        return append, connect, end


class BatchNorm(nn.Block):

    def __init__(self, in_channels, momentum=0.9, eps=1e-5):
        super(BatchNorm, self).__init__()
        self.F = in_channels

        self.bn_weight = self.params.get('bn_weight', shape=(self.F,), init=mx.init.One(),
                                         allow_deferred_init=False)
        self.bn_bias = self.params.get('bn_bias', shape=(self.F,), init=mx.init.Zero(),
                                       allow_deferred_init=False)

        self.running_mean = self.params.get('running_mean', grad_req='null',
                                            shape=(self.F,),
                                            init=mx.init.Zero(),
                                            allow_deferred_init=False,
                                            differentiable=False)
        self.running_var = self.params.get('running_var', grad_req='null',
                                           shape=(self.F,),
                                           init=mx.init.One(),
                                           allow_deferred_init=False,
                                           differentiable=False)
        self.momentum = momentum
        self.eps = eps

    def forward(self, x):
        # return fn.BatchNormFn(self.running_mean.data(x.context),
        #                       self.running_var.data(x.context),
        #                       self.momentum, self.eps)(x, self.bn_weight.data(x.context),
        #                                                self.bn_bias.data(x.context))
        if autograd.is_training():
            return nd.BatchNorm(x,
                                gamma=self.bn_weight.data(x.context),
                                beta=self.bn_bias.data(x.context),
                                moving_mean=self.running_mean.data(x.context),
                                moving_var=self.running_var.data(x.context),
                                eps=self.eps, momentum=self.momentum,
                                use_global_stats=False)
        else:
            return nd.BatchNorm(x,
                                gamma=self.bn_weight.data(x.context),
                                beta=self.bn_bias.data(x.context),
                                moving_mean=self.running_mean.data(x.context),
                                moving_var=self.running_var.data(x.context),
                                eps=self.eps, momentum=self.momentum,
                                use_global_stats=True)
