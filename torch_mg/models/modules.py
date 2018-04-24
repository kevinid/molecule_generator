import math
from functools import reduce
from operator import mul
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from .bottlenec import EfficientDensenetBottleneck, SharedAllocation
from .functions import GraphConvFn, EfficientGraphConvFn, SegmentSumFn


class Linear_BN(nn.Sequential):
    def __init__(self, F_in, F_out):
        super(Linear_BN, self).__init__()
        self.F_in = F_in
        self.F_out = F_out

        self.add_module('linear', nn.Linear(self.F_in, self.F_out, bias=False))
        self.add_module('bn', nn.BatchNorm1d(self.F_out))


class GraphConv(nn.Module):

    def __init__(self, Fin, Fout, D, memory_efficient=False):
        super(GraphConv, self).__init__()

        # model settings
        self.Fin = Fin
        self.Fout = Fout
        self.D = D
        self.memory_efficient = memory_efficient

        # model parameters
        self.W = nn.Parameter(torch.Tensor(self.Fout, self.Fin * (self.D + 1)))
        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1. / math.sqrt(self.Fin * (self.D + 1))
        self.W.data.uniform_(-stdv, stdv)

    def forward(self, X, A_list):
        assert len(A_list) == self.D
        if self.memory_efficient:
            return EfficientGraphConvFn(A_list)(X, self.W)
        else:
            X_out = [X]
            for A in A_list:
                X_out.append(GraphConvFn(A)(X))
            X_out = torch.cat(X_out, dim=1)
            X_out = F.linear(X_out, self.W)
            return X_out


class GraphDenseLayer(nn.Sequential):
    """
    Dense layer for graph convolution,
    adopted from https://github.com/gpleiss/efficient_densenet_pytorch/blob/master/models/densenet_efficient.py
    """

    def __init__(self, shared_allocation_1, shared_allocation_2,
                 F_in, F_bn, D, k, drop_rate, memory_efficient):
        super(GraphDenseLayer, self).__init__()
        self.shared_allocation_1 = shared_allocation_1
        self.shared_allocation_2 = shared_allocation_2
        self.drop_rate = drop_rate
        self.D = D

        self.add_module('bn', EfficientDensenetBottleneck(shared_allocation_1, shared_allocation_2,
                                                          F_in, F_bn * k))
        self.add_module('norm.2', nn.BatchNorm1d(F_bn * k)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        self.add_module('conv', GraphConv(F_bn * k, k, self.D, memory_efficient)),

    def forward(self, x):
        if isinstance(x, Variable):
            prev_features = [x]
        else:
            prev_features = x
        new_features = super(GraphDenseLayer, self).forward(prev_features)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class GraphDenseBlock(nn.Container):
    """
    Dense block for graph convolution,
    adopted from https://github.com/gpleiss/efficient_densenet_pytorch/blob/master/models/densenet_efficient.py
    """

    def __init__(self, N, F_in, F_bn, D, k, drop_rate, memory_efficient=False,  storage_size=1024):
        self.F_out = F_in + (k * N)
        self.shared_allocation_1 = SharedAllocation(storage_size)
        self.shared_allocation_2 = SharedAllocation(storage_size)

        super(GraphDenseBlock, self).__init__()
        for i in range(N):
            layer = GraphDenseLayer(self.shared_allocation_1, self.shared_allocation_2,
                                    F_in + i * k,
                                    F_bn, D, k, drop_rate, memory_efficient)
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, x):
        # Update storage type
        self.shared_allocation_1.type_as(x)
        self.shared_allocation_2.type_as(x)

        # Resize storage
        final_size = list(x.size())
        final_size[1] = self.final_num_features
        final_storage_size = reduce(mul, final_size, 1)
        self.shared_allocation_1.resize_(final_storage_size)
        self.shared_allocation_2.resize_(final_storage_size)

        outputs = [x]
        for module in self.children():
            outputs.append(module.forward(outputs))
        return torch.cat(outputs, dim=1)


class Policy(nn.Module):

    def __init__(self, F_in, F_h, N_A, N_B, k=1):
        super(Policy, self).__init__()
        self.F_in = F_in # number of input features for each atom
        self.F_h = F_h # number of context variables
        self.N_A = N_A # number of atom types
        self.N_B = N_B # number of bond types
        self.k = k # number of softmax used in the mixture


        self.linear_h = Linear_BN(self.F_in * 2, self.F_h * k)
        self.linear_h_t = Linear_BN(self.F_in, self.F_h * k)

        self.linear_x = nn.Linear(self.F_h, self.N_B + self.N_B*self.N_A)
        self.linear_x_t = nn.Linear(self.F_h, 1)

        if self.k > 1:
            self.linear_pi = nn.Linear(self.F_in, self.k)
        else:
            self.linear_pi = None

    def forward(self, X, NX, NX_rep):
        # segment mean for X
        X_end = SegmentSumFn(NX_rep, NX.size(0))(X)/NX.unsqueeze(1).float()
        X = torch.cat([X, torch.index_select(X_end, 0, NX_rep)], dim=1)

        X_h = F.relu(self.linear_h(X)).view(-1, self.F_h)
        X_h_end = F.relu(self.linear_h_t(X_end)).view(-1, self.F_h)

        X_x = torch.exp(self.linear_x(X_h)).view(-1, self.k, self.N_B + self.N_B*self.N_A)
        X_x_end = torch.exp(self.linear_x_t(X_h_end)).view(-1, self.k, 1)

        X_sum = torch.sum(SegmentSumFn(NX_rep, NX.size(0))(X_x), -1, keepdim=True) + X_x_end
        X_sum_gathered = torch.index_select(X_sum, 0, NX_rep)

        X_softmax = X_x / X_sum_gathered
        X_softmax_end = X_x_end/ X_sum

        if self.k > 1:
            pi = F.softmax(self.linear_pi(X_end), dim=1).unsqueeze(-1)
            pi_gathered = torch.index_select(pi, 0, NX_rep)

            X_softmax = torch.sum(X_softmax * pi_gathered, 1)
            X_softmax_end = torch.sum(X_softmax_end * pi, 1)
        else:
            X_softmax = X_softmax.squeeze(1)
            X_softmax_end = X_softmax_end.squeeze(1)

        # generate output
        connect, append = X_softmax[:, :self.N_B], X_softmax[:, self.N_B:]
        append = append.contiguous().view(-1, self.N_A, self.N_B)
        end = X_softmax_end.squeeze(-1)

        return append, connect, end


