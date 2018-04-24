import math
import torch
from torch import nn
from torch.nn.parallel.data_parallel import replicate, parallel_apply, gather
import torch.nn.functional as F
from abc import ABCMeta, abstractmethod

from . import modules
from .functions import SegmentSumFn, logsumexp


class DataParallel_Explicit(nn.Module):
    """
    This data parallel implementation is based largely on
    the original nn.DataParallel, except that the inputs and outputs for each gpu is
    provided in separate lists. This also means that the provided inputs should be
    scattered manually to all gpu devices.
    """

    def __init__(self, module, device_ids=None, output_device=None,
                 gather_dim=None):
        super(DataParallel_Explicit, self).__init__()

        if not torch.cuda.is_available():
            self.module = module
            self.device_ids = []
            return

        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]
        self.gather_dim = gather_dim
        self.module = module
        self.device_ids = device_ids
        self.output_device = output_device

        if len(self.device_ids) == 1:
            self.module.cuda(device_ids[0])

    def forward(self, *inputs, **kwargs):
        # for non-parallel case
        if not self.device_ids:
            return self.module(*inputs[0], **kwargs)

        assert len(inputs) == len(self.device_ids)
        # we assume that inputs have already been scattered to different devices

        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs)

        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, [kwargs, ]*len(inputs))

        if self.gather_dim is None:
            # the outputs should be gathered manually
            return outputs
        else:
            return self.gather(outputs, self.output_device)

    def replicate(self, module, device_ids):
        return replicate(module, device_ids)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.gather_dim)


class MoleculeGenerator(nn.Module):

    __metaclass__ = ABCMeta

    def __init__(self, N_A, N_B, D, F_e, F_skip, F_c, F_h_policy, k_softmax, *args, **kwargs):
        super(MoleculeGenerator, self).__init__()
        self.N_A = N_A
        self.N_B = N_B
        self.D = D
        self.F_e = F_e
        self.F_skip = F_skip
        self.F_c = list(F_c) if isinstance(F_c, tuple) else F_c
        self.F_h_policy = F_h_policy
        self.k_softmax = k_softmax

        # embeddings
        self.embedding_atom = nn.Embedding(self.N_A, self.F_e)
        self.embedding_mask = nn.Embedding(3, self.F_e)

        # graph conv
        self._build_graph_conv(*args, **kwargs)

        # fully connected
        self.linear_fc, self.linear_bn = [], []
        for i, (f_in, f_out) in enumerate(zip([self.F_skip, ] + self.F_c[:-1], self.F_c)):
            # linear
            self.linear_fc.append(nn.Linear(f_in, f_out, bias=False))
            # batch norm
            self.linear_bn.append(nn.BatchNorm1d(f_out))
        self.linear_fc, self.linear_bn = nn.ModuleList(self.linear_fc), nn.ModuleList(self.linear_bn)

        # policy
        self.policy_0 = nn.Parameter(torch.Tensor(self.N_A))
        self.policy_h = modules.Policy(self.F_c[-1], self.F_h_policy, self.N_A, self.N_B, self.k_softmax)

        self._reset_parameters()

    def _reset_parameters(self):
        self.policy_0.data.zero_()

    @abstractmethod
    def _build_graph_conv(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _graph_conv_forward(self, X, A):
        raise NotImplementedError

    def _policy_0(self):
        policy_0 = torch.exp(self.policy_0)
        policy_0 = policy_0/policy_0.sum()
        return policy_0

    def _policy(self, X, A, NX, NX_rep, last_append_mask):
        # get initial embedding
        X = self.embedding_atom(X) + self.embedding_mask(last_append_mask)

        # convolution
        X = self._graph_conv_forward(X, A)

        # policy
        append, connect, end = self.policy_h(X, NX, NX_rep)

        return append, connect, end

    def _likelihood(self, init, append, connect, end,
                    action_0, actions, iw_ids, log_p_sigma,
                    batch_size, iw_size):

        # decompose action:
        action_type, node_type, edge_type, append_pos, connect_pos = \
            actions[:, 0], actions[:, 1], actions[:, 2], actions[:, 3], actions[:, 4]
        _log_mask = lambda _x, _mask: _mask * torch.log(_x + 1e-10) + (1- _mask) * torch.zeros_like(_x).cuda(_x.get_device())

        # init
        init = init.view(batch_size * iw_size, self.N_A)
        loss_init = torch.log(init[torch.arange(0, action_0.size(0)).long().cuda(action_0.get_device()), action_0] + 1e-10)

        # end
        loss_end = _log_mask(end, (action_type == 2).float())

        # append
        loss_append = _log_mask(append[append_pos, node_type, edge_type], (action_type == 0).float())

        # connect
        loss_connect = _log_mask(connect[connect_pos, edge_type], (action_type == 1).float())

        # sum up results
        log_p_x = loss_end + loss_append + loss_connect
        log_p_x = torch.squeeze(SegmentSumFn(iw_ids, batch_size*iw_size)(log_p_x.unsqueeze(-1)))
        log_p_x += loss_init

        # reshape
        log_p_x = log_p_x.view(batch_size, iw_size)
        log_p_sigma = log_p_sigma.view(batch_size, iw_size)
        l = log_p_x - log_p_sigma
        l = logsumexp(l, dim=1) - math.log(float(iw_size))
        return l

    def forward(self, *input, mode='loss'):
        if mode=='loss' or mode=='likelihood':
            X, A, iw_ids, last_append_mask, \
            NX, NX_rep, action_0, actions, log_p, \
            batch_size, iw_size = input

            init = self._policy_0().repeat(batch_size * iw_size, 1)
            append, connect, end = self._policy(X, A, NX, NX_rep, last_append_mask)
            l = self._likelihood(init, append, connect, end, action_0, actions, iw_ids, log_p, batch_size, iw_size)
            if mode=='likelihood':
                return l
            else:
                return -l.mean()
        elif mode == 'decode_0':
            return self._policy_0()
        elif mode == 'decode_step':
            X, A, NX, NX_rep, last_append_mask = input
            return self._policy(X, A, NX, NX_rep, last_append_mask)

class VanillaMolGen(MoleculeGenerator):

    def __init__(self, N_A, N_B, D, F_e, F_h, F_skip, F_c, F_h_policy, k_softmax,
                 memory_efficient=False):
        super(VanillaMolGen, self).__init__(N_A, N_B, D, F_e, F_skip, F_c, F_h_policy, k_softmax,
                                            F_h, memory_efficient)

    def _build_graph_conv(self, F_h, memory_efficient=False):
        self.F_h = list(F_h) if isinstance(F_h, tuple) else F_h
        self.conv, self.bn = [], []
        for i, (f_in, f_out) in enumerate(zip([self.F_e] + self.F_h[:-1], self.F_h)):
            conv = modules.GraphConv(f_in, f_out, self.N_B + self.D, memory_efficient)
            self.conv.append(conv)

            bn = nn.BatchNorm1d(f_out)
            self.bn.append(bn)

        self.conv, self.bn = nn.ModuleList(self.conv), nn.ModuleList(self.bn)

        self.linear_skip = modules.Linear_BN(sum(self.F_h), self.F_skip)

    def _graph_conv_forward(self, X, A):
        X_out = [X]
        for conv, bn in zip(self.conv, self.bn):
            X = X_out[-1]
            X_out.append(F.relu(bn(conv(X, A))))
        X_out = torch.cat(X_out[1:], dim=1)
        return F.relu(self.linear_skip(X_out))

