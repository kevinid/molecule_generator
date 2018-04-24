"""
Contains the implementation of the efficient bottlenec used in Densenet-BC architecture,
as described in "Memory-Efficient Implementation of DenseNets"
The code is adopted from https://github.com/gpleiss/efficient_densenet_pytorch/blob/master/models/densenet_efficient.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function


class SharedAllocation(object):
    """
    A helper class which maintains a shared memory allocation.
    Used for concatenation and batch normalization.
    """
    def __init__(self, size):
        self._cpu_storage = torch.Storage(size)
        self._gpu_storages = []
        if torch.cuda.is_available():
            for device_idx in range(torch.cuda.device_count()):
                with torch.cuda.device(device_idx):
                    self._gpu_storages.append(torch.Storage(size).cuda())

    def type(self, t):
        if not t.is_cuda:
            self._cpu_storage = self._cpu_storage.type(t)
        else:
            for device_idx, storage in enumerate(self._gpu_storages):
                with torch.cuda.device(device_idx):
                    self._gpu_storages[device_idx] = storage.type(t)

    def type_as(self, obj):
        if isinstance(obj, Variable):
            if not obj.is_cuda:
                self._cpu_storage = self._cpu_storage.type(obj.data.storage().type())
            else:
                for device_idx, storage in enumerate(self._gpu_storages):
                    with torch.cuda.device(device_idx):
                        self._gpu_storages[device_idx] = storage.type(obj.data.storage().type())
        elif torch.is_tensor(obj):
            if not obj.is_cuda:
                self._cpu_storage = self._cpu_storage.type(obj.storage().type())
            else:
                for device_idx, storage in enumerate(self._gpu_storages):
                    with torch.cuda.device(device_idx):
                        self._gpu_storages[device_idx] = storage.type(obj.storage().type())
        else:
            if not obj.is_cuda:
                self._cpu_storage = self._cpu_storage.type(obj.storage().type())
            else:
                for device_idx, storage in enumerate(self._gpu_storages):
                    with torch.cuda.device(device_idx):
                        self._gpu_storages[device_idx] = storage.type(obj.type())

    def resize_(self, size):
        if self._cpu_storage.size() < size:
            self._cpu_storage.resize_(size)
        for device_idx, storage in enumerate(self._gpu_storages):
            if storage.size() < size:
                with torch.cuda.device(device_idx):
                    self._gpu_storages[device_idx].resize_(size)
        return self

    def storage_for(self, val):
        if val.is_cuda:
            with torch.cuda.device_of(val):
                curr_device_id = torch.cuda.current_device()
                return self._gpu_storages[curr_device_id]
        else:
            return self._cpu_storage


class _EfficientDensenetBottleneckFn(Function):
    """
    The autograd function which performs the efficient bottlenck operations:
    --
    1) concatenation
    2) Batch Normalization
    3) ReLU
    --
    Convolution is taken care of in a separate function
    NOTE:
    The output of the function (ReLU) is written on a temporary memory allocation.
    If the output is not used IMMEDIATELY after calling forward, it is not guarenteed
    to be the ReLU output
    """
    def __init__(self, shared_allocation_1, shared_allocation_2,
                 running_mean, running_var,
                 training=False, momentum=0.1, eps=1e-5):

        self.shared_allocation_1 = shared_allocation_1
        self.shared_allocation_2 = shared_allocation_2
        self.running_mean = running_mean
        self.running_var = running_var
        self.training = training
        self.momentum = momentum
        self.eps = eps

        # Buffers to store old versions of bn statistics
        self.prev_running_mean = self.running_mean.new(self.running_mean.size())
        self.prev_running_var = self.running_var.new(self.running_var.size())

    def forward(self, bn_weight, bn_bias, *inputs):
        if self.training:
            # Save the current BN statistics for later
            self.prev_running_mean.copy_(self.running_mean)
            self.prev_running_var.copy_(self.running_var)

        # Create tensors that use shared allocations
        # One for the concatenation output (bn_input)
        # One for the ReLU output (relu_output)
        all_num_channels = [input.size(1) for input in inputs]
        size = list(inputs[0].size())
        for num_channels in all_num_channels[1:]:
            size[1] += num_channels
        storage = self.shared_allocation_1.storage_for(inputs[0])
        bn_input_var = Variable(type(inputs[0])(storage).resize_(size), volatile=True)
        relu_output = type(inputs[0])(storage).resize_(size)

        # Create variable, using existing storage
        torch.cat(inputs, dim=1, out=bn_input_var.data)

        # Do batch norm
        bn_weight_var = Variable(bn_weight)
        bn_bias_var = Variable(bn_bias)
        bn_output_var = F.batch_norm(bn_input_var, self.running_mean, self.running_var,
                                     bn_weight_var, bn_bias_var, training=self.training,
                                     momentum=self.momentum, eps=self.eps)

        # Do ReLU - and have the output be in the intermediate storage
        torch.clamp(bn_output_var.data, 0, 1e100, out=relu_output)

        self.save_for_backward(bn_weight, bn_bias, *inputs)
        if self.training:
            # restore the BN statistics for later
            self.running_mean.copy_(self.prev_running_mean)
            self.running_var.copy_(self.prev_running_var)
        return relu_output

    def prepare_backward(self):
        bn_weight, bn_bias = self.saved_tensors[:2]
        inputs = self.saved_tensors[2:]

        # Re-do the forward pass to re-populate the shared storage
        all_num_channels = [input.size(1) for input in inputs]
        size = list(inputs[0].size())
        for num_channels in all_num_channels[1:]:
            size[1] += num_channels
        storage1 = self.shared_allocation_1.storage_for(inputs[0])
        self.bn_input_var = Variable(type(inputs[0])(storage1).resize_(size), requires_grad=True)
        storage2 = self.shared_allocation_2.storage_for(inputs[0])
        self.relu_output = type(inputs[0])(storage2).resize_(size)

        # Create variable, using existing storage
        torch.cat(inputs, dim=1, out=self.bn_input_var.data)

        # Do batch norm
        self.bn_weight_var = Variable(bn_weight, requires_grad=True)
        self.bn_bias_var = Variable(bn_bias, requires_grad=True)
        self.bn_output_var = F.batch_norm(self.bn_input_var, self.running_mean, self.running_var,
                                          self.bn_weight_var, self.bn_bias_var, training=self.training,
                                          momentum=self.momentum, eps=self.eps)

        # Do ReLU
        torch.clamp(self.bn_output_var.data, 0, 1e100, out=self.relu_output)

    def backward(self, grad_output):
        """
        Precondition: must call prepare_backward before calling backward
        """

        grads = [None] * len(self.saved_tensors)
        inputs = self.saved_tensors[2:]

        # If we don't need gradients, don't run backwards
        if not any(self.needs_input_grad):
            return grads

        # BN weight/bias grad
        # With the shared allocations re-populated, compute ReLU/BN backward
        relu_grad_input = grad_output.masked_fill_(self.relu_output <= 0, 0)
        self.bn_output_var.backward(gradient=relu_grad_input)
        if self.needs_input_grad[0]:
            grads[0] = self.bn_weight_var.grad.data
        if self.needs_input_grad[1]:
            grads[1] = self.bn_bias_var.grad.data

        # Input grad (if needed)
        # Run backwards through the concatenation operation
        if any(self.needs_input_grad[2:]):
            all_num_channels = [input.size(1) for input in inputs]
            index = 0
            for i, num_channels in enumerate(all_num_channels):
                new_index = num_channels + index
                grads[2 + i] = self.bn_input_var.grad.data[:, index:new_index]
                index = new_index

        # Delete all intermediate variables
        del self.bn_input_var
        del self.bn_weight_var
        del self.bn_bias_var
        del self.bn_output_var

        return tuple(grads)


class _DummyBackwardHookFn(Function):
    """
    A dummy function, which is just designed to run a backward hook
    This allows us to re-populate the shared storages before running the backward
    pass on the bottleneck layer
    The function itself is just an identity function
    """
    def __init__(self, fn):
        """
        fn: function to call "prepare_backward" on
        """
        self.fn = fn

    def forward(self, input):
        """
        Though this function is just an identity function, we have to return a new
        tensor object in order to trigger the autograd.
        """
        size = input.size()
        res = input.new(input.storage()).view(*size)
        return res

    def backward(self, grad_output):
        self.fn.prepare_backward()
        return grad_output


class EfficientDensenetBottleneck(nn.Module):
    """
    A optimized layer which encapsulates the batch normalization, ReLU, and
    convolution operations within the bottleneck of a DenseNet layer.
    This layer usage shared memory allocations to store the outputs of the
    concatenation and batch normalization features. Because the shared memory
    is not perminant, these features are recomputed during the backward pass.
    """
    def __init__(self, shared_allocation_1, shared_allocation_2, num_input_channels, num_output_channels):
        super(EfficientDensenetBottleneck, self).__init__()
        self.shared_allocation_1 = shared_allocation_1
        self.shared_allocation_2 = shared_allocation_2
        self.num_input_channels = num_input_channels

        self.norm_weight = nn.Parameter(torch.Tensor(num_input_channels))
        self.norm_bias = nn.Parameter(torch.Tensor(num_input_channels))
        self.register_buffer('norm_running_mean', torch.zeros(num_input_channels))
        self.register_buffer('norm_running_var', torch.ones(num_input_channels))
        self.bottlenec_weight = nn.Parameter(torch.Tensor(num_output_channels, num_input_channels))
        self._reset_parameters()

    def _reset_parameters(self):
        self.norm_running_mean.zero_()
        self.norm_running_var.fill_(1)
        self.norm_weight.data.uniform_()
        self.norm_bias.data.zero_()
        stdv = 1. / math.sqrt(self.num_input_channels)
        self.bottlenec_weight.data.uniform_(-stdv, stdv)

    def forward(self, inputs):
        if isinstance(inputs, Variable):
            inputs = [inputs]

        # The EfficientDensenetBottleneckFn performs the concatenation, batch norm, and ReLU.
        # It does not create any new storage
        # Rather, it uses a shared memory allocation to store the intermediate feature maps
        # These intermediate feature maps have to be re-populated before the backward pass
        fn = _EfficientDensenetBottleneckFn(self.shared_allocation_1, self.shared_allocation_2,
                                            self.norm_running_mean, self.norm_running_var,
                                            training=self.training, momentum=0.1, eps=1e-5)
        relu_output = fn(self.norm_weight, self.norm_bias, *inputs)

        # The convolutional output - using relu_output which is stored in shared memory allocation
        bottleneck_output = F.linear(relu_output, self.bottlenec_weight)

        # Register a hook to re-populate the storages (relu_output and concat) on backward pass
        # To do this, we need a dummy function
        dummy_fn = _DummyBackwardHookFn(fn)
        output = dummy_fn(bottleneck_output)

        # Return the convolution output
        return output