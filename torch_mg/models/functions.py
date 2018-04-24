import torch
from torch.autograd import Function, Variable
import torch.nn.functional as F
import gc


class GraphConvFn(Function):

    def __init__(self, A):
        self.A = A
        super(GraphConvFn, self).__init__()

    def forward(self, X):
        if self.A is not None:
            if X.dim() > 2:
                X_resized = X.view(X.size(0), -1)
                output = torch.mm(self.A, X_resized)
                output = output.view(*([-1, ] + [X.size(i) for i in range(1, X.dim())]))
            else:
                output = torch.mm(self.A, X)
            return output
        else:
            return torch.zeros_like(X).cuda(X.get_device())

    def backward(self, grad_output):
        if self.needs_input_grad[0]:
            if self.A is not None:
                if grad_output.dim() > 2:
                    grad_output_resized = grad_output.contiguous().view(grad_output.size(0), -1)
                    grad_input = torch.mm(self.A.t(), grad_output_resized)
                    grad_input = grad_input.view(*([-1] + [grad_output.size(i) for i in range(1, grad_output.dim())]))
                else:
                    grad_input = torch.mm(self.A.t(), grad_output)
                return grad_input
            else:
                return torch.zeros_like(grad_output).cuda(grad_output.get_device())
        else:
            return None


class EfficientGraphConvFn(Function):
    """Save memory by re-computation"""

    def __init__(self, A_list):
        self.A_list = A_list
        self.X, self.W = None, None  # store saved input
        super(EfficientGraphConvFn, self).__init__()

    def forward(self, X, W):
        self.X, self.W = X, W  # store input for grad

        X_list = [X]
        for A in self.A_list:
            if A is not None:
                X_list.append(torch.mm(A, X))
            else:
                X_list.append(torch.zeros_like(X).cuda(X.get_device()))
        X_out = torch.cat(X_list, dim=1)
        return F.linear(X_out, W)

    def backward(self, grad_output):
        # recompute X_out
        X_list = [self.X]
        for A in self.A_list:
            if A is not None:
                X_list.append(torch.mm(A, self.X))
            else:
                X_list.append(torch.zeros_like(self.X).cuda(self.X.get_device()))
        X_out = torch.cat(X_list, dim=1)

        grad_X, grad_W = None, None
        if self.needs_input_grad[0]:
            grad_X_out = torch.matmul(grad_output, self.W)
            grad_X_out_list = torch.split(grad_X_out,
                                          split_size=int(grad_X_out.size(1)/(len(self.A_list) + 1)),
                                          dim=1)

            grad_X = [grad_X_out_list[0], ]
            for A, grad_X_out in zip(self.A_list, grad_X_out_list[1:]):
                if A is not None:
                    grad_X.append(torch.mm(A.t(), grad_X_out))
                else:
                    grad_X.append(torch.zeros_like(grad_X_out).cuda(grad_X_out.get_device()))
            grad_X = sum(grad_X)
            del grad_X_out_list, grad_X_out

        if self.needs_input_grad[1]:
            grad_W = torch.matmul(grad_output.t(), X_out)

        del X_out, X_list

        gc.collect()

        return grad_X, grad_W


class SegmentSumFn(GraphConvFn):

    def __init__(self, idx, num_seg):
        # build A
        device_id = idx.get_device()
        sparse = torch.cuda.sparse.FloatTensor(torch.stack([idx.data, torch.arange(0, idx.size(0)).long().cuda(device_id)], dim=0),
                                               torch.ones(idx.size(0)).cuda(device_id),
                                               torch.Size([num_seg, idx.size(0)]))
        super(SegmentSumFn, self).__init__(sparse)


def logsumexp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).

    Adopted from: https://github.com/pytorch/pytorch/issues/2591
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs