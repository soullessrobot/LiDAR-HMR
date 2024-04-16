from __future__ import division
import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse
import math

class SparseMM(torch.autograd.Function):
    """Redefine sparse @ dense matrix multiplication to enable backpropagation.
    The builtin matrix multiplication operation does not support backpropagation in some cases.
    """
    @staticmethod
    def forward(ctx, sparse, dense):
        ctx.req_grad = dense.requires_grad
        ctx.save_for_backward(sparse)
        return torch.matmul(sparse, dense)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        sparse, = ctx.saved_tensors
        if ctx.req_grad:
            grad_input = torch.matmul(sparse.t(), grad_output)
        return None, grad_input

def spmm(sparse, dense):
    return SparseMM.apply(sparse, dense)


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class BertLayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class GraphResBlock(torch.nn.Module):
    """
    Graph Residual Block similar to the Bottleneck Residual Block in ResNet
    """
    def __init__(self, in_channels, out_channels, mesh_type='body', adj_mat = None):
        super(GraphResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin1 = GraphLinear(in_channels, out_channels // 2)
        self.conv = GraphConvolution(out_channels // 2, out_channels // 2, mesh_type, adj_mat = adj_mat)
        self.lin2 = GraphLinear(out_channels // 2, out_channels)
        self.skip_conv = GraphLinear(in_channels, out_channels)
        # print('Use BertLayerNorm in GraphResBlock')
        self.pre_norm = BertLayerNorm(in_channels)
        self.norm1 = BertLayerNorm(out_channels // 2)
        self.norm2 = BertLayerNorm(out_channels // 2)

    def forward(self, x):
        trans_y = F.relu(self.pre_norm(x)).transpose(1,2)
        y = self.lin1(trans_y).transpose(1,2)

        y = F.relu(self.norm1(y))
        y = self.conv(y)

        trans_y = F.relu(self.norm2(y)).transpose(1,2)
        y = self.lin2(trans_y).transpose(1,2)

        z = x+y

        return z

# class GraphResBlock(torch.nn.Module):
#     """
#     Graph Residual Block similar to the Bottleneck Residual Block in ResNet
#     """
#     def __init__(self, in_channels, out_channels, mesh_type='body'):
#         super(GraphResBlock, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.conv = GraphConvolution(self.in_channels, self.out_channels, mesh_type)
#         print('Use BertLayerNorm and GeLU in GraphResBlock')
#         self.norm = BertLayerNorm(self.out_channels)
#     def forward(self, x):
#         y = self.conv(x)
#         y = self.norm(y)
#         y = gelu(y)
#         z = x+y
#         return z

class GraphLinear(torch.nn.Module):
    """
    Generalization of 1x1 convolutions on Graphs
    """
    def __init__(self, in_channels, out_channels):
        super(GraphLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = torch.nn.Parameter(torch.FloatTensor(out_channels, in_channels))
        self.b = torch.nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        w_stdv = 1 / (self.in_channels * self.out_channels)
        self.W.data.uniform_(-w_stdv, w_stdv)
        self.b.data.uniform_(-w_stdv, w_stdv)

    def forward(self, x):
        return torch.matmul(self.W[None, :], x) + self.b[None, :, None]

class GraphConvolution(torch.nn.Module):
    """Simple GCN layer, similar to https://arxiv.org/abs/1609.02907."""
    def __init__(self, in_features, out_features, mesh='body', bias=True, adj_mat = None):
        super(GraphConvolution, self).__init__()
        device=torch.device('cuda')
        self.in_features = in_features
        self.out_features = out_features
        if adj_mat is None:
            if mesh=='body':
                adj_indices = torch.load('./models/graphormer/data/smpl_431_adjmat_indices.pt')
                adj_mat_value = torch.load('./models/graphormer/data/smpl_431_adjmat_values.pt')
                adj_mat_size = torch.load('./models/graphormer/data/smpl_431_adjmat_size.pt')
            elif mesh=='hand':
                adj_indices = torch.load('./models/graphormer/data/mano_195_adjmat_indices.pt')
                adj_mat_value = torch.load('./models/graphormer/data/mano_195_adjmat_values.pt')
                adj_mat_size = torch.load('./models/graphormer/data/mano_195_adjmat_size.pt')

            self.adjmat = torch.sparse_coo_tensor(adj_indices, adj_mat_value, size=adj_mat_size).to(device)
        else:
            self.adjmat = adj_mat
        self.weight = torch.nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.weight.size(1))
        stdv = 6. / math.sqrt(self.weight.size(0) + self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        if x.ndimension() == 2:
            support = torch.matmul(x, self.weight)
            output = torch.matmul(self.adjmat, support)
            if self.bias is not None:
                output = output + self.bias
            return output
        else:
            output = []
            for i in range(x.shape[0]):
                support = torch.matmul(x[i], self.weight)
                # output.append(torch.matmul(self.adjmat, support))
                output.append(spmm(self.adjmat, support))
            output = torch.stack(output, dim=0)
            # if self.adjmat.shape[0] != self.adjmat.shape[1]:
            #     import pdb; pdb.set_trace()
            if self.bias is not None:
                output = output + self.bias
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

def link_mat(j_num, bone_link, double_link):
    a = torch.eye(j_num)
    a[bone_link[:,0], bone_link[:,1]] = 0.5
    a[bone_link[:,1], bone_link[:,0]] = 0.5
    a[double_link[:,0], double_link[:,1]] = 0.1
    a[double_link[:,1], double_link[:,0]] = 0.1
    return a

class DynamicGraphConvolution(torch.nn.Module):
    """Simple GCN layer, similar to https://arxiv.org/abs/1609.02907."""
    def __init__(self, in_features, out_features, mesh='body', bias=True, norm = False):
        super(DynamicGraphConvolution, self).__init__()
        device=torch.device('cuda')
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.norm = norm
        bone_link = np.array([[1,0], [2,1], [1,3], [1,4], [3,5], [4,6], [5,7], [6,8], [2,9], [2,10], [9,11], [10,12], [11,13], [12,14]])
        double_link = np.array([[0,2],[0,3],[0,4], [1,5], [1,6], [1,9], [1,10], [2, 3], [2,4], [2,11], [2,12], [3,4], [3,7], [9,10], [9,13], [10, 14]])
        self.link_mat = link_mat(15, bone_link, double_link).cuda()
        self.factor = 10
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.weight.size(1))
        stdv = 6. / math.sqrt(self.weight.size(0) + self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, seg = None):
        if x.ndimension() == 2:
            support = torch.matmul(x, self.weight)
            adjmat = torch.matmul(torch.matmul(seg, self.link_mat), seg.permute(0,2,1))
            if self.norm:
                adjmat /= (torch.sum(adjmat, dim = 2).unsqueeze(-1).repeat([1,1,adjmat.shape[1]]) + 1e-3)
                adjmat *= self.factor
            adjmat -= torch.eye(adjmat.shape[1]).to(seg.device)
            output = torch.matmul(adjmat, support)
            if self.bias is not None:
                output = output + self.bias
            return output
        else:
            output = []
            adjmat = torch.matmul(torch.matmul(seg, self.link_mat), seg.permute(0,2,1))
            adjmat -= torch.eye(adjmat.shape[1]).to(seg.device)

            if self.norm:
                adjmat /= (torch.sum(adjmat, dim = 2).unsqueeze(-1).repeat([1,1,adjmat.shape[1]]) + 1e-3)
                adjmat *= self.factor
            for i in range(x.shape[0]):
                support = torch.matmul(x[i], self.weight)
                output.append(spmm(adjmat[i], support))
            output = torch.stack(output, dim=0)
            if self.bias is not None:
                output = output + self.bias
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class DynamicGraphResBlock(torch.nn.Module):
    """
    Graph Residual Block similar to the Bottleneck Residual Block in ResNet
    """
    def __init__(self, in_channels, out_channels, norm = False):
        super(DynamicGraphResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin1 = GraphLinear(in_channels, out_channels // 2)
        self.conv = DynamicGraphConvolution(out_channels // 2, out_channels // 2, norm = norm)
        self.lin2 = GraphLinear(out_channels // 2, out_channels)
        self.skip_conv = GraphLinear(in_channels, out_channels)
        # print('Use BertLayerNorm in GraphResBlock')
        self.pre_norm = BertLayerNorm(in_channels)
        self.norm1 = BertLayerNorm(out_channels // 2)
        self.norm2 = BertLayerNorm(out_channels // 2)

    def forward(self, x, seg = None):
        trans_y = F.relu(self.pre_norm(x)).transpose(1,2)
        y = self.lin1(trans_y).transpose(1,2)

        y = F.relu(self.norm1(y))
        y = self.conv(y, seg)

        trans_y = F.relu(self.norm2(y)).transpose(1,2)
        y = self.lin2(trans_y).transpose(1,2)

        z = x+y

        return z
