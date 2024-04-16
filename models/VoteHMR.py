import torch
import torch.nn as nn
import math

class CA(nn.Module):
    """
    Self Attention module.
    """

    def __init__(self, channels):
        super(CA, self).__init__()

        self.da = channels // 4

        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, y):
        """
        Input
            x: [B, de, N]
        
        Output
            x: [B, de, N]
        """
        # compute query, key and value matrix
        x_q = self.q_conv(x)  # [B, N, da]
        x_k = self.k_conv(y)                   # [B, da, N]        
        x_v = self.v_conv(y)                   # [B, de, N]
        
        # compute attention map and scale, the sorfmax
        energy = torch.bmm(x_q.permute(0,2,1), x_k) / (math.sqrt(self.da))   # [B, N, N]
        attention = self.softmax(energy)                      # [B, N, N]

        # weighted sum
        x_s = torch.bmm(x_v, attention.permute(0,2,1))#.permute(0,2,1)  # [B, de, N]
        # import pdb; pdb.set_trace()
        x_s = self.act(self.after_norm(self.trans_conv(x_s)))
        
        # residual
        x = x_s.permute(0,2,1) 
        # import pdb; pdb.set_trace()
        return x

class GC(torch.nn.Module):
    def __init__(self, in_features, out_features, adj_mat, bias=True):
        super(GC, self).__init__()
        device=torch.device('cuda')
        self.in_features = in_features
        self.out_features = out_features
        self.adjmat = adj_mat.cuda()
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
        support = torch.matmul(x, self.weight)
        output = torch.matmul(self.adjmat, support)
        if self.bias is not None:
            output = output + self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GlobalParamRegressor(nn.Module):
    def __init__(self, channels = 48):
        super().__init__()
        self.CAU = CA(channels=channels)
        self.MLP_mid = nn.Linear(channels * 2, channels)
        self.MLP_global = nn.Linear(channels * 24, 22)
    
    def forward(self, global_f, joint_f):
        # [B,1,C] [B,K,C]
        batch_size, num_j = joint_f.shape[:2]
        agf = self.CAU(global_f, joint_f) #[B,1,C]
        # import pdb; pdb.set_trace()
        joint_f = joint_f.permute(0,2,1)
        jaf = joint_f - agf
        jjf = torch.cat([joint_f, jaf], dim = -1)
        jjf = self.MLP_mid(jjf)
        gl_ = self.MLP_global(jjf.view(batch_size, -1))
        betas = gl_[...,:10]
        global_orient = gl_[...,10:-3].view(-1,1,3,3)
        global_orient = global_orient / (global_orient.norm(dim = -1, keepdim = True) + 1e-8)
        # det_ori = torch.det(global_orient).unsqueeze(2).unsqueeze(3) + 1e-8
        # global_orient = global_orient / (det_ori.pow(1/3))
        trans = gl_[...,-3:]
        return {'betas':betas, 'global_orient':global_orient, 'trans':trans}
    
class LocalParamRegressor(nn.Module):
    def __init__(self, graph_adj, channels = 48):
        super().__init__()
        self.GC = nn.Sequential(
            GC(channels, channels, adj_mat=graph_adj),
            GC(channels, channels, adj_mat=graph_adj)
        )
        self.MLP_local = nn.Linear(channels, 9)
    
    def forward(self, joint_f):
        # [B,1,C] [B,K,C]
        joint_f = joint_f.permute(0,2,1)
        batch_size, num_j = joint_f.shape[:2]
        gf = self.GC(joint_f)
        local_orient = self.MLP_local(gf).view(batch_size,-1,3,3)
        # det_ori = torch.det(local_orient).unsqueeze(2).unsqueeze(3) 
        # import pdb; pdb.set_trace()
        # local_orient = local_orient / (det_ori.pow(1/3) + 1e-8)
        local_orient = local_orient / (local_orient.norm(dim = -1, keepdim = True) + 1e-8)
        return {'local_orient':local_orient}
