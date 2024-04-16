import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch import Tensor
from ..pct.point_transformer_v2 import PointTransformerV2
import numpy as np
from models.lpformer.lpformer import BertSelfAttention
from pytorch3d.transforms import quaternion_to_matrix#, matrix_to_axis_angle #quaternion_to_axis_angle
import smplx

class ChebGraphConv(nn.Module):
    def __init__(self, K, in_features, out_features, bias):
        super(ChebGraphConv, self).__init__()
        self.K = K
        self.weight = nn.Parameter(torch.FloatTensor(K + 1, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, gso):
        # Chebyshev polynomials:
        # x_0 = x,
        # x_1 = gso * x,
        # x_k = 2 * gso * x_{k-1} - x_{k-2},
        # where gso = 2 * gso / eigv_max - id.

        cheb_poly_feat = []
        if self.K < 0:
            raise ValueError('ERROR: The order of Chebyshev polynomials shoule be non-negative!')
        elif self.K == 0:
            # x_0 = x
            cheb_poly_feat.append(x)
        elif self.K == 1:
            # x_0 = x
            cheb_poly_feat.append(x)
            if gso.is_sparse:
                # x_1 = gso * x
                cheb_poly_feat.append(torch.sparse.mm(gso, x))
            else:
                if x.is_sparse:
                    x = x.to_dense
                # x_1 = gso * x
                # import pdb; pdb.set_trace()
                cheb_poly_feat.append(torch.bmm(gso, x))
        else:
            # x_0 = x
            cheb_poly_feat.append(x)
            if gso.is_sparse:
                # x_1 = gso * x
                cheb_poly_feat.append(torch.sparse.mm(gso, x))
                # x_k = 2 * gso * x_{k-1} - x_{k-2}
                for k in range(2, self.K):
                    cheb_poly_feat.append(torch.sparse.mm(2 * gso, cheb_poly_feat[k - 1]) - cheb_poly_feat[k - 2])
            else:
                if x.is_sparse:
                    x = x.to_dense
                # x_1 = gso * x
                cheb_poly_feat.append(torch.mm(gso, x))
                # x_k = 2 * gso * x_{k-1} - x_{k-2}
                for k in range(2, self.K):
                    cheb_poly_feat.append(torch.mm(2 * gso, cheb_poly_feat[k - 1]) - cheb_poly_feat[k - 2])
        
        feature = torch.stack(cheb_poly_feat, dim=1)
        if feature.is_sparse:
            feature = feature.to_dense()
        # import pdb; pdb.set_trace()
        cheb_graph_conv = torch.einsum('bijk,ikl->bjl', feature, self.weight)

        if self.bias is not None:
            cheb_graph_conv = torch.add(input=cheb_graph_conv, other=self.bias, alpha=1)
        else:
            cheb_graph_conv = cheb_graph_conv

        return cheb_graph_conv

    def extra_repr(self) -> str:
        return 'K={}, in_features={}, out_features={}, bias={}'.format(
            self.K, self.in_features, self.out_features, self.bias is not None
        )

class SGM(nn.Module):
    def __init__(self, pose_dim = 24):
        super().__init__()
        self.gso = torch.zeros(24,24)
        bone_link = (
            (0,1), (0,2), (1,4), (2,5), (4,7), (5,8), (8,11), (7,10), (0,3), (3,6), (6,9), 
            (9,14), (9,13), (9,12), (12, 15), (14, 17), (13, 16), (17, 19), (16, 18), (19, 21),
            (18, 20), (21, 23), (20, 22)
        )
        for bl in bone_link:
            b1, b2 = bl[0], bl[1]
            self.gso[b1, b2] = 1
            self.gso[b2, b1] = 1
        self.gso = self.gso / self.gso.norm(dim = 1, keepdim = True)
        self.gso = self.gso.unsqueeze(0)
        self.feat_embed = nn.Linear(48, 128)
        self.gcn_shape = ChebGraphConv(1, 128, 128, True)
        self.gcn1_pose = ChebGraphConv(1, 128, 128, True)
        self.gcn2_pose = ChebGraphConv(1, 128, 4, True)
        self.mlp1 = nn.Linear(128, 10)
        self.mlp2 = nn.Linear(240, 10)
    
    def forward(self, x):
        B = x.shape[0]
        gso_use = self.gso.repeat(B,1,1).to(x.device)
        x = self.feat_embed(x)
        pf = self.gcn1_pose(x, gso_use)
        pose = self.gcn2_pose(pf, gso_use)
        sf = self.mlp1(self.gcn_shape(x, gso_use)) #[B,24,10]
        shape = self.mlp2(sf.view(B,-1))
        return pose, shape

class SAHSR(nn.Module):
    # SAHSR: https://openaccess.thecvf.com/content_ICCV_2019/html/Jiang_Skeleton-Aware_3D_Human_Shape_Reconstruction_From_Point_Clouds_ICCV_2019_paper.html
    def __init__(self, pose_dim = 24, **kwargs):
        super().__init__()
        self.num_kp = pose_dim
        self.encoder = PointTransformerV2(3, pose_dim + 1, enc_depths = (2,2,2,2), enc_channels = (16, 32, 64, 128), \
                                          enc_groups = (2, 4, 8, 16))
        self.kpt_embedding = nn.Embedding(pose_dim, 48)
        self.AM = BertSelfAttention(48, 2)
        self.sgm = SGM()
        self.smpl_model = smplx.create('./smplx_models/', model_type = 'smpl',
                                    gender='neutral', 
                                    use_face_contour=False,
                                    ext="npz").cuda()
        
    def forward(self, xyz):
        B,N,_ = xyz.shape
        seg, feat = self.encoder(xyz.permute(0,2,1))
        feat = feat.view(B,N,-1)
        query = self.kpt_embedding(torch.arange(self.num_kp).to(xyz.device)).unsqueeze(0).repeat(B, 1, 1)#self.kpt_querys.unsqueeze(0).repeat(bs, 1, 1)
        featx = torch.cat([query, feat], dim = 1) #[B,N_kp+N_points,emb_dim]
        featx = self.AM(featx)
        kpt_feats = featx[:,:24]
        pose, shape = self.sgm(kpt_feats)
        pose = quaternion_to_matrix(pose)
        smpl_results = self.smpl_model(body_pose = pose[:,1:], betas = shape, global_orient = pose[:,[0]], pose2rot=False)
        verts = smpl_results.vertices
        joints = smpl_results.joints[...,:24,:]
        return {'pose':pose, 'shape':shape, 'mesh_out':verts, 'joints' : joints}
    
    def all_loss(self, ret_dict, sample):
        verts = ret_dict['mesh_out']
        pose_gt, shape_gt = sample['smpl_pose'], sample['betas']
        gt_verts = self.smpl_model(body_pose = pose_gt[...,3:], betas = shape_gt, global_orient = pose_gt[...,:3]).vertices
        vert_loss = (gt_verts - verts).norm(dim = -1).mean()
        return {'loss': vert_loss, 'verts_loss':vert_loss}