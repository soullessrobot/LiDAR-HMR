import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.geometry import rot6d_to_rotmat
from modules.st_gcn import STGCN
from pointnet2_ops.pointnet2_modules import PointnetSAModule
from typing import Tuple
from modules.geometry import rotation_matrix_to_axis_angle

import torch
import torch.nn as nn
import torch.nn.functional as F


from modules.smpl import SMPL

class PointNet2Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[0, 64, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 512, 1024], use_xyz=True
            )
        )

    def _break_up_pc(self, pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xyz = pc[..., :3].contiguous()
        features = pc[..., 3:].transpose(
            1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, data):
        x = data#['human_points']  # (B, T, N, 3)
        B, T, N, _ = x.shape
        x = x.reshape(-1, N, 3)  # (B * T, N, 3)
        xyz, features = self._break_up_pc(x)
        for module in self.SA_modules:
            xyz, features = module(xyz, features)
        features = features.squeeze(-1).reshape(B, T, -1)
        return features

class RNN(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_rnn_layer=2, bidirectional = True):
        super(RNN, self).__init__()
        self.rnn = nn.GRU(n_hidden, n_hidden, n_rnn_layer,
                          batch_first=True, bidirectional=bidirectional)#True
        self.linear1 = nn.Linear(n_input, n_hidden)
        if bidirectional:
            self.linear2 = nn.Linear(n_hidden * 2, n_output) # * 2
        else:
            self.linear2 = nn.Linear(n_hidden, n_output)

        self.dropout = nn.Dropout()

    def forward(self, x):  # (B, T, D)
        x = self.rnn(F.relu(self.dropout(self.linear1(x)), inplace=True))[0]
        return self.linear2(x)

class Regressor(nn.Module):
    def __init__(self, bidirectional = False):
        super().__init__()
        self.encoder = PointNet2Encoder()
        self.pose_s1 = RNN(1024, 24 * 3, 1024, bidirectional=bidirectional)
        self.ts_embedding = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.Linear(512, 128)
        )
        self.trans_de = RNN(128, 3, 64, bidirectional=bidirectional)
        self.shape_de = RNN(128, 10, 64, bidirectional=bidirectional)
        self.pose_s2 = STGCN(3 + 1024)
        self.smpl = SMPL()

    def forward(self, data):
        pred = {}
        data = data.unsqueeze(1)
        # print(data.shape)
        x = self.encoder(data)  # (B, T, D)
        B, T, _ = x.shape
        full_joints = self.pose_s1(x)  # (B, T, 24 * 3)
        # betas, trans
        x_ts = self.ts_embedding(x.detach())
        pred_beta = self.shape_de(x_ts) # (B, 10)
        trans = self.trans_de(x_ts) # (B, 3)
        
        rot6ds = self.pose_s2(torch.cat((full_joints.reshape(
            B, T, 24, 3), x.unsqueeze(-2).repeat(1, 1, 24, 1)), dim=-1))
        rot6ds = rot6ds.reshape(-1, rot6ds.size(-1))  # (B * T, D)
        rotmats = rot6d_to_rotmat(
            rot6ds).reshape(-1, 3, 3)  # (B * T * 24, 3, 3)
        pred_vertices = self.smpl(
                rotmats.reshape(-1, 24, 3, 3), pred_beta.reshape(-1, 10))
        
        pred['pred_pose'] = rotation_matrix_to_axis_angle(rotmats).reshape(B,T,-1)[:,0]
        pred['pred_full_joints'] = full_joints.reshape(B, T, 24, 3)[:,0]
        pred['pred_beta'] = pred_beta.reshape(B, T, 10)[:,0]
        pred['pred_trans'] = trans.squeeze(1)
        pred['mesh_out'] = pred_vertices
        # import pdb; pdb.set_trace()
        return pred
    