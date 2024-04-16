import torch
import torch.nn as nn
import sys
import os
this_dir = os.path.dirname(__file__)
# print(this_dir)
sys.path.append(this_dir)
from v2v_net import V2VNet, Basic3DBlock
# from v2v_sparse import V2VNet, Basic3DBlock
import torch.nn.functional as F
# torch.backends.cudnn.enabled = False
import math
# import sparseconvnet as scn

class SoftArgmaxLayer(nn.Module):
    def __init__(self):
        super(SoftArgmaxLayer, self).__init__()
        self.beta = 100

    def forward(self, x, grids):
        #[B,C,X,Y,Z]
        
        batch_size = x.size(0)
        channel = x.size(1)
        x = x.reshape(batch_size, channel, -1, 1)
        # x = F.softmax(x, dim=2)
        x = F.softmax(self.beta * x, dim=2)
        grids = grids.view(1,1,-1,3)
        x = torch.mul(x, grids)
        x = torch.sum(x, dim=2)
        return x

class V2VPoseNet(nn.Module):
    # V2V-PoseNet: https://arxiv.org/abs/1711.07399
    def __init__(self, point_dim = 3, mid_feature_dim = 32, num_keypoints = 14):
        super(V2VPoseNet, self).__init__()
        self.voxel_length = 2 #
        self.voxel_offset = -1
        self.voxel_size = (64, 64, 64)
        self.v2v_part = V2VNet(1, mid_feature_dim)
        self.hm_part = Basic3DBlock(mid_feature_dim, num_keypoints, kernel_size = 3)
        self.vis_part = nn.Sequential(
            nn.Linear(mid_feature_dim, 32),
            nn.Linear(32, num_keypoints)
        ) #Basic3DBlock(mid_feature_dim, num_keypoints, kernel_size = 3)
        self.soft_arg = SoftArgmaxLayer()
        self.softmax_ = nn.Softmax(dim = -1)
        self.grids = self.get_grids()
        self.lambda1 = 1
        self.lambda2 = 1
        self.lambda3 = 1
        self.lambda4 = 1

    def forward(self, points_input, center_position = None):
        bs = points_input.shape[0]
        voxel_input = self.get_voxel(points_input, center_position = center_position)
        voxel_feat = self.v2v_part(voxel_input)
        xyz_hm = self.hm_part(voxel_feat)
        vis = self.vis_part(torch.mean(voxel_feat, dim = (2,3,4)))
        vis = F.sigmoid(vis)
        xyz = self.soft_arg(xyz_hm, self.grids.to(xyz_hm.device))
        return {'xyz':xyz, 'xyz_hm':xyz_hm, 'vis':vis}

    def get_grids(self):
        x,y,z = torch.arange(self.voxel_offset, self.voxel_length + self.voxel_offset, self.voxel_length/self.voxel_size[0]), \
            torch.arange(self.voxel_offset, self.voxel_length + self.voxel_offset, self.voxel_length/self.voxel_size[1]), \
            torch.arange(self.voxel_offset, self.voxel_length + self.voxel_offset, self.voxel_length/self.voxel_size[2])
        X,Y,Z = torch.meshgrid(x,y,z)
        return torch.stack([X,Y,Z], dim = 3) #[3,64,64,64]

    def get_voxel(self, points_input, center_position = None):
        bs = points_input.shape[0]
        if center_position is not None:
            center_ = center_position.unsqueeze(1) if len(center_position.shape) < 3 else center_position
            local_points = points_input - center_
        else:
            local_points = points_input
        voxel_input = torch.zeros((bs,) + (1,) + self.voxel_size).to(local_points.device)
        points_loc = ((local_points - self.voxel_offset) / self.voxel_length) #[B,N,3]
        points_index = (points_loc * self.voxel_size[0]).type(torch.long)
        points_index[points_index > self.voxel_size[0] - 1] = self.voxel_size[0] - 1
        points_index[points_index < 0] = 0
        for b in range(bs):
            voxel_input[b, :, points_index[b,:,0], points_index[b,:,1], points_index[b,:,2]] = 1
        return voxel_input
    
    def hm_loss(self, xyz_hm,  sample):
        criterion = nn.L1Loss(reduction='mean')
        gt_xyz = sample['smpl_joints_local']
        center = sample['global_trans']
        flag = sample['valid_flag'] if 'valid_flag' in sample else None
        # import pdb; pdb.set_trace()
        gt_hm = self.get_voxel(gt_xyz, center)
        bs = gt_xyz.shape[0]
        loss = 0
        if flag is not None:
            for b in range(bs):
                flag_ = flag[b] > 0.5
                loss += criterion(xyz_hm[b,flag_], gt_hm[b,flag_])
            loss = loss / bs
        else:
            loss = criterion(xyz_hm, gt_hm)
        return loss

    def vis_loss(self, vis, gt_vis_):
        # import pdb; pdb.set_trace()
        return self.lambda3 * F.binary_cross_entropy(vis, gt_vis_)

    def xyz_loss(self, xyz, gt_xyz_, flag = None):
        if flag is not None:
            return torch.mean(torch.norm(self.lambda2 * (xyz - gt_xyz_), dim = -1) * flag)
        else:
            return torch.mean(torch.norm(self.lambda2 * (xyz - gt_xyz_), dim = -1))

    def all_loss(self, ret_dict, sample):
        xyz = ret_dict['xyz']
        vis = ret_dict['vis']
        xyz_hm = ret_dict['xyz_hm']
        gt_xyz = sample['smpl_joints_local']
        vis_label = sample['vis_label']
        xyz_loss = self.xyz_loss(xyz, gt_xyz, flag = None)
        vis_loss = self.vis_loss(vis, vis_label)
        all_loss = xyz_loss + vis_loss
        return {'loss': all_loss, 'xyz_loss':xyz_loss, 'vis_loss':vis_loss}