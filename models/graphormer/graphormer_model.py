from .e2e_body_network import Graphormer_Body_Network
import torch
import torch.nn as nn
import numpy as np
# from ..pct.pct_model import PCTSeg, Regression
from ..pct.point_transformer_v2 import PointTransformerV2
from .modeling_bert import BertConfig
from .modeling_graphormer import Graphormer, DynamicGraphNet
from ._smpl import SMPL as smpl_k
from ._smpl import Mesh
from ..pose2mesh.loss import get_loss

class graphormer_model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.pct = PCTSeg(samples = [1024, 1024], segment_num = 25, return_feature=True)
        self.pct = PointTransformerV2(3, 25)
        in_channels = [259, 128, 64]
        out_channels = [128, 64, 3]
        modules = []
        for (in_, out_) in zip(in_channels, out_channels):
            config1 = BertConfig.from_pretrained('/models/graphormer')
            config1.output_attentions = False
            config1.img_feature_dim = in_
            config1.output_feature_dim = out_
            config1.hidden_size = 24
            config1.graph_conv = True if out_ != 3 else False
            config1.mesh_type = 'body'
            config1.max_position_embeddings = 1024 + 431 + 15
            gphmr_ = Graphormer(config1)
            modules.append(gphmr_)
        gphmr = nn.Sequential(*modules)
        self.gphmr_net = Graphormer_Body_Network(gphmr, 48)
        self.smpl = smpl_k()
        self.mesh = Mesh()
        self.criterion_vertices = torch.nn.L1Loss()
        self.criterion_keypoints = torch.nn.MSELoss(reduction='none')
        self.criterion_segment = torch.nn.CrossEntropyLoss()
        self.criterion = get_loss(faces=self.smpl.faces)

    def forward(self, points):
        B,N,_ = points.shape
        # import pdb; pdb.set_trace()
        seg, feat = self.pct(points.permute(0,2,1))
        seg = seg.view(B,N,-1).permute(0,2,1)
        feat = feat.view(B,N,-1)
        # import pdb; pdb.set_trace()
        pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices = \
            self.gphmr_net(torch.cat([points, feat], dim = -1), self.smpl, self.mesh)
        ret_dict = {}
        ret_dict.update({'pose_out':pred_3d_joints, 'pred_verts':pred_vertices,
                          'pred_vert_sub': pred_vertices_sub, 'pred_vert_sub2':pred_vertices_sub2, 'seg':seg})
        return ret_dict

    def vertices_loss(self, criterion_vertices, pred_vertices, gt_vertices, has_smpl = None, device = 'cuda'):
        """
        Compute per-vertex loss if vertex annotations are available.
        """
        if has_smpl is not None:
            pred_vertices_with_shape = pred_vertices[has_smpl == 1]
            gt_vertices_with_shape = gt_vertices[has_smpl == 1]
        else:
            pred_vertices_with_shape = pred_vertices
            gt_vertices_with_shape = gt_vertices
        if len(gt_vertices_with_shape) > 0:
            return criterion_vertices(pred_vertices_with_shape, gt_vertices_with_shape)
        else:
            return torch.FloatTensor(1).fill_(0.).to(device) 

    def keypoint_3d_loss(self, criterion_keypoints, pred_keypoints_3d, gt_keypoints_3d):
        """
        Compute 3D keypoint loss if 3D keypoint annotations are available.
        """
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2,:] + pred_keypoints_3d[:, 3,:]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            # import pdb; pdb.set_trace()
            # return (criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
            return (pred_keypoints_3d - gt_keypoints_3d).norm(dim = -1, p = 2).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).cuda()

    def seg_loss(self, pred_seg, seg_label):
        # print(pred_seg.max(dim = 1)[0].mean(dim = 1))
        return self.criterion_segment(pred_seg, seg_label)

    def get_final_mesh_loss(self, pred_mesh, gt_mesh, val_mesh = None):
        # pred_mesh = ret_dict['mesh_out']
        # gt_mesh = sample['smpl_verts_local'].to(pred_mesh.device)
        # import pdb; pdb.set_trace()
        j_r = self.smpl.J_regressor.to(pred_mesh.device).float()
        pred_pose = (j_r @ pred_mesh)[:,self.valid_kpts_indx]
        gt_reg3dpose = (j_r @ gt_mesh)[:,self.valid_kpts_indx]
        # 
        loss1, loss2, loss4 = self.criterion[0](pred_mesh, gt_mesh, val_mesh) * self.final_mesh_weight,  \
                                                self.normal_weight * self.criterion[1](pred_mesh, gt_mesh) * self.final_mesh_weight, \
                                                self.joint_weight * self.criterion[3](pred_pose,  gt_reg3dpose, val_mesh) * self.final_mesh_weight
        loss3 = self.criterion[2](pred_mesh, gt_mesh) * self.edge_weight * self.final_mesh_weight
        # loss5 = self.criterion[5](pred_mesh, gt_mesh) * self.edge_weight * 10
        loss5 = 0
        loss = loss1 + loss4 + loss2 
        # print(loss1, loss2, loss3, loss5)
        # import pdb; pdb.set_trace()
        return {'loss': loss, 'loss_joint':loss4, 'loss_vert':loss1 + loss2, 'edge_loss': loss3 + loss5}

    def all_loss(self, ret_dict, sample):
        # smpl_pose = sample['smpl_pose']
        # global_trans = sample['global_trans']
        # betas = sample['betas']
        pred_vert_full = ret_dict['pred_verts']
        pred_vert_sub = ret_dict['pred_vert_sub']
        pred_vert_sub2 = ret_dict['pred_vert_sub2']
        
        # import pdb; pdb.set_trace()
        gt_verts = sample['smpl_verts_local'] #self.smpl(smpl_pose, betas)
        # gt_verts = gt_verts + global_trans.unsqueeze(1)
        gt_vertices_sub = self.mesh.downsample(gt_verts)
        gt_vertices_sub2 = self.mesh.downsample(gt_vertices_sub, n1=1, n2=2)
        gt_smpl_3d_joints = self.smpl.get_h36m_joints(gt_verts)
        pred_3d_joints = self.smpl.get_h36m_joints(pred_vert_full)

        vertices_loss1 = self.vertices_loss(self.criterion_vertices, pred_vert_sub2, gt_vertices_sub2)
        vertices_loss2 = self.vertices_loss(self.criterion_vertices, pred_vert_sub, gt_vertices_sub)
        vertices_loss_all = self.vertices_loss(self.criterion_vertices, pred_vert_full, gt_verts)
        vertices_loss = vertices_loss1 + vertices_loss2 + vertices_loss_all
        joints_loss = self.keypoint_3d_loss(self.criterion_keypoints, pred_3d_joints, gt_smpl_3d_joints)
        edge_loss = self.get_final_mesh_loss(pred_vert_full, gt_verts)
        seg_loss = self.seg_loss(ret_dict['seg'], sample['seg_label'].to(ret_dict['seg'].device))
        all_loss = vertices_loss + joints_loss + seg_loss
        return {'loss':all_loss, 'vertices_loss':vertices_loss, 'joint_loss':joints_loss, 'seg_loss':seg_loss, 'edge_loss': edge_loss}

# class graphormer_model_smpl(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.pct = PointTransformerV2(3, 25)
#         #PointTransformerV2(3, 25)
#         #PCTSeg(samples = [1024, 1024], segment_num = 25, return_feature=True)
#         # self.trans = Regression(out_dim = 3, input_dim = 48)
#         self.reg_feat = Regression(out_dim = 72, input_dim = 48)
#         # self.betas = Regression(out_dim = 10, input_dim = 48)
#         self.betas_ = nn.Linear(48, 10)
#         self.pose_ = nn.Linear(48, 72)
#         self.trans_ = nn.Linear(48, 3)
#         self.smpl = smpl_k()
#         self.mesh = Mesh()
#         self.criterion_vertices = torch.nn.L1Loss()
#         self.criterion_keypoints = torch.nn.MSELoss(reduction='none')
#         self.criterion_segment = torch.nn.CrossEntropyLoss()
#         self.parameter_criterion = torch.nn.L1Loss()

#     def forward(self, points):
#         B,N,_ = points.shape
#         seg, feat = self.pct(points.permute(0,2,1))
#         feat = feat.view(B,N,-1).permute(0,2,1)
#         seg = seg.view(B,N,-1).permute(0,2,1)
#         # import pdb; pdb.set_trace()
#         x_max = torch.max(feat, dim=-1)[0]
#         x_mean = torch.mean(feat, dim=-1)
#         x, reg_feat = self.reg_feat(feat, x_max, x_mean)
#         reg_feat = reg_feat.mean(dim = -1)
#         pred_pose = self.pose_(reg_feat)
#         pred_betas = self.betas_(reg_feat)
#         pred_trans = self.trans_(reg_feat)
#         # pred_trans = self.trans(feat, x_max, x_mean)
#         # pred_pose = self.body_pose(feat, x_max, x_mean)
#         # pred_betas = self.betas(feat, x_max, x_mean)
#         # import pdb; pdb.set_trace()
#         pred_vertices = self.smpl(pred_pose, pred_betas) + pred_trans.unsqueeze(1)
#         # import pdb; pdb.set_trace()
#         ret_dict = {}
#         ret_dict.update({'pred_verts':pred_vertices, 'seg':seg, 'pred_pose':pred_pose, 'pred_betas':pred_betas})
#         return ret_dict

#     def vertices_loss(self, criterion_vertices, pred_vertices, gt_vertices, has_smpl = None, device = 'cuda'):
#         """
#         Compute per-vertex loss if vertex annotations are available.
#         """
#         if has_smpl is not None:
#             pred_vertices_with_shape = pred_vertices[has_smpl == 1]
#             gt_vertices_with_shape = gt_vertices[has_smpl == 1]
#         else:
#             pred_vertices_with_shape = pred_vertices
#             gt_vertices_with_shape = gt_vertices
#         if len(gt_vertices_with_shape) > 0:
#             return criterion_vertices(pred_vertices_with_shape, gt_vertices_with_shape)
#         else:
#             return torch.FloatTensor(1).fill_(0.).to(device) 

#     def keypoint_3d_loss(self, criterion_keypoints, pred_keypoints_3d, gt_keypoints_3d):
#         """
#         Compute 3D keypoint loss if 3D keypoint annotations are available.
#         """
#         if len(gt_keypoints_3d) > 0:
#             gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2
#             gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
#             pred_pelvis = (pred_keypoints_3d[:, 2,:] + pred_keypoints_3d[:, 3,:]) / 2
#             pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
#             # import pdb; pdb.set_trace()
#             # return (criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
#             return (pred_keypoints_3d - gt_keypoints_3d).norm(dim = -1).mean()
#         else:
#             return torch.FloatTensor(1).fill_(0.).cuda()

#     def seg_loss(self, pred_seg, seg_label):
#         # print(pred_seg.max(dim = 1)[0].mean(dim = 1))
#         return self.criterion_segment(pred_seg, seg_label)

#     def all_loss(self, ret_dict, sample):
        
#         smpl_pose = sample['smpl_pose']
#         global_trans = sample['global_trans']
#         betas = sample['betas']
#         pred_vert_full = ret_dict['pred_verts']
        
#         # import pdb; pdb.set_trace()
#         gt_verts = self.smpl(smpl_pose, betas)
#         # gt_verts = gt_verts + global_trans.unsqueeze(1)
#         gt_vertices_sub = self.mesh.downsample(gt_verts)
#         gt_vertices_sub2 = self.mesh.downsample(gt_vertices_sub, n1=1, n2=2)
#         gt_smpl_3d_joints = self.smpl.get_h36m_joints(gt_verts)
#         pred_3d_joints = self.smpl.get_h36m_joints(pred_vert_full)

#         vertices_loss_all = self.vertices_loss(self.criterion_vertices, pred_vert_full, gt_verts)
#         vertices_loss = vertices_loss_all
#         joints_loss = self.keypoint_3d_loss(self.criterion_keypoints, pred_3d_joints, gt_smpl_3d_joints)
#         # import pdb; pdb.set_trace()
#         seg_loss = self.seg_loss(ret_dict['seg'], sample['seg_label'].to(ret_dict['seg'].device))
#         parameter_loss = self.parameter_criterion(ret_dict['pred_betas'], sample['betas']) + \
#             self.parameter_criterion(ret_dict['pred_pose'], sample['smpl_pose'])
        
#         all_loss = vertices_loss + joints_loss + seg_loss + parameter_loss
#         return {'loss':all_loss, 'vertices_loss':vertices_loss, 'joint_loss':joints_loss, 'seg_loss':seg_loss}
