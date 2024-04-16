import torch
import smplx
import numpy as np
# import __init_path
from .smpl import SMPL
from .graph_utils import build_coarse_graphs, sparse_python_to_torch
from .meshnet import get_model
from .loss import get_loss
# from models.pose2mesh.keypoints_config import JOINT_REGRESSOR_H36M_correct, H36M_J17_TO_J15
import torch.nn as nn
from models.pose2mesh.keypoints_config import JOINT_REGRESSOR_H36M_correct, H36M_J17_TO_J15, flip_pairs_15, skeleton_15
from models.smpl_hybrik.SMPL import SMPL_layer

class pointcloud2mesh_net(nn.Module):
    # Pose2Mesh: https://arxiv.org/abs/2008.09047
    def __init__(self, **kwargs):
        super().__init__()
        self.mesh_model = SMPL()
        joint_num = 15
        self.flip_pairs = flip_pairs_15
        self.skeleton = skeleton_15
        self.graph_Adj, self.graph_L, self.graph_perm, self.graph_perm_reverse = \
                    build_coarse_graphs(self.mesh_model.face, joint_num, self.skeleton, self.flip_pairs, levels=9)
        # g_A = [sparse_python_to_torch(self.graph_Adj) for ga in self.graph_Adj]
        self.meshnet = get_model(3,3,self.graph_L).cuda()
        self.criterion = get_loss(faces=self.mesh_model.face)
        self.normal_weight = 0.1
        self.joint_weight = 1.0
        self.edge_weight = 20
        self.joint_regressor = torch.tensor(np.load(JOINT_REGRESSOR_H36M_correct))
        self.valid_kpts_indx = H36M_J17_TO_J15

    def forward(self, pose_input):
        mesh_out = self.meshnet(pose_input)[:,self.graph_perm_reverse[:self.mesh_model.face.max() + 1],:]
        j_r = self.joint_regressor.to(mesh_out.device).float()
        pred_pose = (j_r @ mesh_out)[:,self.valid_kpts_indx]
        return {'mesh_out':mesh_out, 'pose':pred_pose}

    def all_loss(self, ret_dict, sample, val_mesh = None):
        pred_mesh = ret_dict['mesh_out']
        gt_mesh = sample['smpl_verts_local'].to(pred_mesh.device)
        j_r = self.joint_regressor.to(pred_mesh.device).float()
        pred_pose = (j_r @ pred_mesh)[:,self.valid_kpts_indx]
        gt_reg3dpose = (j_r @ gt_mesh)[:,self.valid_kpts_indx]
        loss1, loss2, loss4 = self.criterion[0](pred_mesh, gt_mesh, val_mesh),  \
                                                self.normal_weight * self.criterion[1](pred_mesh, gt_mesh), \
                                                self.joint_weight * self.criterion[3](pred_pose,  gt_reg3dpose, val_mesh)
        loss3 = self.edge_weight * self.criterion[2](pred_mesh, gt_mesh)
        loss = loss1 + loss2 + loss4
        return {'loss': loss, 'loss_joint':loss4, 'loss_vert':loss1 + loss2, 'edge_loss':loss3}

    def get_final_mesh_loss(self, pred_mesh, gt_mesh, val_mesh = None):
        j_r = self.joint_regressor.to(pred_mesh.device).float()
        pred_pose = (j_r @ pred_mesh)[:,self.valid_kpts_indx]
        gt_reg3dpose = (j_r @ gt_mesh)[:,self.valid_kpts_indx]
        # 
        loss1, loss2, loss4 = self.criterion[0](pred_mesh, gt_mesh, val_mesh),  \
                                                self.normal_weight * self.criterion[1](pred_mesh, gt_mesh), \
                                                self.joint_weight * self.criterion[3](pred_pose,  gt_reg3dpose, val_mesh)
        loss3 = self.criterion[2](pred_mesh, gt_mesh) * self.edge_weight
        loss5 = 0
        loss = loss1 + loss4 + loss2 
        return {'loss': loss, 'loss_joint':loss4, 'loss_vert':loss1 + loss2, 'edge_loss': loss3 + loss5}


class p2m_svd(nn.Module):
    def __init__(self, train_p2m = False):
        super().__init__()
        self.p2m = pointcloud2mesh_net()
        j36m = np.load('models/data/J_regressor_h36m_correct.npy')
        self.smpl_ = SMPL_layer('smplx_models/smpl/SMPL_NEUTRAL.pkl',num_joints = 24, h36m_jregressor=j36m)
        self.beta_regressor = nn.Sequential(
            nn.Linear(23+23+23, 96),
            nn.Linear(96, 24),
            nn.Linear(24, 10)
        )
        self.train_p2m = train_p2m
        if not self.train_p2m:
            for param in self.p2m.parameters():
                param.requires_grad = False
    
    def forward(self, pose_input):
        ret_dict = self.p2m(pose_input)
        mesh_ = ret_dict['mesh_out'].detach()
        batch_size = mesh_.shape[0]
        pose_skeleton = self.smpl_.J_regressor @ mesh_
        rel_ps = (pose_skeleton[:,1:] - pose_skeleton[:, self.smpl_.parents[1:]].clone()).norm(dim = -1)
        template_skeleton = self.smpl_.J_regressor @ self.smpl_.v_template
        rel_ts = (template_skeleton[1:] - template_skeleton[self.smpl_.parents[1:]].clone()).norm(dim = -1)
        rel_ts = rel_ts.unsqueeze(0).repeat(batch_size, 1).to(rel_ps.device) #[B,23,3]
        in_ds = rel_ps - rel_ts
        
        inp_feat = torch.cat([rel_ps, rel_ts, in_ds], dim = -1)
        betas = self.beta_regressor(inp_feat)
        output = self.smpl_.meshik(
            pred_verts=mesh_,
            betas=betas
        )
        verts = output.vertices
        vert_shift = [verts - mesh_]
        ret_dict.update({'mesh_refine':verts, 'vert_shift':vert_shift, 'pose_theta': output.theta, 'pose_beta':betas, 'trans': output.trans})
        return ret_dict

    def all_loss(self, ret_dict, sample):
        before_mesh = ret_dict['mesh_out']
        final_mesh = ret_dict['mesh_refine']
        gt_mesh = sample['smpl_verts_local']
        before_sk = self.smpl_.J_regressor @ before_mesh
        after_sk = self.smpl_.J_regressor @ final_mesh
        before_len = (before_sk[:,1:] - before_sk[:, self.smpl_.parents[1:]].clone()).norm(dim = -1)
        after_len = (after_sk[:,1:] - after_sk[:, self.smpl_.parents[1:]].clone()).norm(dim = -1)
        loss_sk = (before_len - after_len).abs().mean()
        
        if not self.train_p2m:
            loss_dict_pose = self.p2m.get_final_mesh_loss(final_mesh, gt_mesh)
        else:
            loss_dict_pose = self.p2m.all_loss(ret_dict, sample)

        loss_dict_pose['loss_refine'] = loss_sk
        if 'vert_shift' in ret_dict.keys():
            consist_loss = 0
            for vs in ret_dict['vert_shift']:
                consist_loss += vs.norm(dim = -1).mean()
            loss_dict_pose['loss_refine'] += consist_loss
            # loss_dict_pose['loss_shift'] = consist_loss
            # loss_dict_pose['loss'] += consist_loss
        return loss_dict_pose

# import pdb; pdb.set_trace()
# loss1, loss2, loss4 = criterion[0](pred_mesh, gt_mesh, val_mesh),  \
#                                                 normal_weight * criterion[1](pred_mesh, gt_mesh), \
#                                                 joint_weight * criterion[3](pred_pose,  gt_reg3dpose, val_reg3dpose)
# loss3 = 0
# loss = loss1 + loss2 + loss3 + loss4

# import pdb; pdb.set_trace()