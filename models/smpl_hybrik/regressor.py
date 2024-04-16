import torch
import smplx
import numpy as np
# from models.pose2mesh.keypoints_config import JOINT_REGRESSOR_H36M_correct, H36M_J17_TO_J15
import torch.nn as nn
from models.pose2mesh.keypoints_config import JOINT_REGRESSOR_H36M_correct, H36M_J17_TO_J15, flip_pairs_15, skeleton_15
from .SMPL import SMPL_layer
from ..pose2mesh.loss import get_loss

class Hybrik(nn.Module):
    def __init__(self):
        super().__init__()
        self.decshape = nn.Sequential(
            nn.Linear(24 * 3, 1024),
            nn.Linear(1024, 10)
        )
        self.decphi = nn.Sequential(
            nn.Linear(24 * 3, 1024),
            nn.Linear(1024, 23 * 2)
        )
        self.decleaf = nn.Sequential(
            nn.Linear(24 * 3, 1024),
            nn.Linear(1024, 5 * 4)
        )
        self.detrans = nn.Sequential(
            nn.Linear(24 * 3, 1024),
            nn.Linear(1024, 3)
        )
        j36m = np.load('./models/data/J_regressor_h36m_correct.npy')
        self.smpl_ = SMPL_layer('./smplx_models/smpl/SMPL_NEUTRAL.pkl',num_joints = 24, h36m_jregressor=j36m)
        self.smpl_dtype = torch.float32
        
        self.criterion = get_loss(faces=self.smpl_.faces.astype(np.int32))
        self.joint_regressor = torch.tensor(np.load(JOINT_REGRESSOR_H36M_correct))
        self.valid_kpts_indx = H36M_J17_TO_J15
        self.normal_weight = 0.1 #0.1
        self.joint_weight = 1.0
        self.edge_weight = 20

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B,-1)
        pred_pose = x.view(B, 24, 3)
        
        pred_shape = self.decshape(x)
        pred_phi = self.decphi(x).view(B, 23, 2)
        pred_leaf = self.decleaf(x).view(B, 5, 4)
        pred_trans = self.detrans(x).view(B, 3)
        output = self.smpl_.hybrik(
            pose_skeleton=pred_pose.type(self.smpl_dtype) * 2,
            betas=pred_shape.type(self.smpl_dtype),
            phis=pred_phi.type(self.smpl_dtype),
            leaf_thetas=pred_leaf.type(self.smpl_dtype),
            global_orient=None,
            return_verts=True
        )
        pred_vertices = output.vertices.float() + pred_trans[:,None,:]
        return {'pose_skeleton': pred_pose, 'mesh_out':pred_vertices}
    
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
