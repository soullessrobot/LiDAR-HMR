"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

"""

import torch
import numpy as np
J24_TO_J14 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18]
H36M_J17_TO_J14 = [3, 2, 1, 4, 5, 6, 16, 15, 14, 11, 12, 13, 8, 10]
H36M_J17_TO_ITOP = [10, 8, 0, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3]

H36M_J17_NAME = ( 'Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head',
                  'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
import os
file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(file_path)
SMPL_JOINT_INDEX = np.load(os.path.join(dir_path, 'joint_index.npy'))
SMPL_BONE_INDEX = np.load(os.path.join(dir_path, 'bone_index.npy'))

class Graphormer_Body_Network(torch.nn.Module):
    '''
    End-to-end Graphormer network for human pose and mesh reconstruction from a single image.
    '''
    def __init__(self, trans_encoder, input_dim = 256):
        super(Graphormer_Body_Network, self).__init__()
        self.trans_encoder = trans_encoder
        self.upsampling = torch.nn.Linear(431, 1723)
        self.upsampling2 = torch.nn.Linear(1723, 6890)
        self.grid_feat_dim = torch.nn.Sequential(*[torch.nn.Linear(input_dim, 64), torch.nn.Linear(64, 256)])
        # self.query_feat_dim = torch.nn.Sequential(*[torch.nn.Linear(3, 16), torch.nn.Linear(16, 32)])

    def forward(self, features, smpl, mesh_sampler, meta_masks=None, is_train=False):
        batch_size = features.shape[0]
        # Generate T-pose template mesh
        template_pose = torch.zeros((1,72))
        template_pose[:,0] = 3.1416 # Rectify "upside down" reference mesh in global coord
        template_pose = template_pose.cuda()
        template_betas = torch.zeros((1,10)).cuda()
        template_vertices = smpl(template_pose, template_betas)

        # template mesh simplification
        template_vertices_sub = mesh_sampler.downsample(template_vertices)
        template_vertices_sub2 = mesh_sampler.downsample(template_vertices_sub, n1=1, n2=2)

        # template mesh-to-joint regression 
        template_3d_joints = smpl.get_h36m_joints(template_vertices)
        template_pelvis = template_3d_joints[:,H36M_J17_NAME.index('Pelvis'),:]
        template_3d_joints = template_3d_joints[:,H36M_J17_TO_J14,:]
        num_joints = template_3d_joints.shape[1]

        # normalize
        template_3d_joints = template_3d_joints - template_pelvis[:, None, :]
        template_vertices_sub2 = template_vertices_sub2 - template_pelvis[:, None, :]

        # concatinate template joints and template vertices, and then duplicate to batch size
        ref_vertices = torch.cat([template_3d_joints, template_vertices_sub2],dim=1)
        ref_vertices = ref_vertices.expand(batch_size, -1, -1)

        # process grid features
        # concatinate image feat and template mesh to form the joint/vertex queries
        new_feat = self.grid_feat_dim(features[:,:,3:])
        mean_feat = new_feat.mean(dim = 1, keepdim = True)

        # import pdb; pdb.set_trace()
        ref_feat = mean_feat.repeat(1, ref_vertices.shape[1], 1)
        new_feat = torch.cat([ref_feat, new_feat], dim = 1)
        
        features = torch.cat([ref_vertices, features[:,:,:3]], dim=1)
        features = torch.cat([features, new_feat], dim=2)
        # import pdb; pdb.set_trace()
        # prepare input tokens including joint/vertex queries and grid features
        
        # forward pass
        features = self.trans_encoder(features)
        
        pred_3d_joints = features[:,:num_joints,:]
        pred_vertices_sub2 = features[:,num_joints:431+num_joints,:]
        # import pdb; pdb.set_trace()
        # learn camera parameters
        temp_transpose = pred_vertices_sub2.transpose(1,2)
        
        pred_vertices_sub = self.upsampling(temp_transpose)
        pred_vertices_full = self.upsampling2(pred_vertices_sub)
        pred_vertices_sub = pred_vertices_sub.transpose(1,2)
        pred_vertices_full = pred_vertices_full.transpose(1,2)

        return pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices_full

class BERT_SMPL_Net(torch.nn.Module):
    '''
    End-to-end Graphormer network for human pose and mesh reconstruction from a single image.
    '''
    def __init__(self, trans_encoder):
        super(BERT_SMPL_Net, self).__init__()
        self.trans_encoder = trans_encoder
        self.embedd_layer = torch.nn.Sequential(*[torch.nn.Linear(446, 128)])
        self.global_orient_layer = torch.nn.Linear(128, 3)
        self.pose_layer = torch.nn.Linear(128, 32)
        self.beta_layer = torch.nn.Linear(128, 10)
        self.grid_feat_dim = torch.nn.Sequential(*[torch.nn.Linear(256, 64), torch.nn.Linear(64, 32)])

    def forward(self, features, smpl):
        batch_size = features.shape[0]

        # process grid features
        # concatinate image feat and template mesh to form the joint/vertex queries
        new_feat = self.grid_feat_dim(features[:,:,3:])
        features = torch.cat([features[:,:,:3], new_feat], dim=2)

        # prepare input tokens including joint/vertex queries and grid features
        
        # forward pass
        features = self.trans_encoder(features).squeeze()
        features = self.embedd_layer(features)
        global_orient = self.global_orient_layer(features)
        pose_embedding = self.pose_layer(features)
        betas = self.beta_layer(features)

        return global_orient, pose_embedding, betas

class BERT_Joint_Net(torch.nn.Module):
    '''
    End-to-end Graphormer network for human pose and mesh reconstruction from a single image.
    '''
    def __init__(self, trans_encoder):
        super(BERT_Joint_Net, self).__init__()
        self.trans_encoder = trans_encoder
        self.grid_feat_dim = torch.nn.Sequential(*[torch.nn.Linear(256, 64), torch.nn.Linear(64, 32)])

    def forward(self, features, seg = None):
        batch_size = features.shape[0]

        # process grid features
        # concatinate image feat and template mesh to form the joint/vertex queries
        new_feat = self.grid_feat_dim(features[:,:,3:])
        features = torch.cat([features[:,:,:3], new_feat], dim=2)

        # prepare input tokens including joint/vertex queries and grid features
        
        # forward pass
        if seg is not None:
            for ten in self.trans_encoder:
                features = ten(features, seg)
        else:
            features = self.trans_encoder(features)
        pose = features[:,:15,:]

        return pose