import os
import argparse
import torch
import smplx
import numpy as np
import open3d as o3d
import json
# from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import cv2
import pickle
import glob
from tqdm import tqdm

class lidar_Dataset(Dataset):
    def __init__(self, is_train = True,
                 return_torch:bool=True, 
                 fix_pts_num:bool=False,
                 augmentation:bool=False,
                 load_v2v = False, 
                 interval = 1):
        self.return_torch = return_torch
        self.fix_pts_num  = fix_pts_num
        self.point_num = 1024
        with open('smplx_models/smpl/SMPL_NEUTRAL.pkl', 'rb') as smpl_file:
            smpl_data = pickle.load(smpl_file, encoding='latin1')
        self.v_template = smpl_data['v_template']
        self.joint_24_regressor = torch.tensor(smpl_data['J_regressor'].todense()).float()
        self.default_trans = (self.joint_24_regressor @ self.v_template)[0]
        self.load_v2v = load_v2v
        self.interval = interval
        self.augmentation = augmentation
        self.is_train = is_train
        self.pred_mode = 'int'

    @staticmethod
    def nn_distance(pc1, pc2, l1smooth=False, delta=1.0, l1=False, is_cuda = True, return_diff = False):
        """
        Input:
            pc1: (B,N,C) torch tensor
            pc2: (B,M,C) torch tensor
            l1smooth: bool, whether to use l1smooth loss
            delta: scalar, the delta used in l1smooth loss
        Output:
            dist1: (B,N) torch float32 tensor
            idx1: (B,N) torch int64 tensor
            dist2: (B,M) torch float32 tensor
            idx2: (B,M) torch int64 tensor
        """
        pc1, pc2 = torch.tensor(pc1), torch.tensor(pc2)
        if len(pc1.shape) == 2:
            pc1, pc2 = pc1.unsqueeze(0), pc2.unsqueeze(0)
        N = pc1.shape[1]
        M = pc2.shape[1]
        
        if is_cuda:
            pc1, pc2 = pc1.cuda(), pc2.cuda()
        else:
            pc1, pc2 = pc1.cpu(), pc2.cpu()

        pc1_expand_tile = pc1.unsqueeze(2).repeat(1,1,M,1)
        pc2_expand_tile = pc2.unsqueeze(1).repeat(1,N,1,1)
        
        pc_diff = pc2_expand_tile - pc1_expand_tile
        pc_dist = torch.norm(pc_diff, dim=-1, p = 2) # (B,N,M)
        dist1, idx1 = torch.min(pc_dist, dim=2) # (B,N)
        dist2, idx2 = torch.min(pc_dist, dim=1) # (B,M)
        if return_diff:
            return dist1, idx1, dist2, idx2, pc_dist, pc_diff
        else:
            return dist1, idx1, dist2, idx2, pc_dist

    def augment(self, human_points, smpl_joints, global_trans, smpl_verts = None):
        #[Np, 3] [Nkp, 3] [1,3]
        new_size = torch.rand(1) * 40 + 80
        angle = (torch.rand(1) * 90 - 45) /180*np.pi
        trans = torch.rand(1) * 0.4 - 0.2
        resize_scale = new_size / 100
        rel_pcd = human_points - global_trans
        rel_joint = smpl_joints - global_trans
        rel_pcd *= resize_scale
        rel_joint *= resize_scale

        if angle != 0:
            r_p_copy = rel_pcd.clone()
            rel_pcd[:,0] = r_p_copy[:,0]*np.cos(angle) - r_p_copy[:,1]*np.sin(angle)
            rel_pcd[:,1] = r_p_copy[:,0]*np.sin(angle) + r_p_copy[:,1]*np.cos(angle)
            r_j_copy = rel_joint.clone()
            rel_joint[:,0] = r_j_copy[:,0]*np.cos(angle) - r_j_copy[:,1]*np.sin(angle)
            rel_joint[:,1] = r_j_copy[:,0]*np.sin(angle) + r_j_copy[:,1]*np.cos(angle)

        if smpl_verts is not None:
            rel_v = smpl_verts - global_trans
            r_v_copy = rel_v.clone()
            rel_v[:,0] = r_v_copy[:,0]*np.cos(angle) - r_v_copy[:,1]*np.sin(angle)
            rel_v[:,1] = r_v_copy[:,0]*np.sin(angle) + r_v_copy[:,1]*np.cos(angle)
            smpl_verts = rel_v + global_trans
            smpl_verts += trans

        human_points = rel_pcd + global_trans
        smpl_joints = rel_joint + global_trans
        human_points += trans
        smpl_joints += trans
        global_trans += trans
        return human_points, smpl_joints, global_trans, smpl_verts

    def __getitem__(self, ind):
        sam_data = self.valid_hkps[ind * self.interval]
        smpl_verts = sam_data['smpl_verts']
        smpl_joints = sam_data['smpl_joints']
        mesh_dict = sam_data['mesh_dict']
        smpl_verts  =  torch.tensor(smpl_verts).float()
        if 'human_points' not in sam_data.keys():
            pcd_file = sam_data['pcd_file']
            pcd = np.array(o3d.io.read_point_cloud(pcd_file).points)
            root_min, root_max = np.min(smpl_joints, axis = 0) - 0.2, np.max(smpl_joints, axis = 0) + 0.2
            root = (root_min + root_max ) / 2
            valid_p_ind = (pcd[:,0] > root_min[0]) & \
                (pcd[:,1] > root_min[1]) & \
                (pcd[:,2] > root_min[2]) & \
                (pcd[:,0] <= root_max[0]) & \
                (pcd[:,1] <= root_max[1]) & \
                (pcd[:,2] <= root_max[2])
            human_points = torch.tensor(pcd[valid_p_ind,:]).unsqueeze(0)
        else:
            human_points = torch.tensor(sam_data['human_points']).unsqueeze(0)
            root = torch.mean(human_points.squeeze(), dim = 0)

        smpl_joints, root = torch.tensor(smpl_joints).unsqueeze(0), root
        pc_dist = torch.tensor(sam_data['pc_dist']) # (N_p,N_k)
        # print(human_points.shape, pc_dist.shape)
        min_dist, idx1 = torch.min(pc_dist, dim=1)
        vis_label = min_dist < 0.25 #[N]
        seg_label = (torch.ones([human_points.shape[1]]) * len(self.JOINTS_IDX)).long() #[N_p]
        seg_label[vis_label] = idx1.squeeze(0)[vis_label].cpu()

        vis_label_kpts = torch.min(pc_dist, dim = 0)[0] < 0.25
        smpl_joints = smpl_joints.squeeze()
        human_points = human_points.squeeze()
        max_, min_ = human_points.max(dim = 0)[0], human_points.min(dim = 0)[0]
        root = (max_ + min_) / 2

        if self.fix_pts_num:
            now_pt_num = int(human_points.shape[0])
            if now_pt_num > self.point_num:
                choice_indx = np.random.randint(0, now_pt_num, size = [self.point_num])
                human_points = human_points[choice_indx,:]
                seg_label = seg_label[choice_indx]
            else:
                choice_indx = np.random.randint(0, now_pt_num, size = [self.point_num - now_pt_num])
                human_points = torch.cat([human_points, human_points[choice_indx]], dim = 0)
                seg_label = torch.cat([seg_label, seg_label[choice_indx]], dim = 0)
        # import pdb; pdb.set_trace()
        if self.is_train and self.augmentation:
            human_points, smpl_joints, root, smpl_verts = self.augment(human_points, smpl_joints, root, smpl_verts)
        smpl_joint_24 = self.joint_24_regressor @ smpl_verts.float()
        gt_trans = smpl_joint_24[0] - self.default_trans
        sample = {
            'global_trans' : root.float(),
            'gt_trans' : (gt_trans - root).float(),
            'smpl_joints'  : smpl_joints.float(),   
            'smpl_verts'   : smpl_verts.float(),   
            'smpl_verts_local'   : (smpl_verts - root).float(),#.to(self.device), 
            'human_points' : human_points.float(),
            'human_points_local': (human_points - root).float(),#.to(self.device),
            'smpl_joints_local' : (smpl_joints - root).float(),#.to(self.device),
            'smpl_joints24_local' : (smpl_joint_24 - root).float(),#.to(self.device),
            'vis_label' : vis_label_kpts.float(),
            'seg_label' : seg_label.long(),
            'smpl_pose': torch.tensor(np.concatenate([mesh_dict['global_orient'], mesh_dict['body_pose']], axis = -1)).float(),
            'betas': torch.tensor(mesh_dict['betas']).float(),
            'has_3d_joints': 1,
            'has_smpl': 1,
            'num_points': now_pt_num
        }
        if self.load_v2v:
            location_dict = sam_data['location_dict']
            # frame, time, id = location_dict['scene'], location_dict['time'], location_dict['id']
            if self.pred_mode == 'int':
                frame, time, id = location_dict['scene'], location_dict['time'], location_dict['id']
            elif self.pred_mode =='str':
                frame, time, id = str(location_dict['scene']), str(location_dict['time']), str(location_dict['id'])
            pose_pred = self.v2v_pred[frame][time][id]
            pose_pred_local = (torch.tensor(pose_pred) - root).to(self.device)
            sample.update({'pose_pred': pose_pred, 'pose_pred_local': pose_pred_local, 'location_dict' : sam_data['location_dict']})
        if self.return_torch:
            for k, v in sample.items():
                if k == 'mask':
                    sample[k] = torch.tensor(v)#.to(self.device)
                elif type(v) != str and type(v) != torch.Tensor and type(v) != dict:
                    sample[k] = torch.tensor(v).float()#.to(self.device)
                elif type(v) == torch.Tensor and k != 'seg_label':
                    sample[k] = v.float()#.to(self.device)
        return sample
    
    def __len__(self):
        return len(self.valid_hkps) // self.interval