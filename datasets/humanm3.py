import os
import argparse

# import pickle
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
from .LiDAR_dataset import lidar_Dataset

keypoints_name = {1:'nose',
    5:'lshoulder',
    6:'lelbow',
    7:'lwrist',
    8:'lhip',
    9:'lknee',
    10:'lankle',
    13:'rshoulder',
    14:'relbow',
    15:'rwrist',
    16:'rhip',
    17:'rknee',
    18:'rankle',
    19:'head',
    20:'head'
    }

keypoints_index = {
    'nose':0,
    'lshoulder':1,
    'lelbow':2,
    'lwrist':3,
    'lhip':4,
    'lknee':5,
    'lankle':6,
    'rshoulder':7,
    'relbow':8,
    'rwrist':9,
    'rhip':10,
    'rknee':11,
    'rankle':12,
    'head':13
}

def fix_points_num(points: np.array, num_points: int):
    """
    downsamples the points using voxel and uniform downsampling, 
    and either repeats or randomly selects points to reach the desired number.
    
    Args:
      points (np.array): a numpy array containing 3D points.
      num_points (int): the desired number of points 
    
    Returns:
      a numpy array `(num_points, 3)`
    """
    # print(points.shape)
    if len(points) == 0:
        return np.zeros((num_points, 3))
    points = points[~np.isnan(points).any(axis=-1)]

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc = pc.voxel_down_sample(voxel_size=0.05)
    ratio = int(len(pc.points) / num_points + 0.05)
    if ratio > 1:
        pc = pc.uniform_down_sample(ratio)

    points = np.asarray(pc.points)
    origin_num_points = points.shape[0]

    if origin_num_points < num_points:
        num_whole_repeat = num_points // origin_num_points
        res = points.repeat(num_whole_repeat, axis=0)
        num_remain = num_points % origin_num_points
        res = np.vstack((res, res[:num_remain]))
    else:
        res = points[np.random.choice(origin_num_points, num_points)]
    return res

class HumanM3_Dataset(lidar_Dataset):
    def __init__(self, root_folder = '/Extra/fanbohao/posedataset/PointC/humanm3/',
                 dataset_path = 'save_data/humanm3/', is_train = True,
                 device='cpu',
                 return_torch:bool=True, 
                 fix_pts_num:bool=False,
                 augmentation:bool=False, load_v2v = False, interval = 1):
        super().__init__(is_train = is_train,
                 return_torch=return_torch, 
                 fix_pts_num=fix_pts_num,
                 augmentation=augmentation,
                 load_v2v = load_v2v, 
                 interval = interval)
        self.device       = device
        self.return_torch = return_torch
        self.fix_pts_num  = fix_pts_num
        self.point_num = 1024
        self.root_folder = root_folder
        split = 'train' if is_train else 'test'
        data_file = 'train.pkl' if is_train else 'test.pkl'
        self.valid_hkps = []
        os.makedirs(dataset_path, exist_ok=True)
        pkl_file = os.path.join(dataset_path, data_file)
        self.JOINTS_IDX = [0, 1, 2, 4, 5, 7, 8, 12, 15, 16, 17, 18, 19, 20, 21]

        self.load_v2v = load_v2v
        if load_v2v:
            with open(os.path.join('pose_results/lpformer/humanm3', split + '.json'), 'r') as f:
                self.v2v_pred = json.load(f)

        if not os.path.exists(pkl_file):
            human_model = smplx.create('./smplx_models/', model_type = 'smpl',
                                    gender='neutral', 
                                    use_face_contour=False,
                                    ext="npz").cuda()
            json_files = glob.glob(os.path.join(root_folder, split, '*', 'smpl_calib', '*.json'))
            for jf in tqdm(json_files):
                with open(jf, 'r') as f:
                    dict_ = json.load(f)
                names = jf.split('/')
                split_, scene_, time_ = names[-4], names[-3], names[-1].replace('.json', '')
                pcd_file = jf.replace('json', 'pcd').replace('smpl_calib', 'pointcloud').replace(time_, str(int(time_)).zfill(6))

                pcd = np.array(o3d.io.read_point_cloud(pcd_file).points)

                for indx, key_ in enumerate(dict_.keys()):
                    mesh_dict = dict_[key_]
                    body_pose, transl, betas, global_orient = mesh_dict['body_pose'], mesh_dict['transl'], mesh_dict['betas'], mesh_dict['global_orient']
                    global_orient = torch.tensor(global_orient).float().unsqueeze(0)
                    body_pose = torch.tensor(body_pose).float().unsqueeze(0)
                    body_pose = torch.cat([body_pose, torch.zeros(1,6)], dim = 1)
                    transl = torch.tensor(transl).float().unsqueeze(0)
                    betas=torch.tensor(betas).unsqueeze(0)
                    smpl_md = human_model(betas = betas.cuda(), 
                                            return_verts=True, 
                                            body_pose=body_pose.cuda(),
                                            global_orient=global_orient.cuda(),
                                            transl=transl.cuda())
                    
                    smpl_joints = smpl_md.joints.squeeze()[self.JOINTS_IDX].cpu().numpy()
                    root_min, root_max = np.min(smpl_joints, axis = 0) - 0.2, np.max(smpl_joints, axis = 0) + 0.2
                    valid_p_ind = (pcd[:,0] > root_min[0]) & \
                        (pcd[:,1] > root_min[1]) & \
                        (pcd[:,2] > root_min[2]) & \
                        (pcd[:,0] <= root_max[0]) & \
                        (pcd[:,1] <= root_max[1]) & \
                        (pcd[:,2] <= root_max[2])
                    root = (root_min + root_max ) / 2
                    human_points = pcd[valid_p_ind,:]
                    if human_points.shape[0] != 0:
                        location_dict = {'split':split_, 'scene':scene_, 'time':time_, 'id':key_}
                        self.valid_hkps.append({'pcd_file':pcd_file, \
                                                'mesh_dict':mesh_dict, 'smpl_joints':smpl_joints, \
                                                'smpl_verts':smpl_md.vertices.squeeze().cpu().numpy(),
                                                'location_dict':location_dict, 'human_points':human_points})
            with open(pkl_file, 'wb') as f:
                pickle.dump(self.valid_hkps, f)
        else:
            with open(pkl_file, 'rb') as f:
                self.valid_hkps = pickle.load(f)
            for hkps in self.valid_hkps:
                hkps['mesh_dict']['body_pose'] = np.concatenate([hkps['mesh_dict']['body_pose'], np.zeros(6)], axis = -1)
                
        self.interval = interval
        self.augmentation = augmentation
        self.is_train = is_train
    
# class HumanM3_Dataset(Dataset):
#     def __init__(self, root_folder = '/Extra/fanbohao/posedataset/PointC/humanm3/',
#                  dataset_path = 'save_data/humanm3/', is_train = True,
#                  device='cpu',
#                  return_torch:bool=True, 
#                  fix_pts_num:bool=False,
#                  print_info:bool=True,
#                  augmentation:bool=False, load_v2v = False, interval = 1):
#         self.device       = device
#         self.return_torch = return_torch
#         self.print_info   = print_info
#         self.fix_pts_num  = fix_pts_num
#         self.point_num = 1024
#         # self.num_joints = 14
#         self.root_folder = root_folder
#         split = 'train' if is_train else 'test'
#         data_file = 'train.pkl' if is_train else 'test.pkl'
#         self.valid_hkps = []
#         os.makedirs(dataset_path, exist_ok=True)
#         pkl_file = os.path.join(dataset_path, data_file)
#         self.JOINTS_IDX = [0, 1, 2, 4, 5, 7, 8, 12, 15, 16, 17, 18, 19, 20, 21]
#         with open('smplx_models/smpl/SMPL_NEUTRAL.pkl', 'rb') as smpl_file:
#             smpl_data = pickle.load(smpl_file, encoding='latin1')
#         self.v_template = smpl_data['v_template']
#         self.joint_24_regressor = torch.tensor(smpl_data['J_regressor'].todense()).float()
#         self.default_trans = self.joint_24_regressor @ self.v_template
#         # import pdb; pdb.set_trace()
#         self.load_v2v = load_v2v
#         if load_v2v:
#             # self.v2v_pred = []
#             with open(os.path.join('humanm3_v2v_results', split + '.json'), 'r') as f:
#                 self.v2v_pred = json.load(f)

#         if not os.path.exists(pkl_file):
#             human_model = smplx.create('./smplx_models/', model_type = 'smpl',
#                                     gender='neutral', 
#                                     use_face_contour=False,
#                                     ext="npz").cuda()
#             json_files = glob.glob(os.path.join(root_folder, split, '*', 'smpl_calib', '*.json'))
#             for jf in tqdm(json_files):
#                 with open(jf, 'r') as f:
#                     dict_ = json.load(f)
#                 names = jf.split('/')
#                 split_, scene_, time_ = names[-4], names[-3], names[-1].replace('.json', '')
#                 pcd_file = jf.replace('json', 'pcd').replace('smpl_calib', 'pointcloud').replace(time_, str(int(time_)).zfill(6))

#                 pcd = np.array(o3d.io.read_point_cloud(pcd_file).points)

#                 for indx, key_ in enumerate(dict_.keys()):
#                     mesh_dict = dict_[key_]
#                     body_pose, transl, betas, global_orient = mesh_dict['body_pose'], mesh_dict['transl'], mesh_dict['betas'], mesh_dict['global_orient']
#                     global_orient = torch.tensor(global_orient).float().unsqueeze(0)
#                     body_pose = torch.tensor(body_pose).float().unsqueeze(0)
#                     body_pose = torch.cat([body_pose, torch.zeros(1,6)], dim = 1)
#                     transl = torch.tensor(transl).float().unsqueeze(0)
#                     betas=torch.tensor(betas).unsqueeze(0)
#                     # import pdb; pdb.set_trace()
#                     smpl_md = human_model(betas = betas.cuda(), 
#                                             return_verts=True, 
#                                             body_pose=body_pose.cuda(),
#                                             global_orient=global_orient.cuda(),
#                                             transl=transl.cuda())
#                     # import pdb; pdb.set_trace()
#                     smpl_joints = smpl_md.joints.squeeze()[self.JOINTS_IDX].cpu().numpy()
#                     root_min, root_max = np.min(smpl_joints, axis = 0) - 0.2, np.max(smpl_joints, axis = 0) + 0.2
#                     valid_p_ind = (pcd[:,0] > root_min[0]) & \
#                         (pcd[:,1] > root_min[1]) & \
#                         (pcd[:,2] > root_min[2]) & \
#                         (pcd[:,0] <= root_max[0]) & \
#                         (pcd[:,1] <= root_max[1]) & \
#                         (pcd[:,2] <= root_max[2])
#                     root = (root_min + root_max ) / 2
#                     human_points = pcd[valid_p_ind,:]
#                     if human_points.shape[0] == 0:
#                         print('!!???')
#                         print(pcd_file)
#                     else:
#                         location_dict = {'split':split_, 'scene':scene_, 'time':time_, 'id':key_}
#                         self.valid_hkps.append({'pcd_file':pcd_file, \
#                                                 'mesh_dict':mesh_dict, 'smpl_joints':smpl_joints, \
#                                                 'smpl_verts':smpl_md.vertices.squeeze().cpu().numpy(),
#                                                 'location_dict':location_dict, 'human_points':human_points})
#             with open(pkl_file, 'wb') as f:
#                 pickle.dump(self.valid_hkps, f)
#         else:
#             with open(pkl_file, 'rb') as f:
#                 self.valid_hkps = pickle.load(f)        
#         # import pdb; pdb.set_trace()
#         self.interval = interval
#         self.augmentation = augmentation
#         self.is_train = is_train

#     @staticmethod
#     def nn_distance(pc1, pc2, l1smooth=False, delta=1.0, l1=False, is_cuda = True, return_diff = False):
#         """
#         Input:
#             pc1: (B,N,C) torch tensor
#             pc2: (B,M,C) torch tensor
#             l1smooth: bool, whether to use l1smooth loss
#             delta: scalar, the delta used in l1smooth loss
#         Output:
#             dist1: (B,N) torch float32 tensor
#             idx1: (B,N) torch int64 tensor
#             dist2: (B,M) torch float32 tensor
#             idx2: (B,M) torch int64 tensor
#         """
#         # pc1, pc2 = torch.tensor(pc1), torch.tensor(pc2)
#         if len(pc1.shape) == 2:
#             pc1, pc2 = pc1.unsqueeze(0), pc2.unsqueeze(0)
#         N = pc1.shape[1]
#         M = pc2.shape[1]
#         #print(pc1.device)
        
#         if is_cuda:
#             pc1, pc2 = pc1.cuda(), pc2.cuda()
#         else:
#             pc1, pc2 = pc1.cpu(), pc2.cpu()
#         #print(pc1.device)
#         pc1_expand_tile = pc1.unsqueeze(2).repeat(1,1,M,1)
#         pc2_expand_tile = pc2.unsqueeze(1).repeat(1,N,1,1)
        
#         pc_diff = pc2_expand_tile - pc1_expand_tile
#         #print(pc_diff.device)
#         pc_dist = torch.norm(pc_diff, dim=-1, p = 2) # (B,N,M)
#         dist1, idx1 = torch.min(pc_dist, dim=2) # (B,N)
#         dist2, idx2 = torch.min(pc_dist, dim=1) # (B,M)
#         if return_diff:
#             return dist1, idx1, dist2, idx2, pc_dist, pc_diff
#         else:
#             return dist1, idx1, dist2, idx2, pc_dist

#     def augment(self, human_points, smpl_joints, global_trans, smpl_verts = None):
#         #[Np, 3] [Nkp, 3] [1,3]
#         new_size = torch.rand(1) * 40 + 80
#         angle = (torch.rand(1) * 90 - 45) /180*np.pi
#         trans = torch.rand(1) * 0.4 - 0.2
#         resize_scale = new_size / 100
#         rel_pcd = human_points - global_trans
#         rel_joint = smpl_joints - global_trans
#         rel_pcd *= resize_scale
#         rel_joint *= resize_scale

#         if angle != 0:
#             r_p_copy = rel_pcd.clone()
#             rel_pcd[:,0] = r_p_copy[:,0]*np.cos(angle) - r_p_copy[:,1]*np.sin(angle)
#             rel_pcd[:,1] = r_p_copy[:,0]*np.sin(angle) + r_p_copy[:,1]*np.cos(angle)
#             r_j_copy = rel_joint.clone()
#             rel_joint[:,0] = r_j_copy[:,0]*np.cos(angle) - r_j_copy[:,1]*np.sin(angle)
#             rel_joint[:,1] = r_j_copy[:,0]*np.sin(angle) + r_j_copy[:,1]*np.cos(angle)

#         if smpl_verts is not None:
#             rel_v = smpl_verts - global_trans
#             r_v_copy = rel_v.clone()
#             rel_v[:,0] = r_v_copy[:,0]*np.cos(angle) - r_v_copy[:,1]*np.sin(angle)
#             rel_v[:,1] = r_v_copy[:,0]*np.sin(angle) + r_v_copy[:,1]*np.cos(angle)
#             smpl_verts = rel_v + global_trans
#             smpl_verts += trans

#         human_points = rel_pcd + global_trans
#         smpl_joints = rel_joint + global_trans
#         human_points += trans
#         smpl_joints += trans
#         global_trans += trans
#         return human_points, smpl_joints, global_trans, smpl_verts

#     def __getitem__(self, ind):
#         # self.valid_samples[ind * self.interval]
#         sam_data = self.valid_hkps[ind * self.interval]
#         pcd_file = sam_data['pcd_file']
#         smpl_verts = sam_data['smpl_verts']
#         smpl_joints = sam_data['smpl_joints']
#         mesh_dict = sam_data['mesh_dict']
#         smpl_verts  =  torch.tensor(smpl_verts).float()
#         if 'human_points' not in sam_data.keys():
#             pcd = np.array(o3d.io.read_point_cloud(pcd_file).points)
#             root_min, root_max = np.min(smpl_joints, axis = 0) - 0.2, np.max(smpl_joints, axis = 0) + 0.2
#             root = (root_min + root_max ) / 2
#             valid_p_ind = (pcd[:,0] > root_min[0]) & \
#                 (pcd[:,1] > root_min[1]) & \
#                 (pcd[:,2] > root_min[2]) & \
#                 (pcd[:,0] <= root_max[0]) & \
#                 (pcd[:,1] <= root_max[1]) & \
#                 (pcd[:,2] <= root_max[2])
#             human_points = torch.tensor(pcd[valid_p_ind,:]).unsqueeze(0)
#         else:
#             human_points = torch.tensor(sam_data['human_points']).unsqueeze(0)
#             root = torch.mean(human_points.squeeze(), dim = 0)

#         smpl_joints, root = torch.tensor(smpl_joints).unsqueeze(0), root
#         pc_dist = torch.tensor(sam_data['pc_dist'])
#         min_dist, idx1 = torch.min(pc_dist, dim=1) # (B,N)
#         vis_label = min_dist < 0.25 #[N]
#         seg_label = (torch.ones([human_points.shape[1]]) * len(self.JOINTS_IDX)).long() #[M]
#         seg_label[vis_label] = idx1.squeeze(0)[vis_label].cpu()

#         vis_label_kpts = torch.min(pc_dist, dim = 0)[0] < 0.25
#         smpl_joints = smpl_joints.squeeze()
#         human_points = human_points.squeeze()
#         max_, min_ = human_points.max(dim = 0)[0], human_points.min(dim = 0)[0]
#         root = (max_ + min_) / 2

#         if self.fix_pts_num:
#             now_pt_num = int(human_points.shape[0])
#             if now_pt_num > self.point_num:
#                 choice_indx = np.random.randint(0, now_pt_num, size = [self.point_num])
#                 human_points = human_points[choice_indx,:]
#                 seg_label = seg_label[choice_indx]
#             else:
#                 choice_indx = np.random.randint(0, now_pt_num, size = [self.point_num - now_pt_num])
#                 human_points = torch.cat([human_points, human_points[choice_indx]], dim = 0)
#                 seg_label = torch.cat([seg_label, seg_label[choice_indx]], dim = 0)
        
#         if self.is_train and self.augmentation:
#             human_points, smpl_joints, root, smpl_verts = self.augment(human_points, smpl_joints, root, smpl_verts)
#         smpl_joint_24 = self.joint_24_regressor @ smpl_verts.float()
#         # smpl_joint_24_zero = 
#         sample = {
#             'global_trans' : root,
#             'smpl_joints'  : smpl_joints,   
#             'smpl_verts'   : smpl_verts,   
#             'smpl_verts_local'   : (smpl_verts - root),#.to(self.device), 
#             'human_points' : human_points,
#             'human_points_local': (human_points - root),#.to(self.device),
#             'smpl_joints_local' : (smpl_joints - root),#.to(self.device),
#             'smpl_joints24_local' : (smpl_joint_24 - root),#.to(self.device),
#             'vis_label' : vis_label_kpts.float(),
#             'seg_label' : seg_label.long(),
#             'smpl_pose': np.concatenate([mesh_dict['global_orient'], mesh_dict['body_pose'],np.zeros(6)], axis = -1),
#             'betas': mesh_dict['betas'],
#             'location_dict' : sam_data['location_dict'],
#             'has_3d_joints': 1,
#             'has_smpl': 1,
#             'num_points': now_pt_num
#         }
#         if self.load_v2v:
#             location_dict = sam_data['location_dict']
#             frame, time, id = location_dict['scene'], location_dict['time'], location_dict['id']
#             pose_pred = self.v2v_pred[frame][time][id]
#             pose_pred_local = (torch.tensor(pose_pred) - root).to(self.device)
#             sample.update({'pose_pred': pose_pred, 'pose_pred_local': pose_pred_local})
#         if self.return_torch:
#             for k, v in sample.items():
#                 if k == 'mask':
#                     sample[k] = torch.tensor(v)#.to(self.device)
#                 elif type(v) != str and type(v) != torch.Tensor and type(v) != dict:
#                     sample[k] = torch.tensor(v).float()#.to(self.device)
#                 elif type(v) == torch.Tensor and k != 'seg_label':
#                     sample[k] = v.float()#.to(self.device)
#         return sample
    
#     def __len__(self):
#         return len(self.valid_hkps) // self.interval
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SLOPER4D dataset')
    parser.add_argument('--dataset_root', type=str, 
                        default='/Extra/fanbohao/posedataset/PointC/Waymo/resave_files/', 
                        help='Path to data file')
    # parser.add_argument('--pkl_file', type=str, 
    #                     default='/disk1/fanbohao/fbh_data/sloper4d/seq003_street_002/seq003_street_002_labels.pkl', 
    #                     help='Path to the pkl file')
    parser.add_argument('--batch_size', type=int, default=1, 
                        help='The batch size of the data loader')
    parser.add_argument('--index', type=int, default=-1,
                        help='the index frame to be saved to a image')
    args = parser.parse_args()
    test_dataset = HumanM3_Dataset(is_train = False,
                                return_torch=True, device = 'cpu',
                                fix_pts_num=True, interval = 1)
    train_dataset = HumanM3_Dataset(is_train = True,
                                return_torch=True, device = 'cpu',
                                fix_pts_num=True, interval = 1)
    # import pdb; pdb.set_trace()
    #
    # =====> attention 
    # Batch_size > 1 is not supported yet
    # because bbox and 2d keypoints missing in some frames
    dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    
    root_folder = os.path.dirname(args.dataset_root)
    joint_index = np.array([0,1,2,4,5,7,8,10,11,12,15,16,17,18,19,20,21])
    bone_index = [[0,1], [0,2], [1,3], [2,4], [3,5], [4,6], [5,7], [6,8], [0,9], [9,10], [9,11], [9,12], [11,13], [12,14], [13,15], [14,16]]
    color = np.random.rand(18,3)
    for index, sample in enumerate(dataloader):
        if index % 10 != 0:
            continue
        human_points = sample['human_points_local'][0]
        smpl_joints = sample['smpl_joints_local'][0]
        seg_label = sample['seg_label'][0].cpu().numpy()
        smpl_verts = sample['smpl_verts_local'][0]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        for i in range(17+1):
            indexes = seg_label == i
            ax.scatter(human_points[indexes,0], human_points[indexes,1], human_points[indexes,2], c = color[i])
        ax.scatter(smpl_joints[:,0], smpl_joints[:,1], smpl_joints[:,2], c = 'r')
        for i in range(smpl_joints.shape[0]):
            ax.text(smpl_joints[i,0], smpl_joints[i,1], smpl_joints[i,2], s = str(i), c = 'b')
        center = smpl_joints[0,:]
        ax.set_xlim(center[0] - 1.0, center[0] + 1.0)
        ax.set_ylim(center[1] - 1.0, center[1] + 1.0)
        ax.set_zlim(center[2] - 1.0, center[2] + 1.0)
        ax1 = plt.figure().add_subplot(111, projection = '3d')
        ax1.set_xlim(center[0] - 1.0, center[0] + 1.0)
        ax1.set_ylim(center[1] - 1.0, center[1] + 1.0)
        ax1.set_zlim(center[2] - 1.0, center[2] + 1.0)
        ax1.scatter(smpl_verts[:,0], smpl_verts[:,1], smpl_verts[:,2], c = 'b')

        plt.show()
