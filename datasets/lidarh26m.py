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

keypoints_name = {
    1:'nose',
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

class LiDARH26M_Dataset(lidar_Dataset):
    def __init__(self, root_folder = '/Extra/fanbohao/posedataset/PointC/humanm3/',
                 dataset_path = 'save_data/lidarh26m/', is_train = True,
                 device='cpu',
                 return_torch:bool=True, 
                 fix_pts_num:bool=False,
                 print_info:bool=True,
                 augmentation:bool=False, load_v2v = False, interval = 5):
        super().__init__(is_train = is_train,
                 return_torch=return_torch, 
                 fix_pts_num=fix_pts_num,
                 augmentation=augmentation,
                 load_v2v = load_v2v, 
                 interval = interval)
        self.device       = device
        self.return_torch = return_torch
        self.print_info   = print_info
        self.fix_pts_num  = fix_pts_num
        self.point_num = 1024
        split = 'train' if is_train else 'test'
        data_file = 'train.pkl' if is_train else 'test.pkl'
        self.valid_hkps = []
        os.makedirs(dataset_path, exist_ok=True)
        pkl_file = os.path.join(dataset_path, data_file)
        self.JOINTS_IDX = [0, 1, 2, 4, 5, 7, 8, 12, 15, 16, 17, 18, 19, 20, 21]
        self.load_v2v = load_v2v
        if load_v2v:
            with open(os.path.join('pose_results/lpformer/lidarh26m', split + '.json'), 'r') as f:
                self.v2v_pred = json.load(f)
            self.pred_mode = 'str'
            
        if not os.path.exists(pkl_file):
            human_model = smplx.create('./smplx_models/', model_type = 'smpl',
                                    gender='neutral', 
                                    use_face_contour=False,
                                    ext="npz").cuda()
            calib_file = 'lidarcap_train32.npy' if is_train else 'lidarcap_test32.npy'
            calib_data = np.load(os.path.join('save_data/lidarh26m/', calib_file), allow_pickle=True).tolist()
            gt_pose = calib_data['gt_pose']
            pointclouds = calib_data['pointcloud']
            gt_trans = calib_data['gt_trans']
            gt_betas = calib_data['gt_shape']
            num_seq = gt_pose.shape[0]
            seq_len = gt_pose.shape[1]
            for ns in tqdm(range(num_seq)):
                for sl in range(seq_len):
                    pcd = np.array(pointclouds[ns][sl])
                    body_pose, transl, betas, global_orient = gt_pose[ns][sl][3:], gt_trans[ns][sl], gt_betas[ns][sl], gt_pose[ns][sl][:3]
                    global_orient = torch.tensor(global_orient).float().unsqueeze(0)
                    body_pose = torch.tensor(body_pose).float().unsqueeze(0)
                    # body_pose = torch.cat([body_pose, torch.zeros(1,6)], dim = 1)
                    transl = torch.tensor(transl).float().unsqueeze(0)
                    betas=torch.tensor(betas).unsqueeze(0)
                    smpl_md = human_model(betas = betas.cuda().float(), 
                                            return_verts=True, 
                                            body_pose=body_pose.cuda().float(),
                                            global_orient=global_orient.cuda().float(),
                                            transl=transl.cuda())
                    
                    smpl_joints = smpl_md.joints.squeeze()[self.JOINTS_IDX].cpu().numpy()
                    # root_min, root_max = np.min(smpl_joints, axis = 0) - 0.2, np.max(smpl_joints, axis = 0) + 0.2
                    # valid_p_ind = (pcd[:,0] > root_min[0]) & \
                    #     (pcd[:,1] > root_min[1]) & \
                    #     (pcd[:,2] > root_min[2]) & \
                    #     (pcd[:,0] <= root_max[0]) & \
                    #     (pcd[:,1] <= root_max[1]) & \
                    #     (pcd[:,2] <= root_max[2])
                    # root = (root_min + root_max ) / 2
                    # human_points = pcd[valid_p_ind,:]
                    human_points = pcd
                    location_dict = {'seq':ns, 'seq_ind':sl}
                    mesh_dict = {}
                    mesh_dict['body_pose'] = body_pose.squeeze().numpy()
                    mesh_dict['transl'] = transl.squeeze().numpy()
                    mesh_dict['betas'] = betas.squeeze().numpy()
                    mesh_dict['global_orient'] = global_orient.squeeze().numpy()
                    if human_points.shape[0] > 0:
                        dist1, idx1, dist2, idx2, pc_dist = self.nn_distance(torch.tensor(human_points).unsqueeze(0), torch.tensor(smpl_joints).unsqueeze(0), is_cuda=True)
                        pc_dist = pc_dist.squeeze().detach().cpu().numpy()
                        self.valid_hkps.append({'mesh_dict':mesh_dict, 'smpl_joints':smpl_joints, 'pc_dist': pc_dist,\
                                            'smpl_verts':smpl_md.vertices.squeeze().cpu().numpy(),
                                            'location_dict':location_dict, 'human_points':human_points})
            print(len(self.valid_hkps))
            with open(pkl_file, 'wb') as f:
                pickle.dump(self.valid_hkps, f)
        else:
            with open(pkl_file, 'rb') as f:
                self.valid_hkps = pickle.load(f)
        self.interval = interval
        self.augmentation = augmentation
        self.is_train = is_train
        for vhk in self.valid_hkps:
            scene, time, id = vhk['location_dict']['seq'], vhk['location_dict']['seq_ind'], 0
            vhk['location_dict'] = {'scene': scene, 'time': time, 'id': id}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SLOPER4D dataset')
    parser.add_argument('--dataset_root', type=str, 
                        default='/Extra/fanbohao/posedataset/PointC/Waymo/resave_files/', 
                        help='Path to data file')
    parser.add_argument('--batch_size', type=int, default=1, 
                        help='The batch size of the data loader')
    parser.add_argument('--index', type=int, default=-1,
                        help='the index frame to be saved to a image')
    args = parser.parse_args()
    test_dataset = LiDARH26M_Dataset(is_train = False,
                                return_torch=True, device = 'cpu',
                                fix_pts_num=True, interval = 1)
    train_dataset = LiDARH26M_Dataset(is_train = True,
                                return_torch=True, device = 'cpu',
                                fix_pts_num=True, interval = 1)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    root_folder = os.path.dirname(args.dataset_root)
    joint_index = np.array([0,1,2,4,5,7,8,10,11,12,15,16,17,18,19,20,21])
    bone_index = [[0,1], [0,2], [1,3], [2,4], [3,5], [4,6], [5,7], [6,8], [0,9], [9,10], [9,11], [9,12], [11,13], [12,14], [13,15], [14,16]]
    color = np.random.rand(18,3)
    for index, sample in enumerate(dataloader):
        if index % 10 != 0:
            continue
        human_points = sample['human_points_local'][0]
        smpl_joints = sample['smpl_joints24_local'][0]
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
