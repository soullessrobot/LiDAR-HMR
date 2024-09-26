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

class WAYMOV2_Dataset(lidar_Dataset):
    def __init__(self, root_folder, dataset_path = 'save_data/waymov2/', is_train = True,
                 device='cpu', mesh_folder = '/Extra/fanbohao/fbh_code/SMPL/smplify-x/waymo/meshes/',
                 return_torch:bool=True, 
                 fix_pts_num:bool=False,
                 print_info:bool=True,
                 augmentation:bool=False, load_v2v = False):
        super().__init__(is_train = is_train,
                 return_torch=return_torch, 
                 fix_pts_num=fix_pts_num,
                 augmentation=augmentation,
                 load_v2v = load_v2v, 
                 interval = 1)
        self.device       = device
        self.return_torch = return_torch
        self.print_info   = print_info
        self.fix_pts_num  = fix_pts_num
        self.point_num = 1024
        self.num_joints = 14
        self.root_folder = root_folder
        split = 'training' if is_train else 'validation'
        data_file = 'train.pkl' if is_train else 'test.pkl'
        self.valid_hkps = []
        os.makedirs(dataset_path, exist_ok=True)
        pkl_file = os.path.join(dataset_path, data_file)
        self.JOINTS_IDX = [0, 1, 2, 4, 5, 7, 8, 12, 15, 16, 17, 18, 19, 20, 21]
        if not os.path.exists(pkl_file):
            human_model = smplx.create('./smplx_models/', model_type = 'smpl',
                                    gender='neutral', 
                                    use_face_contour=False,
                                    ext="npz")
            json_files = glob.glob(os.path.join(root_folder, split, '*', 'keypoints_3d', '*.json'))
            for jf in tqdm(json_files):
                with open(jf, 'r') as f:
                    dict_ = json.load(f)
                names = jf.split('/')
                split_, frame_, time_ = names[-4], names[-3], names[-1].replace('.json', '')
                mesh_name = split_ + ',' + frame_ + ',' + time_
                mesh_files = sorted(glob.glob(os.path.join(mesh_folder, mesh_name, '*.npy')))
                mesh_verts = [np.load(mf, allow_pickle = True).tolist()['result'] for mf in mesh_files]
                # import pdb; pdb.set_trace()
                pcd_file = jf.replace('json', 'pcd').replace('keypoints_3d', 'pointcloud')
                for indx, key_ in enumerate(dict_.keys()):
                    kpt_this = np.array(dict_[key_]['keypoints'])
                    type_ = np.array(dict_[key_]['type'])
                    kpt_valid = np.zeros([self.num_joints,3])
                    valid_flag = np.zeros([self.num_joints])
                    for ind in range(kpt_this.shape[0]):
                        v_ind = keypoints_index[keypoints_name[type_[ind]]]
                        kpt_valid[v_ind] = kpt_this[ind]
                        valid_flag[v_ind] = 1
                    valid_flag = valid_flag > 0
                    mesh_dict = mesh_verts[indx]
                    body_pose, transl, betas, global_orient = mesh_dict['body_pose'], mesh_dict['transl'], mesh_dict['betas'], mesh_dict['global_orient']
                    global_orient = torch.tensor(global_orient).float().unsqueeze(0)
                    body_pose = torch.tensor(body_pose).float().unsqueeze(0)
                    body_pose = torch.cat([body_pose, torch.zeros(1,6)], dim = 1)
                    transl = torch.tensor(transl).float().unsqueeze(0)
                    betas=torch.tensor(betas).unsqueeze(0)
                    # import pdb; pdb.set_trace()
                    smpl_md = human_model(betas = betas, 
                                            return_verts=True, 
                                            body_pose=body_pose,
                                            global_orient=global_orient,
                                            transl=transl)
                    
                    pcd = np.array(o3d.io.read_point_cloud(pcd_file).points)
                    root_min, root_max = np.min(smpl_joints, axis = 0) - 0.2, np.max(smpl_joints, axis = 0) + 0.2
                    valid_p_ind = (pcd[:,0] > root_min[0]) & \
                        (pcd[:,1] > root_min[1]) & \
                        (pcd[:,2] > root_min[2]) & \
                        (pcd[:,0] <= root_max[0]) & \
                        (pcd[:,1] <= root_max[1]) & \
                        (pcd[:,2] <= root_max[2])
                    human_points = torch.tensor(pcd[valid_p_ind,:]).unsqueeze(0)
                    dist1, idx1, dist2, idx2, pc_dist = self.nn_distance(human_points, smpl_joints.unsqueeze(0), is_cuda=True)
                    pc_dist = pc_dist.squeeze().detach().cpu().numpy()
                    location_dict = {'split':split_, 'frame':frame_, 'time':time_, 'id':key_}
                    self.valid_hkps.append({'keypoints':kpt_valid, 'flag':valid_flag, 'human_points':human_points, 'pc_dist': pc_dist,\
                                             'mesh_dict':mesh_dict, 'smpl_joints':smpl_md.joints.squeeze()[self.JOINTS_IDX].numpy(), \
                                              'smpl_verts':smpl_md.vertices.squeeze().numpy(), 'location_dict':location_dict})
            with open(pkl_file, 'wb') as f:
                pickle.dump(self.valid_hkps, f)
        else:
            with open(pkl_file, 'rb') as f:
                self.valid_hkps = pickle.load(f)
                for hkps in self.valid_hkps:
                    hkps['human_points'] = hkps['human_points'].squeeze(0)
                    hkps['mesh_dict']['body_pose'] = np.concatenate([hkps['mesh_dict']['body_pose'], np.zeros(6)], axis = -1)
        self.load_v2v = load_v2v
        if load_v2v:
            # self.v2v_pred = []
            with open(os.path.join('pose_results/lpformer/waymo', split + '.json'), 'r') as f:
                self.v2v_pred = (json.load(f))
            for vhk in self.valid_hkps:
                vhk['location_dict']['scene'] = vhk['location_dict']['frame']
        self.num_joints = 14
        self.bone_link = [[0,13],[13,1],[13,7], [1,7],[1,2],[2,3],[1,4],[4,5],[5,6],[4,10],[7,8],[8,9],[7,10],[10,11],[11,12]]
        self.augmentation = augmentation
        self.is_train = is_train

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
    train_dataset = WAYMOV2_Dataset(args.dataset_root, is_train = True, dataset_path = './save_data/waymov2/',
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, augmentation = True)
    test_dataset = WAYMOV2_Dataset(args.dataset_root, is_train = False, dataset_path = './save_data/waymov2/',
                               return_torch=False, 
                               fix_pts_num=False)
    import pdb; pdb.set_trace()
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    root_folder = os.path.dirname(args.dataset_root)
    joint_index = np.array([0,1,2,4,5,7,8,10,11,12,15,16,17,18,19,20,21])
    bone_index = [[0,1], [0,2], [1,3], [2,4], [3,5], [4,6], [5,7], [6,8], [0,9], [9,10], [9,11], [9,12], [11,13], [12,14], [13,15], [14,16]]
    color = np.random.rand(18,3)
    for index, sample in enumerate(dataloader):
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
