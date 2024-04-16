import open3d as o3d
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from datasets.sloper4d import SLOPER4D_Dataset
from datasets.waymo_v2 import WAYMOV2_Dataset
from datasets.humanm3 import HumanM3_Dataset
# from models.graphormer.graphormer_model import graphormer_model
# from models.unsupervised.Network import point_net_ssg, smpl_model
from models.pose_mesh_net import pose_mesh_net, pose_meshgraphormer
# from models.v2v_posenet import V2VPoseNet
from tqdm import tqdm
import torch.optim as optim
import logging
import torch.nn.functional as F
import argparse
# import torch.distributed as dist
from scripts.eval_utils import mean_per_vertex_error, setup_seed, get_mesh, mean_per_edge_error
from models.graphormer.data.config import H36M_J17_NAME, H36M_J17_TO_J14, J24_NAME, J24_TO_J14
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import matplotlib.pyplot as plt
from models._smpl import SMPL
import smplx
from models.pmg_config import config, update_config
import glob
import json
import open3d as o3d
from random import shuffle
import math
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--state_dict', default='save_state/waymov2/mesh/pct_mf/default/2023-11-11 23:04:22/epoch28_0.06284_0.08235_0.00146_0.11891.pth', required=False, type=str)
    parser.add_argument(
        '--cfg', default='configs/mesh/default.yaml', required=False, type=str)
    args, rest = parser.parse_known_args()
    return args

def interpolate_world_trajectory(pose_list, interval_num, intrinsic):
    n = len(pose_list)
    rot_list = [np.array(pose)[:3, :3] for pose in pose_list]
    t_list = [np.array(pose)[:3, 3] for pose in pose_list]

    # Interpolate rotation
    slerp = Slerp(range(n), R.from_matrix(rot_list))
    rot_interp_list = slerp(np.linspace(0, n-1, interval_num*(n-1)+1)).as_matrix()
    # Interpolate translation
    interp = interp1d(range(n), t_list, axis=0, kind='linear')
    t_interp_list = interp(np.linspace(0, n-1, interval_num*(n-1)+1))
    # import pdb; pdb.set_trace()
    # Generate trajectory
    param_list = []
    for rot, t in zip(rot_interp_list, t_interp_list):
        pose = compose(rot, t)
        extrinsic = pose #convert_coord_world_to_camera(pose)
        param = combine_camera_param(extrinsic, intrinsic)
        param_list.append(param)
    # trajectory = []#o3d.camera.PinholeCameraTrajectory()
    # trajectory.parameters = param_list
    return param_list

def decompose(pose):
    return pose[:3, :3], pose[:3, 3]

def compose(rotmat, t):
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotmat, t
    return pose

def convert_coord_world_to_camera(W_pose):
    rotmat, t = decompose(W_pose)
    rotmat_inv = np.linalg.inv(rotmat)
    livox2cam = np.array([[0,-1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]])
    camera_extrinsic = livox2cam @ compose(rotmat_inv, rotmat_inv @ -t)
    return camera_extrinsic

def get_camera_intrinsic(height, width):
    fx = fy = 1 / 2 * width #math.sqrt(3)
    cx, cy = (width-1)/2, (height-1)/2
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    return intrinsic

def combine_camera_param(extrinsic, intrinsic):
    param = o3d.camera.PinholeCameraParameters()
    param.extrinsic = extrinsic
    param.intrinsic = intrinsic
    return param

logger = logging.getLogger(__name__)
in_len = 1024
use_gt = False
extrinsic_1=np.array([[
		-0.89708720579826495,
		-0.034173150294390997,
		0.4405300682042243,
		0.0,
		-0.41269527900787267,
		0.42098143273050526,
		-0.80774825284910723,
		0.0,
		-0.15785167682883725,
		-0.9064253025457355,
		-0.39176015497648087,
		0.0,
		3.5227150901381403,
		7.3423157880753296,
		0.31605670414597631,
		1.0
	]]).reshape(4,4).swapaxes(0,1)
extrinsic_2=np.array([[
		0.3068401243404672,
		-0.046503514730810921,
		0.95062430076892535,
		0.0,
		-0.94447387814632044,
		-0.13823896455238247,
		0.29809240543619142,
		0.0,
		0.11755097444928402,
		-0.98930653075635089,
		-0.086338615977194,
		0.0,
		-7.5635313088926299,
		4.7498668313466261,
		1.9980358781411716,
		1.0
	]]).reshape(4,4).swapaxes(0,1)
extrinsic_3=np.array([[
	    0.35160559611087217,
		-0.31755877731870036,
		0.88064177037622415,
		0.0,
		-0.93485037486949363,
		-0.1686233561512252,
		0.3124434994789686,
		0.0,
		0.049277595212078251,
		-0.93312517204721757,
		-0.35615885767725247,
		0.0,
		3.2870677718442085,
		8.5524152202898165,
		-3.0020802096379078,
		1.0
]]).reshape(4,4).swapaxes(0,1)

# extrinsic_1=np.array([[
# 		0.93100651629130216,
# 		0.24561441972756587,
# 		-0.27000078415631384,
# 		0.0,
# 		0.34652035344491439,
# 		-0.36235791559573605,
# 		0.86522851643570164,
# 		0.0,
# 		0.11467567864000139,
# 		-0.89909415403894422,
# 		-0.42246797618452758,
# 		0.0,
# 		-11.177958062075026,
# 		1.5570957759269806,
# 		-0.88254431059294824,
# 		1.0
# 	]]).reshape(4,4).swapaxes(0,1)
# extrinsic_2=np.array([[
# 		0.84918243938706095,
# 		0.28221468665174504,
# 		-0.44636762346153425,
# 		0.0,
# 		0.52768035324775042,
# 		-0.48711299033032768,
# 		0.69589825366053004,
# 		0.0,
# 		-0.021038760252686615,
# 		-0.82648400183518578,
# 		-0.56256694292992948,
# 		0.0,
# 		-3.3430918379474472,
# 		7.4437619347796913,
# 		0.50700492661491259,
# 		1.0
# 	]]).reshape(4,4).swapaxes(0,1)
# extrinsic_3=np.array([[
# 		0.83963636802678321,
# 		0.42229656870808535,
# 		-0.34157924050528848,
# 		0.0,
# 		0.52494801399343594,
# 		-0.46950081839482316,
# 		0.7099285626955284,
# 		0.0,
# 		0.13942866309026944,
# 		-0.77539318386478528,
# 		-0.6158937070021836,
# 		0.0,
# 		10.35573298652376,
# 		15.697113601967315,
# 		-8.4537068787341312,
# 		1.0
# 	]]).reshape(4,4).swapaxes(0,1)
# extrinsic_4=np.array([[
# 		-0.95144869591262793,
# 		-0.10209318241354115,
# 		0.29038312821311646,
# 		0.0,
# 		-0.30432785406727836,
# 		0.45345598513800262,
# 		-0.83771249649347412,
# 		0.0,
# 		-0.046151232756722442,
# 		-0.88541213660486884,
# 		-0.46250990483214072,
# 		0.0,
# 		-13.323414642167835,
# 		3.8699147029076375,
# 		2.3730658768694548,
# 		1.0
# 	]]).reshape(4,4).swapaxes(0,1)

intrinsic = np.array([[
		693.68634843133543,
        0.0,
        0.0,
        0.0,
        693.68634843133543,
        0.0,
        767.5,
        400.0,
        1.0
	]]).reshape(3,3).swapaxes(0,1)
width, height, fx, fy, cx, cy = 1536, 801, 693.68634843, 693.68634843, 767.5, 400.0
intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
# import pdb; pdb.set_trace()
in_len = 1024
use_gt = True
# pcd = np.array(o3d.io.read_point_cloud(pcd_file).points)
def demo_waymo(model):
    model.eval()
    path_waymo = '/Extra/fanbohao/posedataset/PointC/Waymo/demo_validation/validation'
    box_json = glob.glob(os.path.join(path_waymo, '*', 'bbox', '*.json'))
    shuffle(box_json)
    height = 801
    width = 1536

    vis = o3d.visualization.Visualizer()
    vis.create_window(height=height, width=width)
    vis.get_render_option().point_size = 2.0
    
    for box_j in box_json:
        # if box_j != '/Extra/fanbohao/posedataset/PointC/Waymo/demo_validation/validation/17694030326265859208_2340_000_2360_000/bbox/1557197745747516.json':
        # if box_j != '/Extra/fanbohao/posedataset/PointC/Waymo/demo_validation/validation/8079607115087394458_1240_000_1260_000/bbox/1557265270262729.json':
        # if box_j != '/Extra/fanbohao/posedataset/PointC/Waymo/demo_validation/validation/9243656068381062947_1297_428_1317_428/bbox/1508793799912962.json':
        if box_j != '/Extra/fanbohao/posedataset/PointC/Waymo/demo_validation/validation/17791493328130181905_1480_000_1500_000/bbox/1543540620587556.json':
            continue
        with open(box_j, 'r') as f:
            boxes = json.load(f)
        center_, size_ = boxes['center'], boxes['size']
        pcd_file = box_j.replace('bbox', 'pointcloud').replace('json', 'pcd')
        pcd = torch.tensor(np.array(o3d.io.read_point_cloud(pcd_file).points)).cuda()
        pcd_input = []
        center_in = []
        if len(center_) < 4:
            continue
        print(box_j)
        for (center, size) in zip(center_, size_):
            r_ = max(size[0], size[1]) / 2
            h_ = size[2]
            r_c = pcd - torch.tensor(center).unsqueeze(0).cuda()
            ind_in = (r_c[...,:2].norm(dim = -1) < r_) & (r_c[...,2].abs() < h_ / 2)
            if pcd[ind_in].shape[0] > 50:
                pcd_input.append(pcd[ind_in])
                center_in.append(torch.tensor(center).unsqueeze(0))
        pcd_ick = []
        for ind_p, pcd_ in enumerate(pcd_input):
            now_pt_num = pcd_.shape[0]
            if now_pt_num > in_len:
                choice_indx = np.random.randint(0, now_pt_num, size = [in_len])
                human_points = pcd_[choice_indx,:]
            else:
                choice_indx = np.random.randint(0, now_pt_num, size = [in_len - now_pt_num])
                human_points = torch.cat([pcd_, pcd_[choice_indx]], dim = 0)
                
            # print(now_pt_num)
            # choice_indx = np.random.randint(0, now_pt_num, size = [in_len - now_pt_num])
            # human_points = torch.cat([pcd_, pcd_[choice_indx]], dim = 0)
            pcd_ick.append(human_points - center_in[ind_p].cuda())
        pcd_input = torch.stack(pcd_ick, dim = 0)
        
        with torch.no_grad():
            ret_dict = model(pcd_input.float())
        pred_vertices = ret_dict['mesh_out'].cpu().numpy()
        meshes = []
        pcd_ = o3d.open3d.geometry.PointCloud()
        pcd_.points= o3d.open3d.utility.Vector3dVector(pcd.cpu().numpy())
        pcd_.paint_uniform_color([0.5,0.5,0.5])
        vis.add_geometry(pcd_)
        
        # vis.add_geometry(get_mesh(p_k, model.smpl.faces))
        for ind_v, p_v in enumerate(pred_vertices):
            p_k = p_v + center_in[ind_v].numpy()
            vis.add_geometry(get_mesh(p_k, model.smpl.faces))
            
        param1 = interpolate_world_trajectory([extrinsic_1, extrinsic_2], 60, intrinsic)
        param2 = interpolate_world_trajectory([extrinsic_2, extrinsic_3], 60, intrinsic)
        # param3 = interpolate_world_trajectory([extrinsic_3, extrinsic_4], 60, intrinsic)
        param_list = param1 + param2 #+ param3
        for indp, param in enumerate(param_list):
            vis.get_view_control().convert_from_pinhole_camera_parameters(param)
            vis.poll_events()
            vis.update_renderer()
            save_folder = 'demo_waymo/annular2'
            os.makedirs(save_folder, exist_ok = True)
            vis.capture_screen_image(os.path.join(save_folder, f'{indp:04d}.png'),do_render=True)
        vis.clear_geometries()      
        
args = parse_args()
# setup_seed(10)

update_config(args.cfg)
model = pose_meshgraphormer(pmg_cfg = config).cuda()
if args.state_dict != '':
    state_dict = torch.load(args.state_dict)
    model.load_state_dict(state_dict['net'])
demo_waymo(model)
