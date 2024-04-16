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
        '--state_dict', default='save_state/humanm3/mesh/pct_mf/adj/2023-11-11 20:00:01/epoch27_0.07770_0.08972_0.00112_0.08841.pth', required=False, type=str)
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
# pcd = np.array(o3d.io.read_point_cloud(pcd_file).points)
# extrinsic_1=np.array([[
# 		0.28425246975661128,
# 		0.52116770625757847,
# 		-0.8047265096860432,
# 		0.0,
# 		0.95614284801832705,
# 		-0.092245746317272004,
# 		0.27799564109851072,
# 		0.0,
# 		0.070649753153633379,
# 		-0.84845444441088524,
# 		-0.52453185617152631,
# 		0.0,
# 		-9.0937878162675201,
# 		-5.7153073388490805,
# 		18.953076542098042,
# 		1.0
# 	]]).reshape(4,4).swapaxes(0,1)
# extrinsic_2=np.array([[
# 		-0.95923061633103579,
# 		0.14932681258628233,
# 		-0.23995442845674439,
# 		0.0,
# 		0.26231325566202868,
# 		0.15434696341573775,
# 		-0.95255906419934311,
# 		0.0,
# 		-0.10520637146660239,
# 		-0.97666704558266038,
# 		-0.18722473521324323,
# 		0.0,
# 		5.0117950709202912,
# 		-0.79055434723472073,
# 		18.488119666592262,
# 		1.0
# 	]]).reshape(4,4).swapaxes(0,1)
extrinsic_1=np.array([[
		0.1132507338054321,
		0.55930929991117218,
		-0.82118656730697059,
		0.0,
		0.98053134854263357,
		0.070525085741864377,
		0.18326070720770218,
		0.0,
		0.16041367091895758,
		-0.82595358181557454,
		-0.54043328438238081,
		0.0,
		-7.7388010649236545,
		-4.0508157864010252,
		17.795170729331758,
		1.0
	]]).reshape(4,4).swapaxes(0,1)
extrinsic_2=np.array([[
		-0.73409939258105295,
		-0.33652801094553547,
		0.58978553700575831,
		0.0,
		-0.67501967175647148,
		0.45605644283051677,
		-0.57996634703624073,
		0.0,
		-0.073800572856278016,
		-0.82386978267309763,
		-0.5619537851498716,
		0.0,
		10.513425523740047,
		-4.8719592486066583,
		15.429343314244264,
		1.0
	]]).reshape(4,4).swapaxes(0,1)
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
    path_humanm3 = '/Extra/fanbohao/posedataset/PointC/humanm3/test/'
    box_json = sorted(glob.glob(os.path.join(path_humanm3, '*', 'smpl_calib', '*.json')))
    # cameras = glob.glob(os.path.join(path_humanm3, '*', 'smpl_calib', '*.json'))
    # shuffle(box_json)
    camera_bas3_list = [sorted(glob.glob(os.path.join(path_humanm3, 'basketball2', 'images', 'camera_' + str(i), '*.jpg'))) for i in range(3)]
    camera_bas1_list = [sorted(glob.glob(os.path.join(path_humanm3, 'basketball1', 'images', 'camera_' + str(i), '*.jpeg'))) for i in range(4)]
    # import pdb; pdb.set_trace()
    height = 801
    width = 1536

    vis = o3d.visualization.Visualizer()
    vis.create_window(height=height, width=width)
    vis.get_render_option().point_size = 2.0
    # import pdb; pdb.set_trace()
    # intrinsic = get_camera_intrinsic(height=height, width=width)
    param_list = interpolate_world_trajectory([extrinsic_1, extrinsic_2], 60, intrinsic)
    # param = combine_camera_param(extrinsic=extrinsic, intrinsic=intrinsic)
    
    if use_gt:
        smpl_model = smplx.create(gender='neutral', model_type = 'smpl', \
                          model_path='./smplx_models/')
    for ind, box_j in enumerate(box_json):
        scene = box_j.split('/')[-3]
        # if scene != 'basketball1':
        #     continue
        if box_j != os.path.join(path_humanm3, 'basketball2', 'smpl_calib', '4783.json'):
        # if box_j != os.path.join(path_humanm3, 'basketball1', 'smpl_calib', '1894.json'):
            continue
        # print(box_j)
        pcd_file = box_j.replace('smpl_calib', 'pointcloud').replace('json', 'pcd')
        frame = int(pcd_file.split('/')[-1].split('.')[0])
        indk = frame - 4500
        for ca_b in camera_bas3_list:
            print(ca_b[indk])
        # import pdb; pdb.set_trace()
        pcd_file = pcd_file.replace(str(frame).zfill(4), str(frame).zfill(6))
        pcd = torch.tensor(np.array(o3d.io.read_point_cloud(pcd_file).points)).cuda()
        
        with open(box_j, 'r') as f:
            smpl_poses = json.load(f)
        meshes = []
        pcd_ = o3d.open3d.geometry.PointCloud()
        pcd_.points= o3d.open3d.utility.Vector3dVector(pcd.cpu().numpy())
        pcd_.paint_uniform_color([0.5,0.5,0.5])
        vis.add_geometry(pcd_)
        
        pcd_input = []
        center_in = []
        for key in smpl_poses.keys():
            pose = smpl_poses[key]
            center_ = pose["transl"]
            r_ = 0.6
            h_ = 1.9
            r_c = pcd - torch.tensor(center_).unsqueeze(0).cuda()
            ind_in = (r_c[...,:2].norm(dim = -1) < r_) & (r_c[...,2].abs() < h_ / 2)
            if pcd[ind_in].shape[0] > 50:
                pcd_input.append(pcd[ind_in])
                center_in.append(torch.tensor(center_).unsqueeze(0))
            pcd_ick = []
            for ind_p, pcd_ in enumerate(pcd_input):
                now_pt_num = pcd_.shape[0]
                if now_pt_num > in_len:
                    choice_indx = np.random.randint(0, now_pt_num, size = [in_len])
                    human_points = pcd_[choice_indx,:]
                else:
                    choice_indx = np.random.randint(0, now_pt_num, size = [in_len - now_pt_num])
                    human_points = torch.cat([pcd_, pcd_[choice_indx]], dim = 0)
                pcd_ick.append(human_points - center_in[ind_p].cuda())
        pcd_input = torch.stack(pcd_ick, dim = 0)
        
        with torch.no_grad():
            ret_dict = model(pcd_input.float())
        pred_vertices = ret_dict['mesh_out'].cpu().numpy()
        
        for ind_v, p_v in enumerate(pred_vertices):
            p_k = p_v + center_in[ind_v].numpy()
            # meshes.append(get_mesh(p_k, model.smpl.faces))
            vis.add_geometry(get_mesh(p_k, model.smpl.faces))
        for indp, param in enumerate(param_list):
            vis.get_view_control().convert_from_pinhole_camera_parameters(param)
            vis.poll_events()
            vis.update_renderer()
            save_folder = 'demo_humanm3/annular1'
            os.makedirs(save_folder, exist_ok = True)
            # time.sleep(0.5)
            vis.capture_screen_image(os.path.join(save_folder, f'{indp:04d}.png'),do_render=True)
        # import pdb; pdb.set_trace()
        vis.clear_geometries()

args = parse_args()
# setup_seed(10)

update_config(args.cfg)
model = pose_meshgraphormer(pmg_cfg = config).cuda()
if args.state_dict != '':
    state_dict = torch.load(args.state_dict)
    model.load_state_dict(state_dict['net'])
demo_waymo(model)
