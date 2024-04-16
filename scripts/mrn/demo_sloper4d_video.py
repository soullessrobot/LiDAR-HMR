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
import pickle
import math
from models.pose2mesh.pointcloud2mesh_net import pointcloud2mesh_net
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--state_dict', default='save_state/sloper4d/mesh/pct_mf/adj/2023-11-11 20:01:40/epoch27_0.05103_0.05189_0.00104_0.09351.pth', required=False, type=str)
    parser.add_argument(
        '--cfg', default='configs/mesh/default.yaml', required=False, type=str)
    parser.add_argument(
        '--which', default='LiDAR-HMR', required=False, type=str)
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
    fx = fy = math.sqrt(3) / 2 * height
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
extrinsic1=np.array([[
		1.0,
		-0.0,
		-0.0,
		0.0,
		0.0,
		-1.0,
		-0.0,
		0.0,
		0.0,
		-0.0,
		-1.0,
		0.0,
		0.019714757800102234,
		-0.030236056074500084,
		1.7749301583855281,
		1.0
	]]).reshape(4,4).swapaxes(0,1)
extrinsic2=np.array([[
		0.046966780818727322,
		-0.0046471538932741398,
		0.99888564183304684,
		0.0,
		-0.016427983599161443,
		-0.99985752625654578,
		-0.0038792451593920758,
		0.0,
		0.99876135430561763,
		-0.016227481284327865,
		-0.047036432653587569,
		0.0,
		0.24414968314633315,
		-0.034283239858513573,
		2.0270504151575284,
		1.0
	]]).reshape(4,4).swapaxes(0,1)
extrinsic3=np.array([[
		-0.98460131847019505,
		0.022735559076101516,
		-0.17333014169569869,
		0.0,
		-0.030223711921058927,
		-0.99871479596126134,
		0.040685176241098284,
		0.0,
		-0.17218237686960514,
		0.045297358439024736,
		0.98402305786702982,
		0.0,
		-0.062341434236105331,
		-0.018695385937214412,
		2.2568898581473014,
		1.0
	]]).reshape(4,4).swapaxes(0,1)
# extrinsic

def demo_sloper(model, which):
    model.eval()
    path_sloper4d_pkl = 'save_data/sloper4d/test.pkl'
    with open(path_sloper4d_pkl, 'rb') as f:
        scene_data_list = pickle.load(f)
    file_name = scene_data_list[0]['file_basename']
    human_points_all = scene_data_list[0]['human_points']
    global_trans = scene_data_list[0]['global_trans']
    # import pdb; pdb.set_trace()
    smpl_ = smplx.create(gender='neutral', model_type = 'smpl', \
                          model_path='./smplx_models/')
    if which == 'gt':
        smpl_pose = scene_data_list[0]['smpl_pose']
        betas = scene_data_list[0]['betas']
        
    if which == 'V2V_P2M':
        with open(os.path.join('sloper4d_v2v_results', 'seq009_running_002' + '.json'), 'r') as f:
            v2v_pred = json.load(f)
    # import pdb; pdb.set_trace()
    length = len(file_name)
    index = np.arange(length)
    # shuffle(index)
    height = 720
    width = 1280

    vis = o3d.visualization.Visualizer()
    vis.create_window(height=height, width=width)
    vis.get_render_option().point_size = 2.0
    intrinsic = get_camera_intrinsic(height=height, width=width)
    # extrinsic = np.array([[0.9084606054319955, 0.41489385926628997, -0.0506202915959241, -17.29760034536154], [0.10859004813655097, -0.3512346167993969, -0.929969056155893, -2.3696316972124682], [-0.4036180494277157, 0.8393433918873163, -0.3641361567755408, 21.946030870146064], [0.0, 0.0, 0.0, 1.0]])

    # param = combine_camera_param(extrinsic=extrinsic, intrinsic=intrinsic)
    param1 = interpolate_world_trajectory([extrinsic1, extrinsic2], 125, intrinsic)
    param2 = interpolate_world_trajectory([extrinsic2, extrinsic3], 125, intrinsic)
    # param3 = interpolate_world_trajectory([extrinsic_3, extrinsic_4], 60, intrinsic)
    param_list = param1 + param2
    # vis.get_view_control().convert_from_pinhole_camera_parameters(param)
    for ind in index:
        if ind <= 110 or ind > 360:
            continue
        print(ind, file_name[ind])
        
        pcd = torch.tensor(human_points_all[ind]).cuda()
        trans = torch.tensor(global_trans[ind]).cuda().unsqueeze(0)
        meshes = []
        # meshes.append(pcd_)
        now_pt_num = pcd.shape[0]
        if now_pt_num <= 20:
            continue
        if now_pt_num > in_len:
            choice_indx = np.random.randint(0, now_pt_num, size = [in_len])
            human_points = pcd[choice_indx,:]
        elif now_pt_num < in_len:
            choice_indx = np.random.randint(0, now_pt_num, size = [in_len - now_pt_num])
            human_points = torch.cat([pcd, pcd[choice_indx]], dim = 0)
        # pcd_ick.append(human_points - center_in[ind_p].cuda())
        pcd_input = human_points - trans
        pcd_input = pcd_input.unsqueeze(0)
        
        if which != 'pcd' and which != 'gt':
            with torch.no_grad():
                if which == 'V2V_P2M':
                    v2v_pose = torch.tensor(v2v_pred[str(ind)]).cuda() - trans[:,None,:]
                    ret_dict = model(v2v_pose.float())
                else:
                    ret_dict = model(pcd_input.float())
        
        if which == 'pcd':
            pcd_ = o3d.open3d.geometry.PointCloud()
            pcd_input = pcd_input.squeeze().cpu().numpy()
            pcd_input[:,[1,2]] = pcd_input[:,[2,1]]
            pcd_.points= o3d.open3d.utility.Vector3dVector(pcd_input)
            pcd_.paint_uniform_color([0.5,0.5,0.5])
            vis.add_geometry(pcd_)
        elif which == 'gt':
            pose_ = torch.tensor(smpl_pose[[ind]]).float()
            pred_vertices = smpl_(body_pose = pose_[:,3:], \
                global_orient = pose_[:,:3], betas = torch.tensor(betas).unsqueeze(0).float()).vertices.squeeze().detach().cpu().numpy()
            pred_vertices[:,[1,2]] = pred_vertices[:,[2,1]]
            vis.add_geometry(get_mesh(pred_vertices, smpl_.faces))
        else:
            pred_vertices = ret_dict['mesh_out'].squeeze().cpu().numpy()
            pred_vertices[:,[1,2]] = pred_vertices[:,[2,1]]
            vis.add_geometry(get_mesh(pred_vertices, smpl_.faces))
        
        # pcd_ = o3d.open3d.geometry.PointCloud()
        # pcd_input = pcd_input.squeeze().cpu().numpy()
        # pcd_input[:,[1,2]] = pcd_input[:,[2,1]]
        # pcd_.points= o3d.open3d.utility.Vector3dVector(pcd_input)
        # pcd_.paint_uniform_color([0.5,0.5,0.5])
        # vis.add_geometry(pcd_)
        vis.get_view_control().convert_from_pinhole_camera_parameters(param_list[ind - 110])
        vis.poll_events()
        vis.update_renderer()
        save_folder = os.path.join("demo_sloper", which)
        os.makedirs(save_folder, exist_ok = True)
        vis.capture_screen_image(os.path.join(save_folder, f'{ind:04d}.png'),do_render=True)
        vis.clear_geometries()

args = parse_args()
# setup_seed(10)

update_config(args.cfg)
which = args.which
if which == 'LiDAR_HMR':
    model = pose_meshgraphormer(pmg_cfg = config).cuda()
elif which == 'V2V_P2M':
    model = pointcloud2mesh_net().cuda()
elif which == 'PCT_P2M':
    model = pose_mesh_net().cuda()
else:
    model = pose_meshgraphormer(pmg_cfg = config).cuda()
if args.state_dict != '' and which != 'pcd':
    state_dict = torch.load(args.state_dict)
    model.load_state_dict(state_dict['net'])

demo_sloper(model, which)
# import cv2
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# fps = 20.0
# frame_size = (1280, 720)
# video_writer = cv2.VideoWriter('demo_sloper/demo.mp4', fourcc, fps, frame_size)
# for frame in tqdm(range(8112)):
#     if os.path.exists('demo_sloper/single_view/'+str(frame).zfill(4)+'.png'):
#         img = cv2.imread('demo_sloper/single_view/'+str(frame).zfill(4)+'.png')
#         video_writer.write(img)
# video_writer.release()