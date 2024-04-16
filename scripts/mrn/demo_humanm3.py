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

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--state_dict', default='save_state/humanm3/mesh/pct_mf/adj/2023-11-11 20:00:01/epoch27_0.07770_0.08972_0.00112_0.08841.pth', required=False, type=str)
    parser.add_argument(
        '--cfg', default='configs/mesh/default.yaml', required=False, type=str)
    args, rest = parser.parse_known_args()
    return args

logger = logging.getLogger(__name__)

in_len = 1024
use_gt = True
# pcd = np.array(o3d.io.read_point_cloud(pcd_file).points)
def demo_waymo(model):
    model.eval()
    path_humanm3 = '/Extra/fanbohao/posedataset/PointC/humanm3/test/'
    box_json = sorted(glob.glob(os.path.join(path_humanm3, '*', 'smpl_calib', '*.json')))
    # cameras = glob.glob(os.path.join(path_humanm3, '*', 'smpl_calib', '*.json'))
    shuffle(box_json)
    camera_bas3_list = [sorted(glob.glob(os.path.join(path_humanm3, 'basketball3', 'images', 'camera_' + str(i), '*.jpg'))) for i in range(3)]
    camera_bas1_list = [sorted(glob.glob(os.path.join(path_humanm3, 'basketball1', 'images', 'camera_' + str(i), '*.jpeg'))) for i in range(4)]
    # import pdb; pdb.set_trace()
    if use_gt:
        smpl_model = smplx.create(gender='neutral', model_type = 'smpl', \
                          model_path='./smplx_models/')
    for ind, box_j in enumerate(box_json):
        # if box_j != os.path.join(path_humanm3, 'basketball3', 'smpl_calib', '4783.json'):
        #     continue
        scene = box_j.split('/')[-3]
        if scene != 'basketball1':#== 'basketball3' or scene == 'crossdata' or scene == 'basketball1' or scene == 'basketball2':
            continue
        print(box_j)
        pcd_file = box_j.replace('smpl_calib', 'pointcloud').replace('json', 'pcd')
        frame = int(pcd_file.split('/')[-1].split('.')[0])
        indk = frame - 1800
        # for ca_b in camera_bas3_list:
        for ca_b in camera_bas1_list:
            print(ca_b[indk])
        pcd_file = pcd_file.replace(str(frame).zfill(4), str(frame).zfill(6))
        pcd = torch.tensor(np.array(o3d.io.read_point_cloud(pcd_file).points)).cuda()
        
        with open(box_j, 'r') as f:
            smpl_poses = json.load(f)
        meshes = []
        pcd_ = o3d.open3d.geometry.PointCloud()
        pcd_.points= o3d.open3d.utility.Vector3dVector(pcd.cpu().numpy())
        pcd_.paint_uniform_color([0.5,0.5,0.5])
        meshes.append(pcd_)
        if use_gt:
            for key in smpl_poses.keys():
                pose = smpl_poses[key]
                transl = torch.tensor(pose["transl"]).unsqueeze(0)
                betas = torch.tensor(pose["betas"]).unsqueeze(0)
                global_orient = torch.tensor(pose["global_orient"]).unsqueeze(0)
                body_pose = torch.tensor(pose["body_pose"]).unsqueeze(0)
                body_pose = torch.cat([body_pose, torch.zeros(1,6)], dim = 1)
                body_model_output = smpl_model(transl = transl, betas = betas, body_pose = body_pose, global_orient = global_orient)
                vertices = (body_model_output.vertices).detach().squeeze()
                # verts = smpl_model()
                meshes.append(get_mesh(vertices, model.smpl.faces))
        else:
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
                meshes.append(get_mesh(p_k, model.smpl.faces))
        o3d.visualization.draw_geometries(meshes)

args = parse_args()
# setup_seed(10)

update_config(args.cfg)
model = pose_meshgraphormer(pmg_cfg = config).cuda()
if args.state_dict != '':
    state_dict = torch.load(args.state_dict)
    model.load_state_dict(state_dict['net'])
demo_waymo(model)
