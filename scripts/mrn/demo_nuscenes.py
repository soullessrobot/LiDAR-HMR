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

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--state_dict', default='', required=False, type=str)
    parser.add_argument(
        '--cfg', default='configs/mesh/default.yaml', required=False, type=str)
    args, rest = parser.parse_known_args()
    return args

logger = logging.getLogger(__name__)

in_len = 1024

def demo_waymo(model):
    model.eval()
    pcd = np.load('nuscenes/nusc_8408/points.npy')[:,:3]
    with open('nuscenes/nusc_8408/info.pkl', 'rb') as f:
        bboxes = pickle.load(f)
    bbox = bboxes['gt_boxes']
    names = bboxes['gt_names']
    pcd_input = []
    center_in = []
    for bb, na_ in zip(bbox, names):
        if na_ != 'pedestrian':
            continue
        
        center_, size_ = bb[:3], bb[3:6]
        r_ = max(size_[0], size_[1])
        h_ = size_[2]
        # import pdb; pdb.set_trace()
        pcd = torch.tensor(pcd).cuda()
        r_c = pcd - torch.tensor(center_).unsqueeze(0).cuda()
        ind_in = (r_c[...,:2].norm(dim = -1) < r_) & (r_c[...,2].abs() < h_ / 2)
        # print(pcd[ind_in].shape)
        if pcd[ind_in].shape[0] > 15:
            pcd_input.append(pcd[ind_in])
            center_in.append(torch.tensor(center_).unsqueeze(0))
    # import pdb; pdb.set_trace()
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
    meshes = []
    pcd_ = o3d.open3d.geometry.PointCloud()
    pcd_.points= o3d.open3d.utility.Vector3dVector(pcd.cpu().numpy())
    pcd_.paint_uniform_color([0.5,0.5,0.5])
    meshes.append(pcd_)
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

# import pdb; pdb.set_trace()
demo_waymo(model)
