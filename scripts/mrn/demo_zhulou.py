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
        '--state_dict', default='', required=False, type=str)
    parser.add_argument(
        '--cfg', default='configs/mesh/default.yaml', required=False, type=str)
    args, rest = parser.parse_known_args()
    return args

logger = logging.getLogger(__name__)

in_len = 1024
trans_mat = torch.tensor(np.array(
    [
    [[-0.8041653896,-0.5930122412,-0.0406756463,-9.5162420064],
    [0.5931830962,-0.8050168728,0.0090359792,-6.5540031325],
    [-0.0381030278,-0.0168616841,0.9991315443,-2.8273230817],
    [0.0,0.0,0.0,1.0]],
    [[-0.5688457415,0.8223273838,0.0138634862,-14.2211825549],
    [-0.8222375008,-0.5689998783,0.0128308616,10.8492725667],
    [0.0184394908,-0.0041002973,0.9998215704,-2.8251007288],
    [0.0,0.0,0.0,1.0]],
    [[0.8484059872,0.5293419958,0.0020814278,-35.9893552126],
    [-0.5293280926,0.8483381644,0.0115814114,6.0654978474],
    [0.0043647728,-0.010927497,0.9999307669,-2.9019719374],
    [0.0,0.0,0.0,1.0]],
    [[0.5796325479,-0.8148761715,0.0017130232,-31.5360188839],
    [0.8148670668,0.5796343858,0.0039550281,-12.8844605272],
    [-0.0042157853,-0.0008965768,0.9999907116,-2.9513651556],
    [0.0,0.0,0.0,1.0]]
    ]
)).cuda()
def demo_waymo(model):
    model.eval()
    path_zhulou = 'night_'
    annotation_list = sorted(glob.glob(os.path.join(path_zhulou, 'annotations', '*.json')))
    img_list = [sorted(glob.glob(os.path.join(path_zhulou, 'camera', 'camera_'+str(i), '*.jpeg'))) for i in range(4)]
    lidar_list = [sorted(glob.glob(os.path.join(path_zhulou, 'lidar', 'lidar_'+str(i), '*.pcd'))) for i in range(4)]
    # shuffle(box_json)
    for ind, box_j in enumerate(annotation_list):
        with open(box_j, 'r') as f:
            boxes = json.load(f)
        pcd = [torch.tensor(np.array(o3d.io.read_point_cloud(lidar_l[ind]).points)).cuda() for lidar_l in lidar_list]
        # for i in range(len(pcd)):
        #     pck = (trans_mat[i][:3,:3] @ pcd[i].permute(1,0) + trans_mat[i][:3,[3]]).permute(1,0)
        #     pcd[i] = pck
        pcd = torch.cat(pcd, dim = 0)#.cpu().numpy()
        
        # meshes = []
        # pcd_ = o3d.open3d.geometry.PointCloud()
        # pcd_.points= o3d.open3d.utility.Vector3dVector(pcd.cpu().numpy())
        # pcd_.paint_uniform_color([0,0,1])
        # meshes.append(pcd_)
        
        pcd_input = []
        center_in = []
        print([im_l[ind] for im_l in img_list])
        for box_ in boxes['labels']:
            # box_ = box_k['labels']
            w,l,h_ = box_['box3d']['dimension']['width'], box_['box3d']['dimension']['length'], box_['box3d']['dimension']['height']
            x,y,z = box_['box3d']['location']['x'] - 35, box_['box3d']['location']['y'], box_['box3d']['location']['z'] - 3# + h_ / 2
            r_ = max(w, l) / 2
            r_c = pcd - torch.tensor([x,y,z]).unsqueeze(0).cuda()
            ind_in = (r_c[...,:2].norm(dim = -1) < r_) & (r_c[...,2].abs() < h_ / 2)
            pcd_input.append(pcd[ind_in])
            center_in.append(torch.tensor([x,y,z]).unsqueeze(0))
        pcd_ick = []
        
        # pcd_ = o3d.open3d.geometry.PointCloud()
        # pcd_.points= o3d.open3d.utility.Vector3dVector(torch.cat(center_in, dim = 0).numpy())
        # pcd_.paint_uniform_color([1,0,0])
        # meshes.append(pcd_)
        # o3d.visualization.draw_geometries(meshes)
        # import pdb; pdb.set_trace()
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
demo_waymo(model)
