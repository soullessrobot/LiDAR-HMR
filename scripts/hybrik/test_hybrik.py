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
from datasets.lidarh26m import LiDARH26M_Dataset
# from models.graphormer.graphormer_model import graphormer_model
# from models.unsupervised.Network import point_net_ssg, smpl_model
# from models.pose_mesh_net import pose_mesh_net, pose_meshgraphormer
from models.pose_mesh_net import pose_mesh_hybrik
# from models.v2v_posenet import V2VPoseNet
from tqdm import tqdm
import torch.optim as optim
import logging
import torch.nn.functional as F
import argparse
# import torch.distributed as dist
from scripts.eval_utils import mean_per_vertex_error, setup_seed, get_mesh, mean_per_edge_error, smpl_model
from models.graphormer.data.config import H36M_J17_NAME, H36M_J17_TO_J14, J24_NAME, J24_TO_J14
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import matplotlib.pyplot as plt
from models._smpl import SMPL
import smplx
from models.pmg_config import config, update_config
from models.pct_config import pct_config, update_pct_config

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--model', default='lpformer', required=False, type=str)
    parser.add_argument(
        '--dataset', default='sloper4d', required=False, type=str)
    parser.add_argument(
        '--state_dict', default='', required=False, type=str)
    parser.add_argument(
        '--cfg', default='configs/mesh/default.yaml', required=False, type=str)
    args, rest = parser.parse_known_args()
    return args

logger = logging.getLogger(__name__)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_areas(verts, faces):
    point1 = verts[faces[:,0]]
    point2 = verts[faces[:,1]]
    point3 = verts[faces[:,2]]
    v1 = point2 - point1
    v2 = point3 - point1
    area = np.linalg.norm(np.cross(v1, v2), axis = -1) / 2
    return area

def get_edges(verts, faces):
    point1 = verts[faces[:,0]]
    point2 = verts[faces[:,1]]
    point3 = verts[faces[:,2]]
    v1 = point2 - point1
    v2 = point3 - point1
    v3 = point3 - point2
    len1, len2, len3 = np.linalg.norm(v1, axis = -1), np.linalg.norm(v2, axis = -1), np.linalg.norm(v3, axis = -1) 
    edges = np.stack([len1, len2, len3], axis = 1)
    return edges

def train(model, dataloader, optimizer, epoch):
    model.train()
    loss_all = AverageMeter()
    loss_seg = AverageMeter()
    loss_joint = AverageMeter()
    loss_vert = AverageMeter()
    PRINT_FREQ = 50
    for i, sample in enumerate(dataloader):
        pcd = sample['human_points_local']
        # center = sample['global_trans']
        optimizer.zero_grad()
        # flops = FlopCountAnalysis(model, pcd)
        # print("FLOPs: ", flops.total())
        # import pdb; pdb.set_trace()

        ret_dict = model(pcd)
        # import pdb; pdb.set_trace()
        loss_dict = model.all_loss(ret_dict, sample)
        all_loss = loss_dict['loss']#loss_dict['loss']
        # import pdb; pdb.set_trace()
        all_loss.backward()
        optimizer.step()
        loss_all.update(loss_dict['loss'].item())
        loss_joint.update(loss_dict['loss_joint'].item())
        loss_seg.update(loss_dict['loss_seg'].item())
        # if 'loss_vert' in loss_dict.keys():
        # import pdb; pdb.set_trace()
        loss_vert.update(loss_dict['loss_vert'].item())
        if i % PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Loss_all: {loss_all.val:.3f} ({loss_all.avg:.3f})\t' \
                  'loss_seg: {loss_seg.val:.3f} ({loss_seg.avg:.3f})\t' \
                    'loss_joint: {loss_joint.val:.3f} ({loss_joint.avg:.3f})\t' \
                    'loss_vert: {loss_vert.val:.3f} ({loss_vert.avg:.3f})'.format(
                    epoch, i, len(dataloader), loss_all=loss_all, loss_seg=loss_seg, \
                        loss_joint = loss_joint, loss_vert = loss_vert)
            print(msg)
            # logger.info(msg)

def test(model, dataloader):
    model.eval()
    mpjpe = 0
    mpvpe = 0
    mpee = 0
    mpere = 0
    number = 0
    precision = 0
    print('============Testing============')
    mesh_face = model.mesh_faces
    J_r = smpl_model.J_regressor.cuda()
    for sample in tqdm(dataloader):
        for key in sample:
            if type(sample[key]) is not dict and type(sample[key]) is not list:
                sample[key] = sample[key].cuda()
        pcd = sample['human_points_local']
        # gt_pose = sample['smpl_joints_local']
        gt_pose = sample['smpl_joints24_local'] if model.pct_pose.pose_dim == 24 else sample['smpl_joints_local'] 
        
        with torch.no_grad():
            ret_dict = model(pcd)
        pose = ret_dict['pose']
        pose -= pose[:,[0]]
        gt_pose -= gt_pose[:,[0]]

        mpjpe += (pose - gt_pose.to(pose.device)).norm(dim = -1).mean() 
        seg = ret_dict['seg']
        gt_seg = sample['seg_label'].to(seg.device)
        precision += (seg.argmax(dim = 1) == gt_seg).sum() / gt_seg.numel() 
        pred_vertices = ret_dict['mesh_out']
        gt_vertices = sample['smpl_verts_local'].to(pred_vertices.device)

        # gt_root = torch.einsum('ji,bik->bjk', [J_r, gt_vertices])[:,[0]]
        # pred_root = torch.einsum('ji,bik->bjk', [J_r, pred_vertices])[:,[0]]
        # pred_vertices -= pred_root
        # gt_vertices -= gt_root

        error_vertices = mean_per_vertex_error(pred_vertices, gt_vertices)
        # error_edges = mean_per_edge_error(pred_vertices, gt_vertices, mesh_face).mean()
        error_edges, error_relative_edges = mean_per_edge_error(pred_vertices, gt_vertices, mesh_face)
        # error_vertices_smpl = mean_per_vertex_error(pred_vertices_smpl, gt_vertices).mean()
        mpvpe += error_vertices.mean()
        mpee += error_edges
        mpere += error_relative_edges.mean()
        number += 1
    mpjpe = mpjpe.item() / number    
    precision = precision.item() / number
    mpvpe = mpvpe.item() / number
    mpee = mpee.item() / number
    mpere = mpere.item() / number
    # print('MPJPE: '+str(mpjpe))
    return mpjpe, precision, mpvpe, mpee, mpere
    # print('loss:{:4f}')

args = parse_args()
setup_seed(10)
dataset_task = args.dataset #'sloper4d', 'waymov2', 'collect'
model_type = 'pct_mf' #'lpformer', 'v2v'
if dataset_task == 'sloper4d':
    scene_train = [
            'seq002_football_001',
            'seq003_street_002',
            'seq005_library_002',
            'seq007_garden_001',
            'seq008_running_001'
        ]
    scene_test = ['seq009_running_002']
    dataset_root = '/Extra/fanbohao/posedataset/PointC/sloper4d/'
    train_dataset = SLOPER4D_Dataset(dataset_root, scene_train, is_train = True, dataset_path = './save_data/sloper4d',
                                return_torch=False, device = 'cuda',
                                fix_pts_num=True, return_smpl = True, augmentation = False, interval = 1)
    test_dataset = SLOPER4D_Dataset(dataset_root, scene_test, is_train = False, dataset_path = './save_data/sloper4d',
                                return_torch=False, device = 'cuda',
                                fix_pts_num=True, return_smpl = True, interval = 1)
    num_keypoints = 15

elif dataset_task == 'waymov2':
    dataset_root = '/Extra/fanbohao/posedataset/PointC/Waymo/resave_files/'
    train_dataset = WAYMOV2_Dataset(dataset_root, is_train = True, dataset_path = './save_data/waymov2/',
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, augmentation = True)
    test_dataset = WAYMOV2_Dataset(dataset_root, is_train = False, dataset_path = './save_data/waymov2/',
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True)
    num_keypoints = 14

elif dataset_task == 'humanm3':
    train_dataset = HumanM3_Dataset(is_train = True,
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, augmentation = True, interval = 5)
    test_dataset = HumanM3_Dataset(is_train = False,
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, interval = 5)
    num_keypoints = 15

elif dataset_task == 'lidarh26m':
    train_dataset = LiDARH26M_Dataset(is_train = True,
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, augmentation = True, interval = 5)
    test_dataset = LiDARH26M_Dataset(is_train = False,
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, interval = 5)
    num_keypoints = 15
# import pdb; pdb.set_trace()
# model = pose_meshformer().cuda()
update_config(args.cfg)
update_pct_config('configs/pose/pose_24.yaml')
model = pose_mesh_hybrik(pct_config).cuda()

# total = sum([param.nelement() for param in model.parameters()])
# print(parameter_count_table(model))
# import pdb; pdb.set_trace()
# bs = 8
show = False
bs = 16 if not show else 1

train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

# optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr = 3e-3) #, betas = (0.85. 0.95)
# optimizer = optim.AdamW(model.parameters(), lr = 1e-4, betas = (0.85, 0.95)) #
# optimizer = optim.SGD(model.parameters(), lr = 1e-4)
if args.state_dict != '':
    state_dict = torch.load(args.state_dict)
    model.load_state_dict(state_dict['net'])

optimizer = torch.optim.Adam(params=list(model.parameters()),
                                           lr=1e-4,
                                           betas=(0.9, 0.999),
                                           weight_decay=0)
import datetime
now = datetime.datetime.now()
time_str = now.strftime("%Y-%m-%d %H:%M:%S")
save = False

mpjpe, precision, mpvpe, mpee, mpere = test(model, test_loader)
print('MPJPE: '+str(mpjpe) + '; MPEE:' + str(mpee) + '; MPVPE:'+str(mpvpe) + '; MPERE:'+str(mpere))
