import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from datasets.sloper4d import SLOPER4D_Dataset
from datasets.waymo_v2 import WAYMOV2_Dataset
from datasets.humanm3 import HumanM3_Dataset
from datasets.cimi4d import CIMI4D_Dataset
from datasets.lidarh26m import LiDARH26M_Dataset
# from models.graphormer.graphormer_model import graphormer_model
# from models.unsupervised.Network import point_net_ssg, smpl_model
from models.pose_mesh_net import votehmr
# from models.v2v_posenet import V2VPoseNet
from tqdm import tqdm
import torch.optim as optim
import logging
import torch.nn.functional as F
import argparse
# import torch.distributed as dist
from scripts.eval_utils import mean_per_vertex_error, setup_seed, mean_per_edge_error, smpl_model
from models.graphormer.data.config import H36M_J17_NAME, H36M_J17_TO_J14, J24_NAME, J24_TO_J14
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from models.pct_config import pct_config, update_pct_config

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--model', default='lpformer', required=False, type=str)
    parser.add_argument(
        '--dataset', required=True, type=str)
    parser.add_argument(
        '--state_dict', default='', required=False, type=str)
    parser.add_argument(
        '--pct_state_dict', default='', required=False, type=str)
    parser.add_argument(
        '--pct_config', default='configs/pose/pose_24.yaml', required=False, type=str)
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

def train(model, dataloader, optimizer, epoch):
    model.train()
    loss_all = AverageMeter()
    PRINT_FREQ = 100#500
    # import pdb; pdb.set_trace()
    for i, sample in enumerate(dataloader):
        for key in sample:
            if type(sample[key]) is not dict and type(sample[key]) is not list:
                sample[key] = sample[key].cuda()
        pcd = sample['human_points_local']
        # center = sample['global_trans']
        optimizer.zero_grad()

        # flops = FlopCountAnalysis(model, pcd)
        # print("FLOPs: ", flops.total() / 1e9)
        # import pdb; pdb.set_trace()
        # flops, params = profile(model, inputs = (pcd,))
        # flops, params = clever_format([flops, params], "%.3f")
        # print(flops, params)
        # import pdb; pdb.set_trace()

        ret_dict = model(pcd)
        loss_dict = model.all_loss(ret_dict, sample)
        all_loss = loss_dict['loss']
        all_loss.backward()
        optimizer.step()
        loss_all.update(loss_dict['loss'].item())
        # print(loss_dict['loss_vert'])
        if i % PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Loss_all: {loss_all.val:.3f} ({loss_all.avg:.3f})'.format(
                    epoch, i, len(dataloader), loss_all=loss_all)
            print(msg)
            # logger.info(msg)

def test(model, dataloader):
    model.eval()
    mpjpe = 0
    mpvpe = 0
    mpere = 0
    # mpvpe_smpl = 0
    number = 0
    # precision = 0
    mesh_face = smpl_model.faces
    J_r = smpl_model.J_regressor.cuda()
    print('============Testing============')
    for sample in tqdm(dataloader):
        pcd = sample['human_points_local'].cuda()
        pose_gt, shape_gt = sample['smpl_pose'].cuda(), sample['betas'].cuda()
        with torch.no_grad():
            ret_dict = model(pcd)
        pose = ret_dict['joints']

        gt_vertices = sample['smpl_verts_local'].to(pose.device) #gt_smpl.vertices
        gt_pose = sample['smpl_joints24_local'].to(pose.device) #gt_smpl.joints[:,:24]

        pose -= pose[:,[0]]
        gt_pose -= gt_pose[:,[0]]

        mpjpe += (pose - gt_pose.to(pose.device)).norm(dim = -1).mean() 
        
        pred_vertices = ret_dict['mesh_out']

        # gt_root = torch.einsum('ji,bik->bjk', [J_r, gt_vertices])[:,[0]]
        # pred_root = torch.einsum('ji,bik->bjk', [J_r, pred_vertices])[:,[0]]
        # pred_vertices -= pred_root
        # gt_vertices -= gt_root

        error_vertices = mean_per_vertex_error(pred_vertices, gt_vertices)
        error_edges, error_relative_edges = mean_per_edge_error(pred_vertices, gt_vertices, mesh_face)
        mpvpe += error_vertices.mean()
        mpere += error_relative_edges.mean()
        number += 1
    mpjpe = mpjpe.item() / number    
    mpvpe = mpvpe.item() / number
    mpere = mpere.item() / number
    return mpjpe, mpvpe, mpere#, mpvpe_smpl
    # print('loss:{:4f}')

args = parse_args()
setup_seed(10)
dataset_task = args.dataset #'sloper4d', 'waymov2', 'collect'
model_type = 'votehmr' #'lpformer', 'v2v'
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
    train_dataset = SLOPER4D_Dataset(dataset_root, scene_train, is_train = True, dataset_path = './save_data/sloper4d/',
                                return_torch=False, device = 'cuda',
                                fix_pts_num=True, return_smpl = True, augmentation = True, interval = 1)
    test_dataset = SLOPER4D_Dataset(dataset_root, scene_test, is_train = False, dataset_path = './save_data/sloper4d/',
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
    num_keypoints = 15

elif dataset_task == 'humanm3':
    train_dataset = HumanM3_Dataset(is_train = True,
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, augmentation = True, interval = 5)
    test_dataset = HumanM3_Dataset(is_train = False,
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, interval = 5)
    num_keypoints = 15

elif dataset_task == 'cimi4d':
    train_dataset = CIMI4D_Dataset(is_train = True,
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, augmentation = True, interval = 5)
    test_dataset = CIMI4D_Dataset(is_train = False,
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

update_pct_config(args.pct_config)
model = votehmr(pct_config).cuda()

bs = 32
# bs = 1
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last = True)
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
import pdb; pdb.set_trace()
optimizer = torch.optim.Adam(params=list(model.parameters()),
                                           lr=1e-3,
                                           betas=(0.9, 0.999),
                                           weight_decay=0)
if args.pct_state_dict != '':
    state_dict = torch.load(args.pct_state_dict)
    # import pdb; pdb.set_trace()
    model.pct_pose.load_state_dict(state_dict['net'])

import datetime
now = datetime.datetime.now()
time_str = now.strftime("%Y-%m-%d %H:%M:%S")

if args.state_dict != '':
    state_dict = torch.load(args.state_dict)
    model.load_state_dict(state_dict['net'])
mpjpe, mpvpe, mpere = test(model, test_loader)
print('MPJPE: '+str(mpjpe) + '; MPVPE:'+str(mpvpe) + '; MPERE:'+str(mpere))
