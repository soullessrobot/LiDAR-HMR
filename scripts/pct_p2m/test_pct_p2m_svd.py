import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from datasets.sloper4d import SLOPER4D_Dataset
from datasets.waymo_v2 import WAYMOV2_Dataset
from datasets.humanm3 import HumanM3_Dataset
from models.pose_mesh_net import pose_mesh_net_svd
from datasets.lidarh26m import LiDARH26M_Dataset
from tqdm import tqdm
import torch.optim as optim
import logging
import torch.nn.functional as F
# import os
import argparse
# import torch.distributed as dist
from scripts.eval_utils import mean_per_vertex_error, mean_per_joint_position_error, reconstruction_error, \
    all_gather, setup_seed, mean_per_edge_error, smpl_model, J_regressor, JOINTS_IDX
from models.graphormer.data.config import H36M_J17_NAME, H36M_J17_TO_J14, J24_NAME, J24_TO_J14
# from thop import profile, clever_format
# from fvcore.nn import FlopCountAnalysis, parameter_count_table

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--model', default='lpformer', required=False, type=str)
    parser.add_argument(
        '--dataset', default='sloper4d', required=False, type=str)
    parser.add_argument(
        '--state_dict', default='', required=False, type=str)
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
    loss_vert = AverageMeter()
    loss_joint = AverageMeter()
    PRINT_FREQ = 50
    for i, sample in enumerate(dataloader):
        # pcd = sample['human_points']
        # import pdb; pdb.set_trace()
        for key in sample:
            if type(sample[key]) is not dict and type(sample[key]) is not list:
                sample[key] = sample[key].cuda()
        pcd = sample['human_points_local']
        # pose_input = sample['pose_pred_local']
        optimizer.zero_grad()
        
        # flops = FlopCountAnalysis(model, pose_input)
        # print("FLOPs: ", flops.total() / 1e9)
        # import pdb; pdb.set_trace()
        
        # flops, params = profile(model, inputs = (pose_input,))
        # flops, params = clever_format([flops, params], "%.3f")
        # print(flops, params)
        # import pdb; pdb.set_trace()

        ret_dict = model(pcd)
        loss_dict = model.all_loss(ret_dict, sample)
        all_loss = loss_dict['loss']

        all_loss += loss_dict['edge_loss']
        loss_dict['loss_vert'] += loss_dict['edge_loss']

        all_loss += loss_dict['loss_refine']
        loss_dict['loss_vert'] += loss_dict['loss_refine']

        all_loss.backward()
        optimizer.step()
        loss_all.update(loss_dict['loss'].item())
        loss_joint.update(loss_dict['loss_joint'].item())
        loss_vert.update(loss_dict['loss_vert'].item())
        if i % PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Loss_all: {loss_all.val:.3f} ({loss_all.avg:.3f})\t' \
                  'loss_vert: {loss_vert.val:.3f} ({loss_vert.avg:.3f})\t' \
                    'loss_joint: {loss_joint.val:.3f} ({loss_joint.avg:.3f})\t'.format(
                    epoch, i, len(dataloader), loss_all=loss_all, loss_vert=loss_vert, \
                        loss_joint = loss_joint)
            print(msg)
            # logger.info(msg)

def test(model, dataloader, show = False):
    model.eval()
    mpve = 0
    number = 0
    mpjpe = 0
    mpere = 0
    save = False
    mesh_face = model.p2m_.p2m_.mesh_model.face
    if save:
        saver_dict = {}
        saver_dict['mesh_o'] = []
        saver_dict['mesh_r'] = []
        saver_dict['mesh_gt'] = []
        saver_dict['mpvpe_o'] = []
        saver_dict['mpere_o'] = []
        saver_dict['mpvpe_r'] = []
        saver_dict['mpere_r'] = []
    J_r = smpl_model.J_regressor.cuda()
    print('============Testing============')
    for sample in tqdm(dataloader):
        for key in sample:
            if type(sample[key]) is not dict and type(sample[key]) is not list:
                sample[key] = sample[key].cuda()
        pcd = sample['human_points_local']
        gt_pose = sample['smpl_joints_local']
        num_pose = pcd.shape[0]
        with torch.no_grad():
            ret_dict = model(pcd)
        pred_vertices = ret_dict['mesh_out']
        gt_vertices = sample['smpl_verts_local'].to(pred_vertices.device)
        pose = torch.einsum('bnk,mn->bmk', [ret_dict['mesh_out'], J_regressor])[:,JOINTS_IDX]
        # pose -= pose[:,[0]]
        # gt_pose -= gt_pose[:,[0]]
        if args.dataset == 'lidarh26m':
            pred_vertices -= pose[:,[0]]
            gt_vertices -= gt_pose[:,[0]]
            pose -= pose[:,[0]]
            gt_pose -= gt_pose[:,[0]]
        mpjpe += (pose - gt_pose.to(pose.device)).norm(dim = -1).mean() * num_pose
        seg = ret_dict['seg']
        gt_seg = sample['seg_label'].to(seg.device)

        error_vertices = mean_per_vertex_error(pred_vertices, gt_vertices)
        error_edges, error_relative_edges = mean_per_edge_error(pred_vertices, gt_vertices, mesh_face)
        # import pdb; pdb.set_trace()
        # print(inds, error_vertices, error_relative_edges)
        mpve += error_vertices.mean() * num_pose
        # mpee += error_edges * num_pose
        mpere += error_relative_edges.mean() * num_pose
        number += num_pose
        if save:
            saver_dict['mesh_o'].append(ret_dict['mesh_refine'].detach().cpu().numpy())
            saver_dict['mesh_r'].append(ret_dict['mesh_out'].detach().cpu().numpy())
            saver_dict['mesh_gt'].append(gt_vertices.detach().cpu().numpy())
            saver_dict['mpvpe_r'].append(error_vertices)
            saver_dict['mpere_r'].append(error_relative_edges.mean(dim = -1).numpy())

            error_vertices = mean_per_vertex_error(ret_dict['mesh_out'], gt_vertices)
            error_edges, error_relative_edges = mean_per_edge_error(ret_dict['mesh_out'], gt_vertices, mesh_face)
            saver_dict['mpvpe_o'].append(error_vertices)
            saver_dict['mpere_o'].append(error_relative_edges.mean(dim = -1).numpy())
            # import pdb; pdb.set_trace()
    if save:
        saver_dict['mesh_o'] = np.concatenate(saver_dict['mesh_o'], axis = 0)
        saver_dict['mesh_r'] = np.concatenate(saver_dict['mesh_r'], axis = 0)
        saver_dict['mesh_gt'] = np.concatenate(saver_dict['mesh_gt'], axis = 0)
        saver_dict['mpvpe_r'] = np.concatenate(saver_dict['mpvpe_r'], axis = 0)
        saver_dict['mpere_r'] = np.concatenate(saver_dict['mpere_r'], axis = 0)
        saver_dict['mpvpe_o'] = np.concatenate(saver_dict['mpvpe_o'], axis = 0)
        saver_dict['mpere_o'] = np.concatenate(saver_dict['mpere_o'], axis = 0)
        np.save('saver_results/'+dataset_task+'_p2m.npy', np.array(saver_dict))

    mpve = mpve.item() / number
    mpjpe = mpjpe.item() / number
    mpere = mpere.item() / number    
    return mpve, mpjpe, mpere

args = parse_args()
setup_seed(10)
dataset_task = args.dataset #'sloper4d', 'waymov2', 'collect'
model_type = 'pct_p2m_svd' #'lpformer', 'v2v'
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
                                fix_pts_num=True, return_smpl = True, augmentation = True, interval = 1, load_v2v = False)
    test_dataset = SLOPER4D_Dataset(dataset_root, scene_test, is_train = False, dataset_path = './save_data/sloper4d/',
                                return_torch=False, device = 'cuda',
                                fix_pts_num=True, return_smpl = True, interval = 1, load_v2v = False)
    num_keypoints = 15

elif dataset_task == 'waymov2':
    dataset_root = '/Extra/fanbohao/posedataset/PointC/Waymo/resave_files/'
    train_dataset = WAYMOV2_Dataset(dataset_root, is_train = True, dataset_path = './save_data/waymov2/',
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, augmentation = True, load_v2v = False)
    test_dataset = WAYMOV2_Dataset(dataset_root, is_train = False, dataset_path = './save_data/waymov2/',
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, load_v2v = False)
    num_keypoints = 14

elif dataset_task == 'humanm3':
    train_dataset = HumanM3_Dataset(is_train = True,
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, augmentation = True, interval = 5, load_v2v = False)
    test_dataset = HumanM3_Dataset(is_train = False,
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, interval = 5, load_v2v = False)
    num_keypoints = 15
    # train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
    
elif dataset_task == 'lidarh26m':
    train_dataset = LiDARH26M_Dataset(is_train = True,
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, augmentation = True, interval = 5)
    test_dataset = LiDARH26M_Dataset(is_train = False,
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, interval = 5)
    num_keypoints = 15

model = pose_mesh_net_svd().cuda()
bs = 64
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

optimizer = torch.optim.Adam(params=list(model.parameters()),
                                           lr=1e-3,
                                           betas=(0.9, 0.999),
                                           weight_decay=0)
if args.state_dict != '':
    state_dict = torch.load(args.state_dict)
    model.load_state_dict(state_dict['net'])

import datetime
now = datetime.datetime.now()
time_str = now.strftime("%Y-%m-%d %H:%M:%S")

mpve, mpjpe, mpere = test(model, test_loader)
print('MPJPE: '+str(mpjpe) + '; MPVPE:' + str(mpve) + '; MPERE:'+str(mpere))