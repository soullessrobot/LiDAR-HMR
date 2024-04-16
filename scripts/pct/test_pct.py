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
from models.graphormer.graphormer_model import graphormer_model
# from models.unsupervised.Network import point_net_ssg, smpl_model
from models.pct.pct_pose import PCTv2_SegReg
# from models.v2v_posenet import V2VPoseNet
from tqdm import tqdm
import torch.optim as optim
import logging
import torch.nn.functional as F
import argparse
# import torch.distributed as dist
from scripts.eval_utils import mean_per_vertex_error, mean_per_joint_position_error, reconstruction_error, all_gather, setup_seed
from models.graphormer.data.config import H36M_J17_NAME, H36M_J17_TO_J14, J24_NAME, J24_TO_J14
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from models.pct_config import pct_config, update_pct_config
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--model', default='lpformer', required=False, type=str)
    parser.add_argument(
        '--dataset', default='sloper4d', required=False, type=str)
    parser.add_argument(
        '--state_dict', default='', required=False, type=str)
    parser.add_argument(
        '--cfg', default='configs/pose/default.yaml', required=False, type=str)
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
    loss_seg = AverageMeter()
    loss_joint = AverageMeter()
    loss_prior = AverageMeter()
    PRINT_FREQ = 50
    for i, sample in enumerate(dataloader):
        pcd = sample['human_points_local']
        # center = sample['global_trans']
        # optimizer.zero_grad()
        # flops = FlopCountAnalysis(model, pcd)
        # print("FLOPs: ", flops.total())
        
        ret_dict = model(pcd)
        loss_dict = model.all_loss(ret_dict, sample)
        all_loss = loss_dict['loss']#loss_dict['loss']
        all_loss.backward()
        optimizer.step()
        loss_all.update(loss_dict['loss'].item())
        loss_joint.update(loss_dict['loss_joint'].item())
        loss_seg.update(loss_dict['loss_seg'].item())
        if 'loss_prior' in loss_dict.keys():
            loss_prior.update(loss_dict['loss_prior'].item())
        if i % PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Loss_all: {loss_all.val:.3f} ({loss_all.avg:.3f})\t' \
                  'loss_seg: {loss_seg.val:.3f} ({loss_seg.avg:.3f})\t' \
                    'loss_joint: {loss_joint.val:.3f} ({loss_joint.avg:.3f})\t' \
                    'loss_prior: {loss_prior.val:.3f} ({loss_prior.avg:.3f})'.format(
                    epoch, i, len(dataloader), loss_all=loss_all, loss_seg=loss_seg, \
                        loss_joint = loss_joint, loss_prior = loss_prior)
            print(msg)
            # logger.info(msg)

def test(model, dataloader):
    model.eval()
    mpjpe = 0
    number = 0
    pa_mpjpe = 0
    print('============Testing============')
    show = False
    for sample in tqdm(dataloader):
        for key in sample:
            if type(sample[key]) is not dict and type(sample[key]) is not list:
                sample[key] = sample[key].cuda()
        pcd = sample['human_points_local']
        if model.pose_dim == 24:
            sample['smpl_joints_local'] = sample['smpl_joints24_local']
        gt_pose = sample['smpl_joints_local']
        
        with torch.no_grad():
            ret_dict = model(pcd)
        pose = ret_dict['pose']
        if args.dataset == 'lidarh26m':
            pose -= pose[:,[0]]
            gt_pose -= gt_pose[:,[0]]
        mpjpe += (pose - gt_pose.to(pose.device)).norm(dim = -1).mean() 
        err_pa = reconstruction_error(pose.cpu().numpy(), gt_pose.cpu().numpy(), reduction=None)
        pa_mpjpe += np.mean(err_pa[err_pa < 10])
        number += 1
        # import pdb; pdb.set_trace()
        if show:
            print((pose - gt_pose.to(pose.device)).norm(dim = -1).mean() )
            bone_link = np.array([[0,1], [0,2], [1,3], [2,4], [3,5], [4,6], [0,7], [7,8], [7,9], [7,10], [9,11], [10,12], [11,13], [12,14]])
            pose_show, gt_show, pcd_show = pose[0].detach().cpu().numpy(), gt_pose[0].detach().cpu().numpy(), pcd[0].detach().cpu().numpy()
            ax = plt.figure().add_subplot(111, projection = '3d')
            ax.scatter(pcd_show[:,0], pcd_show[:,1], pcd_show[:,2], c = 'b')
            for bl in bone_link:
                ax.plot(pose_show[bl,0], pose_show[bl,1], pose_show[bl,2], c = 'r')
                ax.plot(gt_show[bl,0], gt_show[bl,1], gt_show[bl,2], c = 'g')
            plt.show()
    mpjpe = mpjpe.item() / number    
    pa_mpjpe = pa_mpjpe.item() / number
    return mpjpe, pa_mpjpe

args = parse_args()
setup_seed(10)
dataset_task = args.dataset #'sloper4d', 'waymov2', 'collect'
model_type = 'pct' #'lpformer', 'v2v'
bs = 64

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
    # train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

elif dataset_task == 'waymov2':
    dataset_root = '/Extra/fanbohao/posedataset/PointC/Waymo/resave_files/'
    train_dataset = WAYMOV2_Dataset(dataset_root, is_train = True, dataset_path = './save_data/waymov2/',
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, augmentation = True)
    test_dataset = WAYMOV2_Dataset(dataset_root, is_train = False, dataset_path = './save_data/waymov2/',
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True)
    num_keypoints = 15
    # train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

elif dataset_task == 'humanm3':
    train_dataset = HumanM3_Dataset(is_train = True,
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, augmentation = True, interval = 5)
    test_dataset = HumanM3_Dataset(is_train = False,
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, interval = 5)
    num_keypoints = 15
    # train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

elif dataset_task == 'cimi4d':
    train_dataset = CIMI4D_Dataset(is_train = True,
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, augmentation = True, interval = 1)
    test_dataset = CIMI4D_Dataset(is_train = False,
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, interval = 1)
    num_keypoints = 15

elif dataset_task == 'lidarh26m':
    train_dataset = LiDARH26M_Dataset(is_train = True,
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, augmentation = True, interval = 5)
    test_dataset = LiDARH26M_Dataset(is_train = False,
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, interval = 5)
    num_keypoints = 15

train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=True)

update_pct_config(args.cfg)
model = PCTv2_SegReg(pct_config).cuda()
state_dict = torch.load(args.state_dict)
model.load_state_dict(state_dict['net'])
mpjpe, pa_mpjpe = test(model, test_loader)
print('MPJPE: '+str(mpjpe) + '; PA-MPJPE:' + str(pa_mpjpe) + ';')
