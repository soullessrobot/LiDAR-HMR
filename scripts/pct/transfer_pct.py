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
# from thop import profile, clever_format
from models.pct_config import pct_config, update_pct_config

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--target_dataset', default='sloper4d', required=False, type=str)
    parser.add_argument(
        '--source_dataset', default='sloper4d', required=False, type=str)
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
    PRINT_FREQ = 500
    for i, sample in enumerate(dataloader):
        for key in sample:
            if type(sample[key]) is not dict and type(sample[key]) is not list:
                sample[key] = sample[key].cuda()
        pcd = sample['human_points_local']
        # center = sample['global_trans']
        optimizer.zero_grad()
        # flops = FlopCountAnalysis(model, pcd)
        # print("FLOPs: ", flops.total())
        # import pdb; pdb.set_trace()

        # flops, params = profile(model, inputs = (pcd,))
        # flops, params = clever_format([flops, params], "%.3f")
        # print(flops, params)
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
    precision = torch.tensor(0.0).cuda()
    print('============Testing============')
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
        mpjpe += (pose - gt_pose.to(pose.device)).norm(dim = -1).mean() 
        if 'seg' in ret_dict:
            seg = ret_dict['seg']
            gt_seg = sample['seg_label'].to(seg.device)
            precision += (seg.argmax(dim = 1) == gt_seg).sum() / gt_seg.numel() 
        number += 1
    mpjpe = mpjpe.item() / number    
    precision = precision.item() / number
    return mpjpe, precision

def run_validate(val_loader, Graphormer_model):
    batch_time = AverageMeter()
    mPVE = AverageMeter()
    mPJPE = AverageMeter()
    PAmPJPE = AverageMeter()
    # switch to evaluate mode
    Graphormer_model.eval()
    # smpl.eval()
    PRINT_FREQ = 100
    print('Testing:')
    JOINTS_IDX = [8, 5, 29, 30, 4, 7, 21, 19, 17, 16, 18, 20, 31, 32, 33, 34, 35, 36, 37, 24, 26, 25, 28, 27]
    with torch.no_grad():
        for i, sample in tqdm(enumerate(val_loader)):
            # if i <= 251:
            #     continue
            # print(i)
            has_3d_joints = sample['has_3d_joints'].cuda()
            has_smpl = sample['has_smpl'].cuda()
            pcd = sample['human_points']
            ret_dict = model(pcd)
            pred_shape, pred_pose, pred_trans, pred_R6D_3D = \
            ret_dict['pred_shape'], ret_dict['pred_pose'], ret_dict['pred_trans'], ret_dict['pred_R6D_3D']
            pred_smpl = smpl_model(betas = pred_shape, body_pose = pred_pose, global_orient = pred_R6D_3D, transl = pred_trans)
            
            gt_shape, gt_pose, gt_trans, gt_joints = sample['betas'], sample['smpl_pose'], sample['global_trans'], sample['smpl_joints']            
            gt_smpl = smpl_model(betas = gt_shape, body_pose = gt_pose[...,3:], global_orient = gt_pose[...,:3], transl = gt_trans)
            pred_vertices = pred_smpl.vertices
            gt_vertices = gt_smpl.vertices

            pred_3d_joints_from_smpl = pred_smpl.joints[:,JOINTS_IDX]
            gt_3d_joints = gt_smpl.joints[:,JOINTS_IDX]
            # measure errors
            error_vertices = mean_per_vertex_error(pred_vertices, gt_vertices, has_smpl)
            error_joints = mean_per_joint_position_error(pred_3d_joints_from_smpl, gt_3d_joints,  has_3d_joints)
            error_joints_pa = reconstruction_error(pred_3d_joints_from_smpl.cpu().numpy(), gt_3d_joints[:,:,:3].cpu().numpy(), reduction=None)
            
            if len(error_vertices)>0:
                mPVE.update(np.mean(error_vertices), int(torch.sum(has_smpl)) )
            if len(error_joints)>0:
                mPJPE.update(np.mean(error_joints), int(torch.sum(has_3d_joints)) )
            if len(error_joints_pa)>0:
                PAmPJPE.update(np.mean(error_joints_pa), int(torch.sum(has_3d_joints)) )
            # if i % PRINT_FREQ == 0:
            #     val_mPVE = all_gather(float(mPVE.avg))
            #     val_mPVE = sum(val_mPVE)/len(val_mPVE)
            #     val_mPJPE = all_gather(float(mPJPE.avg))
            #     val_mPJPE = sum(val_mPJPE)/len(val_mPJPE)
            #     val_PAmPJPE = all_gather(float(PAmPJPE.avg))
            #     val_PAmPJPE = sum(val_PAmPJPE)/len(val_PAmPJPE)
            #     print(val_mPVE, val_mPJPE, val_PAmPJPE)

    val_mPVE = all_gather(float(mPVE.avg))
    val_mPVE = sum(val_mPVE)/len(val_mPVE)
    val_mPJPE = all_gather(float(mPJPE.avg))
    val_mPJPE = sum(val_mPJPE)/len(val_mPJPE)

    val_PAmPJPE = all_gather(float(PAmPJPE.avg))
    val_PAmPJPE = sum(val_PAmPJPE)/len(val_PAmPJPE)

    val_count = all_gather(float(mPVE.count))
    val_count = sum(val_count)

    return val_mPVE, val_mPJPE, val_PAmPJPE, val_count

args = parse_args()
setup_seed(10)
dataset_task = args.target_dataset #'sloper4d', 'waymov2', 'collect'
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

train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers = 4)
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers = 4)
    
config_name = args.cfg.split('/')[-1].split('.')[0]
update_pct_config(args.cfg)
model = PCTv2_SegReg(pct_config).cuda()

# total = sum([param.nelement() for param in model.parameters()])
# print(parameter_count_table(model))
# import pdb; pdb.set_trace()

# optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr = 3e-3) #, betas = (0.85. 0.95)
# optimizer = optim.AdamW(model.parameters(), lr = 1e-4, betas = (0.85, 0.95)) #
# optimizer = optim.SGD(model.parameters(), lr = 1e-4)

optimizer = torch.optim.Adam(params=list(model.parameters()),
                                           lr=5e-4,
                                           betas=(0.9, 0.999))

source_ = args.source_dataset
target_ = args.target_dataset

if args.state_dict != '':
    state_dict = torch.load(args.state_dict)
    model.load_state_dict(state_dict['net'])
    optimizer.load_state_dict(state_dict['optimizer'])
    # import pdb; pdb.set_trace()

import datetime
now = datetime.datetime.now()
time_str = now.strftime("%Y-%m-%d %H:%M:%S")

all_epochs = 50 #50
for epoch in range(all_epochs):
    train(model, train_loader, optimizer, epoch)
    state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
    # save_dir = os.path.join('save_state', dataset_task, 'pose', model_type, time_str)
    save_dir = os.path.join('save_state', source_ + '_' + target_, 'pose', model_type, time_str)
    os.makedirs(save_dir, exist_ok=True)
    mpjpe, precision = test(model, test_loader)
    torch.save(state, os.path.join(save_dir, 'epoch'+str(epoch)+'_'+str(mpjpe)+'.pth'))
    print('MPJPE: '+str(mpjpe) + '; Precision:' + str(precision) + ';')
    