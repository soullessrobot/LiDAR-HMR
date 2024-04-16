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
        '--dataset', default='sloper4d', required=True, type=str)
    # parser.add_argument(
    #     '--source_dataset', default='sloper4d', required=False, type=str)
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
    pa_mpjpe = 0
    number = 0
    rc_15 = 0
    # map = torch.tensor(0.0).cuda()
    ap = [0,0,0,0] #75 100 125 150
    threshold = [0.075, 0.1, 0.125, 0.15]
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
        bs = pose.shape[0]
        diff = (pose - gt_pose.to(pose.device)).norm(dim = -1)
        mpjpe += diff.mean() * bs
        pa_mpjpe += reconstruction_error(pose.cpu().numpy(), gt_pose.cpu().numpy()) * bs
        rc_15 += (diff < 0.15).sum() / torch.numel(diff) * bs
        for ind, thre in enumerate(threshold):
            ap[ind] += (diff < thre).sum() / torch.numel(diff) * bs
        number += bs
    mpjpe = mpjpe.item() / number    
    # precision = precision.item() / number
    pa_mpjpe = pa_mpjpe.item() / number
    mAP_75_125 = (ap[0] + ap[1] + ap[2]) / 3
    mAP_75_125 = mAP_75_125.item() / number
    recall = rc_15.item() / number
    for ind, ap_ in enumerate(ap):
        ap[ind] = ap_.item() / number
    return mpjpe, pa_mpjpe, mAP_75_125, recall, ap

args = parse_args()
setup_seed(10)
dataset_task = args.dataset #'sloper4d', 'waymov2', 'collect'
model_type = 'pct' #'lpformer', 'v2v'
bs = 64
# import pdb; pdb.set_trace()
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

# source_ = args.source_dataset
# target_ = args.target_dataset

if args.state_dict != '':
    state_dict = torch.load(args.state_dict)
    model.load_state_dict(state_dict['net'])
    # optimizer.load_state_dict(state_dict['optimizer'])
    # import pdb; pdb.set_trace()

mpjpe, pa_mpjpe, mAP, recall, ap = test(model, test_loader)
print(mpjpe, pa_mpjpe, mAP, recall, ap)
# all_epochs = 50 #50
# for epoch in range(all_epochs):
#     train(model, train_loader, optimizer, epoch)
#     state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
#     # save_dir = os.path.join('save_state', dataset_task, 'pose', model_type, time_str)
#     save_dir = os.path.join('save_state', source_ + '_' + target_, 'pose', model_type, time_str)
#     os.makedirs(save_dir, exist_ok=True)
#     mpjpe, precision = test(model, test_loader)
#     torch.save(state, os.path.join(save_dir, 'epoch'+str(epoch)+'_'+str(mpjpe)+'.pth'))
#     print('MPJPE: '+str(mpjpe) + '; Precision:' + str(precision) + ';')
    