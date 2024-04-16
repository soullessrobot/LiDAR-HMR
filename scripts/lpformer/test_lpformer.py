import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from datasets.sloper4d import SLOPER4D_Dataset
from datasets.waymo_v2 import WAYMOV2_Dataset
from datasets.humanm3 import HumanM3_Dataset
from models.lpformer.lpformer import LPFormer
from datasets.lidarh26m import LiDARH26M_Dataset
# from models.v2v_posenet import V2VPoseNet
from tqdm import tqdm
import torch.optim as optim
import logging
import torch.nn.functional as F
# import os
import argparse
from scripts.eval_utils import setup_seed
# from thop import profile, clever_format
# from fvcore.nn import FlopCountAnalysis, parameter_count_table

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--model', default='lpformer', required=False, type=str)
    parser.add_argument(
        '--dataset', default='sloper4d', required=False, type=str)
    parser.add_argument(
        '--state_dict', default='', required=True, type=str)
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
    loss_xyz = AverageMeter()
    loss_vis = AverageMeter()
    loss_seg = AverageMeter()
    PRINT_FREQ = 1000
    for i, sample in enumerate(dataloader):
        for key in sample:
            if type(sample[key]) is not dict and type(sample[key]) is not list:
                sample[key] = sample[key].cuda()
        pcd = sample['human_points']
        center = sample['global_trans']

        # flops = FlopCountAnalysis(model, (pcd,center))
        # print("FLOPs: ", flops.total())
        # import pdb; pdb.set_trace()
        
        # flops, params = profile(model, inputs = (pcd,center,))
        # flops, params = clever_format([flops, params], "%.3f")
        # print(flops, params)
        # import pdb; pdb.set_trace()

        optimizer.zero_grad()
        ret_dict = model(pcd, center)
        loss_dict = model.all_loss(ret_dict, sample)
        all_loss = loss_dict['loss']#loss_dict['loss']
        all_loss.backward()
        optimizer.step()
        loss_xyz.update(loss_dict['xyz_loss'].item())
        loss_vis.update(loss_dict['vis_loss'].item())
        if 'seg_loss' in loss_dict:
            loss_seg.update(loss_dict['seg_loss'].item())
        if i % PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Loss_xyz: {loss_xyz.val:.7f} ({loss_xyz.avg:.7f})\t' \
                  'Loss_vis: {loss_vis.val:.7f} ({loss_vis.avg:.7f})\t' \
                  'Loss_seg: {loss_seg.val:.7f} ({loss_seg.avg:.7f})\t'.format(
                    epoch, i, len(dataloader), loss_xyz=loss_xyz, loss_vis=loss_vis, loss_seg = loss_seg)
            print(msg)
            # logger.info(msg)

def test(model, dataloader):
    model.eval()
    mpjpe = 0
    number = 0
    precision = 0
    recall = 0
    print('============Testing============')
    for sample in tqdm(dataloader):
        for key in sample:
            if type(sample[key]) is not dict and type(sample[key]) is not list:
                sample[key] = sample[key].cuda()
        pcd = sample['human_points']
        gt_xyz = sample['smpl_joints_local']
        center = sample['global_trans']
        gt_vis = sample['vis_label']
        flag = sample['valid_flag'] if 'valid_flag' in sample else None
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            ret_dict = model(pcd, center)
        xyz, vis = ret_dict['xyz'], ret_dict['vis']
        if args.dataset == 'lidarh26m':
            xyz -= xyz[:,[0]]
            gt_xyz -= gt_xyz[:,[0]]
        mpjpe += (xyz - gt_xyz.to(xyz.device)).norm(dim = -1).mean() if flag is None \
              else ((xyz - gt_xyz.to(xyz.device)).norm(dim = -1) * flag).sum() / flag.sum()
        # import pdb; pdb.set_trace()
        vis_result = vis
        vis_tp = vis_result * gt_vis
        precision += torch.sum(vis_tp) / torch.sum(vis_result)
        recall += torch.sum(vis_tp) / torch.sum(gt_vis)
        number += 1
    mpjpe = mpjpe.item() / number    
    precision = precision.item() / number
    recall = recall.item() / number
    # print('MPJPE: '+str(mpjpe))
    return mpjpe, precision, recall
    # print('loss:{:4f}')

args = parse_args()
setup_seed(10)
dataset_task = args.dataset #'sloper4d', 'waymov2', 'collect'
model_type = 'lpformer' #'lpformer', 'v2v'
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
                                fix_pts_num=True, return_smpl = True, augmentation = True, interval = 1)
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
    num_keypoints = 15
    
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
    
model = LPFormer(num_keypoints = num_keypoints).cuda()
bs = 16
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

# optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr = 3e-3) #, betas = (0.85. 0.95)
optimizer = optim.AdamW(model.parameters(), lr = 5e-4, betas = (0.85, 0.95)) #
import datetime
now = datetime.datetime.now()
time_str = now.strftime("%Y-%m-%d %H:%M:%S")

state_dict = torch.load(args.state_dict)
model.load_state_dict(state_dict['net'])
mpjpe, precision, recall = test(model, test_loader)
print(mpjpe, precision, recall)
# for epoch in range(50):
#     train(model, train_loader, optimizer, epoch)
#     state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
#     save_dir = os.path.join('save_state', dataset_task, 'pose', model_type, time_str)
#     os.makedirs(save_dir, exist_ok=True)
    
#     torch.save(state, os.path.join(save_dir, 'epoch'+str(epoch)+'_'+str(mpjpe)+'.pth'))
#     print('MPJPE: '+str(mpjpe) + '; Precision:' + str(precision) + '; Recall:' + str(recall))
