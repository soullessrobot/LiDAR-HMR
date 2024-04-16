import torch
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from torch.utils.data import Dataset, DataLoader
from datasets.sloper4d import SLOPER4D_Dataset
from datasets.waymo_v2 import WAYMOV2_Dataset
from datasets.humanm3 import HumanM3_Dataset
from datasets.lidarh26m import LiDARH26M_Dataset
from models.v2v.v2v_posenet import V2VPoseNet
from tqdm import tqdm
import logging
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
# import os
import json
from scripts.eval_utils import setup_seed

bone_index = [[0,1], [0,2], [1,3], [2,4], [3,5], [4,6], [5,7], \
            [6,8], [0,9], [9,10], [9,11], [9,12], [11,13], [12,14], [13,15], [14,16]]

logger = logging.getLogger(__name__)

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

def test_and_save_sloper(model, dataloader, save_folder = 'sloper4d_v2v_results'):
    print('============Testing============')
    pred_poses = {}
    for sample in tqdm(dataloader):
        pcd = sample['human_points']
        center = sample['global_trans']
        with torch.no_grad():
            ret_dict = model(pcd, center)
        xyz, vis = ret_dict['xyz'], ret_dict['vis']
        xyz = xyz + center.unsqueeze(1)
        scene, frame = sample['location']['scene'][0], int(sample['location']['frame'].cpu().numpy())
        if scene not in pred_poses.keys():
            pred_poses[scene] = {}
        pred_poses[scene][frame] = xyz.cpu().numpy().tolist()
        
    os.makedirs(save_folder, exist_ok = True)
    for scene in pred_poses.keys():
        with open(os.path.join(save_folder, scene + '.json'), 'w') as f:
            json.dump(pred_poses[scene], f, indent=10)
    return

def test_and_save_waymo(model, dataloader, save_folder = 'waymo_v2v_results'):
    print('============Testing============')
    pred_poses = {}
    for sample in tqdm(dataloader):
        pcd = sample['human_points'].cuda()
        center = sample['global_trans'].cuda()
        with torch.no_grad():
            ret_dict = model(pcd, center)
        xyz, vis = ret_dict['xyz'], ret_dict['vis']
        xyz = xyz + center.unsqueeze(1)
        # import pdb; pdb.set_trace()
        split, frame, time, id = sample['location_dict']['split'][0], \
            sample['location_dict']['frame'][0], sample['location_dict']['time'][0], sample['location_dict']['id'][0]
        
        # scene, frame = sample['location']['scene'][0], int(sample['location']['frame'].cpu().numpy())
        # if split not in pred_poses.keys():
        #     pred_poses[split] = {}
        if frame not in pred_poses.keys():
            pred_poses[frame] = {}
        if time not in pred_poses[frame].keys():
            pred_poses[frame][time] = {}
        pred_poses[frame][time][id] = xyz.cpu().numpy().tolist()
        
    os.makedirs(save_folder, exist_ok = True)
    save_text = split
    with open(os.path.join(save_folder, save_text + '.json'), 'w') as f:
        json.dump(pred_poses, f, indent=10)
    return

def test_and_save_humanm3(model, dataloader, save_folder = 'humanm3_v2v_results'):
    print('============Testing============')
    pred_poses = {}
    for sample in tqdm(dataloader):
        pcd = sample['human_points'].cuda()
        center = sample['global_trans'].cuda()
        with torch.no_grad():
            ret_dict = model(pcd, center)
        xyz, vis = ret_dict['xyz'], ret_dict['vis']
        xyz = xyz + center.unsqueeze(1)
        # import pdb; pdb.set_trace()
        split, scene, time, id = sample['location_dict']['split'][0], \
            sample['location_dict']['scene'][0], sample['location_dict']['time'][0], sample['location_dict']['id'][0]
        
        # scene, scene = sample['location']['scene'][0], int(sample['location']['scene'].cpu().numpy())
        # if split not in pred_poses.keys():
        #     pred_poses[split] = {}
        if scene not in pred_poses.keys():
            pred_poses[scene] = {}
        if time not in pred_poses[scene].keys():
            pred_poses[scene][time] = {}
        pred_poses[scene][time][id] = xyz.cpu().numpy().tolist()
        
    os.makedirs(save_folder, exist_ok = True)
    save_text = split
    with open(os.path.join(save_folder, save_text + '.json'), 'w') as f:
        json.dump(pred_poses, f, indent=10)
    return

def test_and_save_lidarh26m(model, dataloader, save_folder = 'lidarh26m_v2v_results', is_train = True):
    print('============Testing============')
    pred_poses = {}
    for sample in tqdm(dataloader):
        pcd = sample['human_points'].cuda()
        center = sample['global_trans'].cuda()
        with torch.no_grad():
            ret_dict = model(pcd, center)
        xyz, vis = ret_dict['xyz'], ret_dict['vis']
        xyz = xyz + center.unsqueeze(1)
        # import pdb; pdb.set_trace()
        # split, scene, time, id = sample['location_dict']['split'][0], \
        #     sample['location_dict']['scene'][0], sample['location_dict']['time'][0], sample['location_dict']['id'][0]
        scene_, time_ = sample['location_dict']['scene'], sample['location_dict']['time']
        id = 0
        # import pdb; pdb.set_trace()
        # scene, scene = sample['location']['scene'][0], int(sample['location']['scene'].cpu().numpy())
        # if split not in pred_poses.keys():
        #     pred_poses[split] = {}
        for indk, (scene, time) in enumerate(zip(scene_, time_)):
            sc, ti = int(scene.cpu()), int(time.cpu())
            if sc not in pred_poses.keys():
                pred_poses[sc] = {}
            if ti not in pred_poses[sc].keys():
                pred_poses[sc][ti] = {}
            pred_poses[sc][ti][id] = xyz[indk].cpu().numpy().tolist()
        
    os.makedirs(save_folder, exist_ok = True)
    save_text = 'train' if is_train else 'test'
    with open(os.path.join(save_folder, save_text + '.json'), 'w') as f:
        json.dump(pred_poses, f, indent=10)
    return

args = parse_args()
setup_seed(10)
dataset_task = args.dataset #'sloper4d', 'waymov2', 'collect'
model_type = args.model #'lpformer', 'v2v'
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
    num_keypoints = 15
    model = V2VPoseNet(num_keypoints = num_keypoints).cuda()
    state_dict = torch.load(args.state_dict)
    model.load_state_dict(state_dict['net'])
    bs = 1
    test_dataset = SLOPER4D_Dataset(dataset_root, scene_test,
                            is_train = False, dataset_path = './save_data/sloper4d',
                            return_torch=False, device = 'cuda',
                            fix_pts_num=True, return_smpl = True, interval= 1, augmentation = False)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
    test_and_save_sloper(model, test_loader)
    train_dataset = SLOPER4D_Dataset(dataset_root, scene_train,
                            is_train = False, dataset_path = './save_data/sloper4d',
                            return_torch=False, device = 'cuda',
                            fix_pts_num=True, return_smpl = True, interval= 1, augmentation = False)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=False)
    test_and_save_sloper(model, train_loader)

elif dataset_task == 'waymov2':
    dataset_root = '/Extra/fanbohao/posedataset/PointC/Waymo/resave_files/'
    train_dataset = WAYMOV2_Dataset(dataset_root, is_train = True, dataset_path = './save_data/waymov2/',
                                return_torch=False, device = 'cuda',
                                fix_pts_num=True, augmentation = False)
    test_dataset = WAYMOV2_Dataset(dataset_root, is_train = False, dataset_path = './save_data/waymov2/',
                                return_torch=False, device = 'cuda',
                                fix_pts_num=True)
    num_keypoints = 15
    model = V2VPoseNet(num_keypoints = num_keypoints).cuda()
    state_dict = torch.load(args.state_dict)
    model.load_state_dict(state_dict['net'])
    bs = 1
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
    test_and_save_waymo(model, test_loader)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=False)
    test_and_save_waymo(model, train_loader)

elif dataset_task == 'humanm3':
    train_dataset = HumanM3_Dataset(is_train = True,
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, augmentation = False, interval = 1)#5)
    test_dataset = HumanM3_Dataset(is_train = False,
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, interval = 1)#5)
    num_keypoints = 15
    model = V2VPoseNet(num_keypoints = num_keypoints).cuda()
    state_dict = torch.load(args.state_dict)
    model.load_state_dict(state_dict['net'])
    bs = 1
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
    test_and_save_humanm3(model, test_loader)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=False)
    test_and_save_humanm3(model, train_loader)

elif dataset_task == 'lidarh26m':
    train_dataset = LiDARH26M_Dataset(is_train = True,
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, augmentation = False, interval = 1)#5)
    test_dataset = LiDARH26M_Dataset(is_train = False,
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, interval = 1)#5)
    num_keypoints = 15
    model = V2VPoseNet(num_keypoints = num_keypoints).cuda()
    state_dict = torch.load(args.state_dict)
    model.load_state_dict(state_dict['net'])
    bs = 16
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
    test_and_save_lidarh26m(model, test_loader, is_train = False)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=False)
    test_and_save_lidarh26m(model, train_loader, is_train = True)

# train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)

