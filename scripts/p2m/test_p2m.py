import open3d as o3d
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from datasets.sloper4d import SLOPER4D_Dataset
from datasets.waymo_v2 import WAYMOV2_Dataset
from datasets.humanm3 import HumanM3_Dataset
from models.pose2mesh.pointcloud2mesh_net import pointcloud2mesh_net
from models.pose_mesh_net import pose_mesh_net
from tqdm import tqdm
import torch.optim as optim
import logging

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
import argparse
import matplotlib.pyplot as plt
# import torch.distributed as dist
from scripts.eval_utils import mean_per_vertex_error, mean_per_joint_position_error, reconstruction_error, all_gather, setup_seed, get_mesh, mean_per_edge_error
from models.graphormer.data.config import H36M_J17_NAME, H36M_J17_TO_J14, J24_NAME, J24_TO_J14
from models.pose2mesh.keypoints_config import COLLECTION_15_NAME, skeleton_15

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

def config_fig(ax, center):
    ax.set_xlim(center[0] - 1.0, center[0] + 1.0)
    ax.set_ylim(center[1] - 1.0, center[1] + 1.0)
    ax.set_zlim(center[2] - 1.0, center[2] + 1.0)
    ax.axis('off')

def train(model, dataloader, optimizer, epoch):
    model.train()
    loss_all = AverageMeter()
    loss_vert = AverageMeter()
    loss_joint = AverageMeter()
    PRINT_FREQ = 50
    for i, sample in enumerate(dataloader):
        # pcd = sample['human_points']
        # import pdb; pdb.set_trace()
        pose_input = sample['smpl_joints']
        optimizer.zero_grad()
        ret_dict = model(pose_input)
        loss_dict = model.all_loss(ret_dict, sample)
        all_loss = loss_dict['loss']
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
    print('============Testing============')
    for sample in tqdm(dataloader):
        pose_input = sample['pose_pred_local'].squeeze(1)
        gt_pose = sample['smpl_joints_local']
        # import pdb; pdb.set_trace()
        pose_pelvis = pose_input[:,COLLECTION_15_NAME.index('pelvis')].cpu().numpy()[0]
        # print(pose_pelvis.shape)
        with torch.no_grad():
            ret_dict = model(pose_input)
        pred_vertices = ret_dict['mesh_out']
        gt_vertices = sample['smpl_verts_local'].to(pred_vertices.device)
        pose = ret_dict['pose']
        mpjpe += (pose - gt_pose.to(pose.device)).norm(dim = -1).mean() 

        # has_3d_joints = sample['has_3d_joints'].cuda(device)
        # import pdb; pdb.set_trace()
        error_vertices = mean_per_vertex_error(pred_vertices, gt_vertices).mean()
        error_edges, error_relative_edges = mean_per_edge_error(pred_vertices, gt_vertices, model.mesh_model.face)
        # error_joints = mean_per_joint_position_error(pred_3d_joints_from_smpl, gt_smpl_3d_joints,  has_3d_joints)
        # error_joints_pa = reconstruction_error(pred_3d_joints_from_smpl.cpu().numpy(), gt_smpl_3d_joints[:,:,:3].cpu().numpy(), reduction=None)

        # if error_vertices > 1:
        #     import pdb; pdb.set_trace()
        # print(error_vertices)
        mpve += error_vertices
        mpere += error_relative_edges
        number += 1
        if show:
            # if error_vertices <= 0.1:
            #     continue
            print(error_vertices)

            # ax = plt.figure().add_subplot(111, projection = '3d')
            # ax0 = plt.figure().add_subplot(111, projection = '3d')
            # ax1 = plt.figure().add_subplot(111, projection = '3d')
            # ax2 = plt.figure().add_subplot(111, projection = '3d')
            # pose = pose_input[0].cpu().numpy()
            # pcd = sample['human_points_local'][0].cpu().numpy()
            pred_show, gt_show = pred_vertices[0].cpu().numpy(), gt_vertices[0].cpu().numpy()
            # config_fig(ax, pose_pelvis)
            # config_fig(ax0, pose_pelvis)
            # config_fig(ax1, pose_pelvis)
            # config_fig(ax2, pose_pelvis)
            # ax.scatter(pcd[:,0], pcd[:,1], pcd[:,2], c = 'b', s = 2)
            # for bn in skeleton_15:
            #     ax0.plot(pose[bn,0], pose[bn,1], pose[bn,2], c = 'r')
            # ax1.plot_trisurf(pred_show[:, 0], pred_show[:,1], pred_show[:,2], triangles=model.mesh_model.face)
            # ax2.plot_trisurf(gt_show[:, 0], gt_show[:,1], gt_show[:,2], triangles=model.mesh_model.face)
            # plt.show()
            meshes = []
            meshes.append(get_mesh(pred_show, model.mesh_model.face))
            gt_show[:,0] += 1.0
            meshes.append(get_mesh(gt_show, model.mesh_model.face))
            o3d.visualization.draw_geometries(meshes)

    mpve = mpve.item() / number
    mpjpe = mpjpe.item() / number
    mpere = mpere.item() / number    
    return mpve, mpjpe, mpere

args = parse_args()
setup_seed(10)
dataset_task = args.dataset #'sloper4d', 'waymov2', 'collect'
model_type = 'p2m' #'lpformer', 'v2v'
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
                                fix_pts_num=True, return_smpl = True, augmentation = True, interval = 1, load_v2v=True)
    test_dataset = SLOPER4D_Dataset(dataset_root, scene_test, is_train = False, dataset_path = './save_data/sloper4d',
                                return_torch=False, device = 'cuda',
                                fix_pts_num=True, return_smpl = True, interval = 1, load_v2v=True)

elif dataset_task == 'waymov2':
    dataset_root = '/Extra/fanbohao/posedataset/PointC/Waymo/resave_files/'
    train_dataset = WAYMOV2_Dataset(dataset_root, is_train = True, dataset_path = './save_data/waymov2',
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, augmentation = True, load_v2v=True)
    test_dataset = WAYMOV2_Dataset(dataset_root, is_train = False, dataset_path = './save_data/waymov2',
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, load_v2v=True)

elif dataset_task == 'humanm3':
    train_dataset = HumanM3_Dataset(is_train = True,
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, augmentation = True, interval = 5, load_v2v = True)
    test_dataset = HumanM3_Dataset(is_train = False,
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, interval = 5, load_v2v = True)
    num_keypoints = 15
    # train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

model = pointcloud2mesh_net().cuda()
bs = 64
# train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

# optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr = 3e-3) #, betas = (0.85. 0.95)
# optimizer = optim.AdamW(model.parameters(), lr = 1e-4, betas = (0.85, 0.95)) #
# optimizer = optim.SGD(model.parameters(), lr = 1e-4)

optimizer = torch.optim.Adam(params=list(model.parameters()),
                                           lr=1e-4,
                                           betas=(0.9, 0.999),
                                           weight_decay=0)

state_dict = torch.load(args.state_dict)
model.load_state_dict(state_dict['net'])
# train(model, train_loader, optimizer, 0)
mpve, mpjpe, mpere = test(model, test_loader, True)
print(mpve, mpjpe, mpere)