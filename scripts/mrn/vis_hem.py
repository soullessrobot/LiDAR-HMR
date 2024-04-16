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
import copy
# import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--model', default='lpformer', required=False, type=str)
    parser.add_argument(
        '--dataset', default='sloper4d', required=False, type=str)
    parser.add_argument(
        '--state_dict', default='', required=False, type=str)
    parser.add_argument(
        '--cfg', default='configs/default.yaml', required=False, type=str)
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

def test(model, dataloader, show = False):
    model.eval()
    mpjpe = 0
    mpvpe = 0
    mpee = 0
    mpere = 0
    number = 0
    precision = 0
    print('============Testing============')
    kps15_bone = ((0, 7), (7, 8), (7, 9), (7, 10), (9, 11), (10, 12), (11, 13), (12, 14), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))
    mesh_face = model.mesh_faces[0]
    for inds, sample in enumerate(tqdm(dataloader)):
        if inds < 20:
            continue
        for key in sample:
            if type(sample[key]) is not dict and type(sample[key]) is not list:
                sample[key] = sample[key].cuda()
        pcd = sample['human_points_local']
        gt_pose = sample['smpl_joints_local']
        
        # pcd = sample['human_points_local']
        with torch.no_grad():
            ret_dict = model(pcd)
        pose = ret_dict['pose']
        # import pdb; pdb.set_trace()
        
        mpjpe += (pose - gt_pose.to(pose.device)).norm(dim = -1).mean() 
        seg = ret_dict['seg']
        gt_seg = sample['seg_label'].to(seg.device)
        precision += (seg.argmax(dim = 1) == gt_seg).sum() / gt_seg.numel() 
        pred_vertices = ret_dict['mesh_out']
        mid_meshes = ret_dict['pose_lister']
        mesh_faces = model.mesh_faces
        # pred_vertices_smpl = ret_dict['mesh_smpl']
        gt_vertices = sample['smpl_verts_local'].to(pred_vertices.device)

        # import pdb; pdb.set_trace()
        error_vertices = mean_per_vertex_error(pred_vertices, gt_vertices).mean()
        error_edges, error_relative_edges = mean_per_edge_error(pred_vertices, gt_vertices, mesh_face)#.mean()
        mpvpe += error_vertices
        mpee += error_edges
        mpere += error_relative_edges
        number += 1
        # import pdb; pdb.set_trace()
        if show:
            # if error_vertices <= 0.08:
            #     continue
            print(error_vertices)

            pcd_show = pcd[0].cpu().numpy()
            pred_show, gt_show = pred_vertices[0].cpu().numpy(), gt_vertices[0].cpu().numpy()#, pred_vertices_smpl[0].cpu().numpy()
            pose_show = pose[0].cpu().numpy()
            pose_pelvis = pose_show[0]
            # fig = plt.figure()
            # ax = fig.add_subplot(221, projection = '3d')
            # # ax.scatter(pose_show[:,0], pose_show[:,1], pose_show[:,2], s = 3)
            # ax.scatter(pcd_show[:,0], pcd_show[:,1], pcd_show[:,2], s = 1, c= 'gray')
            # # for bone in kps15_bone:
            # #     ax.plot(pose_show[bone,0], pose_show[bone,1], pose_show[bone,2], c= 'b')
            # ax.set_xlim(pose_pelvis[0] - 1.0, pose_pelvis[0] + 1.0)
            # ax.set_ylim(pose_pelvis[1] - 1.0, pose_pelvis[1] + 1.0)
            # ax.set_zlim(pose_pelvis[2] - 1.0, pose_pelvis[2] + 1.0)
            # ax.axis('off')
            
            # ax = fig.add_subplot(222, projection = '3d')
            # ax.scatter(pose_show[:,0], pose_show[:,1], pose_show[:,2], s = 3)
            # ax.scatter(pcd_show[:,0], pcd_show[:,1], pcd_show[:,2], s = 1, c= 'gray')
            # for bone in kps15_bone:
            #     ax.plot(pose_show[bone,0], pose_show[bone,1], pose_show[bone,2], c= 'b')
            # ax.set_xlim(pose_pelvis[0] - 1.0, pose_pelvis[0] + 1.0)
            # ax.set_ylim(pose_pelvis[1] - 1.0, pose_pelvis[1] + 1.0)
            # ax.set_zlim(pose_pelvis[2] - 1.0, pose_pelvis[2] + 1.0)
            # ax.axis('off')
            # plt.show()
            # # ax.set_xlim(pose_pelvis[0,0] - 1.0, pose_pelvis[0,0] + 1.0)
            # # ax.set_ylim(pose_pelvis[0,1] - 1.0, pose_pelvis[0,1] + 1.0)
            # # ax.set_zlim(pose_pelvis[0,2] - 1.0, pose_pelvis[0,2] + 1.0)
            # # print(pose_pelvis.shape)
            # ax1.set_xlim(pose_pelvis[0] - 1.0, pose_pelvis[0] + 1.0)
            # ax1.set_ylim(pose_pelvis[1] - 1.0, pose_pelvis[1] + 1.0)
            # ax1.set_zlim(pose_pelvis[2] - 1.0, pose_pelvis[2] + 1.0)
            # ax1.axis('off')

            # ax2.set_xlim(pose_pelvis[0] - 1.0, pose_pelvis[0] + 1.0)
            # ax2.set_ylim(pose_pelvis[1] - 1.0, pose_pelvis[1] + 1.0)
            # ax2.set_zlim(pose_pelvis[2] - 1.0, pose_pelvis[2] + 1.0)
            # ax2.axis('off')

            # ax0.set_xlim(pose_pelvis[0] - 1.0, pose_pelvis[0] + 1.0)
            # ax0.set_ylim(pose_pelvis[1] - 1.0, pose_pelvis[1] + 1.0)
            # ax0.set_zlim(pose_pelvis[2] - 1.0, pose_pelvis[2] + 1.0)
            # ax0.axis('off')

            # # ax.scatter(pcd_show[0,:,0], pcd_show[0,:,1], pcd_show[0,:,2])
            # # ax1.scatter(pred_show[0,:,0], pred_show[0,:,1], pred_show[0,:,2], c = 'b')
            # # ax2.scatter(gt_show[0,:,0], gt_show[0,:,1], gt_show[0,:,2], c = 'b')
            # ax1.plot_trisurf(pred_show[:, 0], pred_show[:,1], pred_show[:,2], triangles=model.smpl.faces)
            # # ax1.scatter(pcd_show[0,:,0], pcd_show[0,:,1], pcd_show[0,:,2], c = 'r', s = 1)
            # ax2.plot_trisurf(gt_show[:, 0], gt_show[:,1], gt_show[:,2], triangles=model.smpl.faces)
            # # ax2.scatter(pcd_show[0,:,0], pcd_show[0,:,1], pcd_show[0,:,2], c = 'r', s = 1)
            # # ax1.scatter(pcd_show[0,:,0], pcd_show[0,:,1], pcd_show[0,:,2], c = 'r', s = 1)
            # # ax2.scatter(pcd_show[0,:,0], pcd_show[0,:,1], pcd_show[0,:,2], c = 'r', s = 1)
            # ax0.scatter(pcd_show[0,:,0], pcd_show[0,:,1], pcd_show[0,:,2], c = 'r', s = 1)
            # ax0.scatter(pose_show[0,:,0], pose_show[0,:,1], pose_show[0,:,2], c = 'b', s = 2)

            # fig1 = plt.figure()
            # for ind, pose_e in enumerate(ret_dict['pose_lister']):
            #     if ind == 0:
            #         continue
            #     ax = fig1.add_subplot(3,4,ind, projection = '3d')
            #     pose_show = pose_e[0].cpu().numpy()
            #     ax.plot_trisurf(pose_show[:, 0], pose_show[:,1], pose_show[:,2], triangles=model.mesh_faces[-ind], edgecolor='Gray', color = 'Gray')
            # plt.show()
            

            meshes = []
            # meshes.append(get_mesh(pred_show, model.smpl.faces))

            # pcd_show = pcd[0].cpu().numpy()
            pcd_sk = copy.deepcopy(pcd_show)
            pcd_sk[:,0] -= 3.0
            pcd_ = o3d.open3d.geometry.PointCloud()
            pcd_.points= o3d.open3d.utility.Vector3dVector(pcd_sk)
            pcd_.paint_uniform_color([0.5,0.5,0.5])
            meshes.append(pcd_)
            
            lines = kps15_bone
            lines_pcd = o3d.geometry.LineSet()
            lines_pcd.lines = o3d.utility.Vector2iVector(lines)
            lines_pcd.points = o3d.utility.Vector3dVector(pose_show)
            lines_pcd.paint_uniform_color([0.5,0.5,0.5])
            # meshes.append(lines_pcd)
            pose_ = o3d.open3d.geometry.PointCloud()
            pose_.points= o3d.open3d.utility.Vector3dVector(pose_show)
            pose_.paint_uniform_color([0.5,0.5,0.5])
            # meshes.append(pose_)
            
            pcd_ = o3d.open3d.geometry.PointCloud()
            pcd_.points= o3d.open3d.utility.Vector3dVector(pcd_show)
            pcd_.paint_uniform_color([0,0,1])
            # meshes.append(pcd_)

            pcd_gt = pcd[0].cpu().numpy(); pcd_gt[:,0] -= 1.0
            pcd_ = o3d.open3d.geometry.PointCloud()
            pcd_.points= o3d.open3d.utility.Vector3dVector(pcd_gt)
            pcd_.paint_uniform_color([0,0,1])
            meshes.append(pcd_)

            gt_show[:,0] -= 1.0
            meshes.append(get_mesh(gt_show, model.smpl.faces))
            for ind, p_l in enumerate(mid_meshes):
                
                if ind == 0:
                    continue
                if ind != 6 and ind != 4: #4, 6
                    continue
                print(ind, p_l.shape, mesh_faces[-ind].shape)
                # print(ind, p_l.shape, mesh_faces[-ind].shape)
                # if p_l.shape[1] == 1946:
                # import pdb; pdb.set_trace()
                mid_show = p_l[0].cpu().numpy()
                mid_show[:,1] += len(mid_meshes) - 1 - ind
                # meshes.append(get_mesh(mid_show, mesh_faces[-ind]))
            
            # gt_meshes = []
            # gt_meshes.append(torch.tensor(gt_show))
            # for ind, indx in enumerate(model.multi_index):
            #     sub_mesh_this = (gt_meshes[-1][indx[:,0]] + gt_meshes[-1][indx[:,1]] ) / 2
            #     gt_meshes.append(sub_mesh_this)
            #     mid_show = sub_mesh_this.clone().cpu().numpy()
            #     mid_show[:,1] += ind + 1
            #     meshes.append(get_mesh(mid_show, mesh_faces[ind + 1]))
                
            o3d.visualization.draw_geometries(meshes)

    mpjpe = mpjpe.item() / number    
    mpee = mpee.item() / number
    mpere = mpere.item() / number
    precision = precision.item() / number
    mpvpe = mpvpe.item() / number
    # print('MPJPE: '+str(mpjpe))
    return mpjpe, precision, mpvpe, mpee, mpere
    # print('loss:{:4f}')

def show_hem(model, dataloader):
    model.eval()
    mpjpe = 0
    mpvpe = 0
    mpee = 0
    mpere = 0
    number = 0
    precision = 0
    kps15_bone = ((0, 7), (7, 8), (7, 9), (7, 10), (9, 11), (10, 12), (11, 13), (12, 14), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))
    mesh_face = model.mesh_faces[0]
    for inds, sample in enumerate(tqdm(dataloader)):
        if inds < 45:
            continue
        pcd = sample['human_points_local']
        gt_pose = sample['smpl_joints_local']
        
        mesh_faces = model.mesh_faces
        gt_vertices = sample['smpl_verts_local']
        
        pcd_show = pcd[0].cpu().numpy()
        gt_show = gt_vertices[0].cpu().numpy()#, pred_vertices_smpl[0].cpu().numpy()
        pose_pelvis = gt_show[0] #[1,3]
        
        meshes = []
        pcd_sk = copy.deepcopy(pcd_show)
        pcd_sk[:,0] -= 3.0
        pcd_ = o3d.open3d.geometry.PointCloud()
        pcd_.points= o3d.open3d.utility.Vector3dVector(pcd_sk)
        pcd_.paint_uniform_color([0.5,0.5,0.5])
        meshes.append(pcd_)
       
        gt_meshes = []
        gt_meshes.append(torch.tensor(gt_show))
        fig1 = plt.figure()
        ax = fig1.add_subplot(111, projection = '3d')
        ax.scatter(pcd_show[:,0], pcd_show[:,1], pcd_show[:,2], c = 'gray', s = 2)
        ax.set_xlim(pose_pelvis[0] - 1.0, pose_pelvis[0] + 1.0)
        ax.set_ylim(pose_pelvis[1] - 1.0, pose_pelvis[1] + 1.0)
        ax.set_zlim(pose_pelvis[2] - 1.0, pose_pelvis[2] + 1.0)
        ax.axis('off')
        
        # for ind, indx in enumerate(model.multi_index):
        #     sub_mesh_this = (gt_meshes[-1][indx[:,0]] + gt_meshes[-1][indx[:,1]] ) / 2
        #     gt_meshes.append(sub_mesh_this)
        #     mid_show = sub_mesh_this.clone().cpu().numpy()
        #     mid_show[:,1] += ind + 1
        #     meshes.append(get_mesh(mid_show, mesh_faces[ind + 1]))
        #     print(ind, mid_show.shape, mesh_faces[ind+1].shape)
        #     mid_line_show = sub_mesh_this.clone()#.cpu().numpy()
        #     # mid_line_show = (mid_line_show - pose_pelvis) / (mid_line_show - pose_pelvis).norm(dim = -1, keepdim = True) * 0.001 + mid_line_show
        #     # mid_line_show[:,1] += ind + 1
        #     mid_line_show = mid_line_show.cpu().numpy()
        #     line_this = []
        #     for face_ in mesh_faces[ind + 1]:
        #         line_this.append([face_[0], face_[1]])
        #         line_this.append([face_[0], face_[2]])
        #         line_this.append([face_[1], face_[2]])
        #     fig1 = plt.figure()
        #     # ax = fig1.add_subplot(3,4,ind + 1, projection = '3d')
        #     ax = fig1.add_subplot(111, projection = '3d')
        #     c = ((0.6, 0.823, 0.941),(0.0, 0.0, 0.0))
        #     ax.plot_trisurf(sub_mesh_this[:, 0], sub_mesh_this[:,1], sub_mesh_this[:,2], triangles=mesh_faces[ind + 1], edgecolors=c[1], color = c[0], linewidth = 0.15)
        #     ax.set_xlim(pose_pelvis[0] - 1.0, pose_pelvis[0] + 1.0)
        #     ax.set_ylim(pose_pelvis[1] - 1.0, pose_pelvis[1] + 1.0)
        #     ax.set_zlim(pose_pelvis[2] - 1.0, pose_pelvis[2] + 1.0)
        #     ax.axis('off')
        #     # lines_pcd = o3d.geometry.LineSet()
        #     # lines_pcd.lines = o3d.utility.Vector2iVector(line_this)
        #     # lines_pcd.points = o3d.utility.Vector3dVector(mid_line_show)
        #     # lines_pcd.paint_uniform_color([0,0,0])
        #     # meshes.append(lines_pcd)
        #     # ax.show(elevation=-13/180*3.14, azimuth=161/180*3.14, interactive=False, zoom=1.5)
        #     # break
        plt.show()
        # o3d.visualization.draw_geometries(meshes)

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

# import pdb; pdb.set_trace()
# model = pose_meshformer().cuda()
update_config(args.cfg)
model = pose_meshgraphormer(pmg_cfg = config).cuda()

# total = sum([param.nelement() for param in model.parameters()])
# print(parameter_count_table(model))
# import pdb; pdb.set_trace()
# bs = 8
show = True
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

show_hem(model, test_loader)
# print('MPJPE: '+str(mpjpe) + '; MPEE:' + str(mpee) + '; MPVPE:'+str(mpvpe) + '; MPERE:'+str(mpere))
