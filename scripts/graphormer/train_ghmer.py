import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from datasets.sloper4d import SLOPER4D_Dataset
from datasets.waymo_v2 import WAYMOV2_Dataset
from datasets.humanm3 import HumanM3_Dataset
from models.graphormer.graphormer_model import graphormer_model#, graphormer_model_smpl
# from models.v2v_posenet import V2VPoseNet
from tqdm import tqdm
import torch.optim as optim
import logging
# import torch.nn.functional as F
# import os
import argparse
# import torch.distributed as dist
from scripts.eval_utils import mean_per_vertex_error, mean_per_joint_position_error, reconstruction_error, all_gather, setup_seed
from models.graphormer.data.config import H36M_J17_NAME, H36M_J17_TO_J14, J24_NAME, J24_TO_J14
from models.pose2mesh.keypoints_config import JOINT_REGRESSOR_H36M_correct, H36M_J17_TO_J15
import matplotlib.pyplot as plt
from thop import profile, clever_format
from fvcore.nn import FlopCountAnalysis, parameter_count_table

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--model', default='graphormer', required=False, type=str)
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

def train(model, dataloader, optimizer, epoch, PRINT_FREQ = 100):
    model.train()
    loss_all = AverageMeter()
    loss_vert = AverageMeter()
    loss_joint = AverageMeter()
    loss_seg = AverageMeter()
    PRINT_FREQ = PRINT_FREQ
    for i, sample in enumerate(dataloader):
        for key in sample:
            if type(sample[key]) is not dict and type(sample[key]) is not list:
                sample[key] = sample[key].cuda()
        pcd = sample['human_points_local']
        optimizer.zero_grad()
        
        # flops = FlopCountAnalysis(model, pcd)
        # print("FLOPs: ", flops.total())
        # import pdb; pdb.set_trace()
        
        # flops, params = profile(model, inputs = (pcd,))
        # flops, params = clever_format([flops, params], "%.3f")
        # print(flops, params)
        # import pdb; pdb.set_trace()

        ret_dict = model(pcd)
        loss_dict = model.all_loss(ret_dict, sample)
        if epoch >=7 :
            loss_dict['loss'] += loss_dict['edge_loss']
        all_loss = loss_dict['loss']#loss_dict['loss']
        
        # import pdb; pdb.set_trace()
        all_loss.backward()
        optimizer.step()
        loss_all.update(loss_dict['loss'].item())
        loss_vert.update(loss_dict['vertices_loss'].item())
        loss_joint.update(loss_dict['joint_loss'].item())
        # loss_xyz.update(loss_dict['xyz_loss'].item())
        # loss_vis.update(loss_dict['vis_loss'].item())
        if 'seg_loss' in loss_dict:
            loss_seg.update(loss_dict['seg_loss'].item())
        if i % PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Loss_all: {loss_all.val:.7f} ({loss_all.avg:.7f})\t' \
                  'loss_vert: {loss_vert.val:.7f} ({loss_vert.avg:.7f})\t' \
                    'loss_joint: {loss_joint.val:.7f} ({loss_joint.avg:.7f})\t' \
                    'loss_seg: {loss_seg.val:.7f} ({loss_seg.avg:.7f})\t'.format(
                    epoch, i, len(dataloader), loss_all=loss_all, loss_vert=loss_vert, loss_joint = loss_joint, loss_seg = loss_seg)
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
        pcd = sample['human_points']
        gt_xyz = sample['smpl_joints_local']
        center = sample['global_trans']
        gt_vis = sample['vis_label']
        flag = sample['valid_flag'] if 'valid_flag' in sample else None
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            ret_dict = model(pcd, center)
        xyz, vis = ret_dict['xyz'], ret_dict['vis']
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

def run_validate(val_loader, Graphormer_model, smpl, mesh_sampler, device = 0, show = False):
    batch_time = AverageMeter()
    mPVE = AverageMeter()
    mPJPE = AverageMeter()
    PAmPJPE = AverageMeter()
    Precision = AverageMeter()
    # switch to evaluate mode
    Graphormer_model.eval()
    smpl.eval()
    PRINT_FREQ = 100
    print('Testing:')
    color = np.random.rand(100,3)
    with torch.no_grad():
        for i, sample in tqdm(enumerate(val_loader)):
            # gt_3d_joints = sample['smpl_joints'].cuda(device)
            # gt_3d_pelvis = gt_3d_joints[:,J24_NAME.index('Pelvis'),:3]
            # gt_3d_joints = gt_3d_joints[:,J24_TO_J14,:] 
            # gt_3d_joints[:,:,:3] = gt_3d_joints[:,:,:3] - gt_3d_pelvis[:, None, :]
            
            has_3d_joints = sample['has_3d_joints'].cuda(device)
            pcd = sample['human_points_local']
            # gt_pose = sample['smpl_pose'].cuda(device)
            # gt_betas = sample['betas'].cuda(device)
            has_smpl = sample['has_smpl'].cuda(device)

            # generate simplified mesh
            # gt_vertices = smpl(gt_pose, gt_betas)
            gt_vertices = sample['smpl_verts_local']
            # gt_vertices_sub = mesh_sampler.downsample(gt_vertices)
            # gt_vertices_sub2 = mesh_sampler.downsample(gt_vertices_sub, n1=1, n2=2)
            # import pdb; pdb.set_trace()
            # normalize gt based on smpl pelvis 
            gt_smpl_3d_joints = smpl.get_h36m_joints(gt_vertices)
            # gt_smpl_3d_pelvis = gt_smpl_3d_joints[:,H36M_J17_NAME.index('Pelvis'),:]
            gt_smpl_3d_joints = gt_smpl_3d_joints[:,H36M_J17_TO_J15,:] 
            # gt_smpl_3d_joints[:,:,:3] = gt_smpl_3d_joints[:,:,:3] - gt_smpl_3d_pelvis[:, None, :]

            # gt_vertices_sub2 = gt_vertices_sub2 - gt_smpl_3d_pelvis[:, None, :] 
            # gt_vertices = gt_vertices - gt_smpl_3d_pelvis[:, None, :] 

            # forward-pass
            ret_dict = Graphormer_model(pcd)
            pred_vertices = ret_dict['pred_verts']
            # pred_betas = ret_dict['pred_betas'] if 'pred_betas' in ret_dict else None
            # obtain 3d joints from full mesh
            pred_3d_joints_from_smpl = smpl.get_h36m_joints(pred_vertices)

            # pred_3d_pelvis = pred_3d_joints_from_smpl[:,H36M_J17_NAME.index('Pelvis'),:]
            pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:,H36M_J17_TO_J15,:]
            # pred_3d_joints_from_smpl = pred_3d_joints_from_smpl - pred_3d_pelvis[:, None, :]
            # pred_vertices = pred_vertices - pred_3d_pelvis[:, None, :]

            # measure errors
            error_vertices = mean_per_vertex_error(pred_vertices, gt_vertices, has_smpl)
            error_joints = mean_per_joint_position_error(pred_3d_joints_from_smpl, gt_smpl_3d_joints,  has_3d_joints)
            error_joints_pa = reconstruction_error(pred_3d_joints_from_smpl.cpu().numpy(), gt_smpl_3d_joints[:,:,:3].cpu().numpy(), reduction=None)
            
            if len(error_vertices)>0:
                mPVE.update(np.mean(error_vertices), int(torch.sum(has_smpl)) )
            if len(error_joints)>0:
                mPJPE.update(np.mean(error_joints), int(torch.sum(has_3d_joints)) )
            if len(error_joints_pa)>0:
                PAmPJPE.update(np.mean(error_joints_pa), int(torch.sum(has_3d_joints)) )
            
            seg_ = ret_dict['seg']
            seg_index = torch.argmax(seg_, axis = 1)
            seg_label = sample['seg_label'].to(seg_index.device)
            tp = seg_index == seg_label
            precision = tp.sum()/tp.numel()
            Precision.update(precision.item())
            # import pdb; pdb.set_trace()
            # if i % PRINT_FREQ == 0:
            #     val_mPVE = all_gather(float(mPVE.avg))
            #     val_mPVE = sum(val_mPVE)/len(val_mPVE)
            #     val_mPJPE = all_gather(float(mPJPE.avg))
            #     val_mPJPE = sum(val_mPJPE)/len(val_mPJPE)
            #     val_PAmPJPE = all_gather(float(PAmPJPE.avg))
            #     val_PAmPJPE = sum(val_PAmPJPE)/len(val_PAmPJPE)
            #     print(val_mPVE, val_mPJPE, val_PAmPJPE)
            if show:
                # if pred_betas is not None:
                #     print(pred_betas[0], sample['betas'][0])
                fig = plt.figure()
                ax = fig.add_subplot(231, projection = '3d')
                ax1 = fig.add_subplot(232, projection = '3d')
                ax2 = fig.add_subplot(233, projection = '3d')
                pcd_show = pcd.cpu().numpy()
                pred_show, gt_show = pred_vertices.cpu().numpy(), gt_vertices.cpu().numpy()
                ax.scatter(pcd_show[0,:,0], pcd_show[0,:,1], pcd_show[0,:,2])
                ax1.scatter(pred_show[0,:,0], pred_show[0,:,1], pred_show[0,:,2], c = 'b')
                ax2.scatter(gt_show[0,:,0], gt_show[0,:,1], gt_show[0,:,2], c = 'b')
                pj_show, gj_show = pred_3d_joints_from_smpl.cpu().numpy(), gt_smpl_3d_joints.cpu().numpy()
                ax3 = fig.add_subplot(234, projection = '3d')
                ax4 = fig.add_subplot(235, projection = '3d')
                ax3.scatter(pj_show[0,:,0], pj_show[0,:,1], pj_show[0,:,2], c = 'r')
                ax4.scatter(gj_show[0,:,0], gj_show[0,:,1], gj_show[0,:,2], c = 'r')

                fig1 = plt.figure()
                ax = fig1.add_subplot(121, projection = '3d')
                ax1 = fig1.add_subplot(122, projection = '3d')
                seg_ = ret_dict['seg'].cpu().numpy()
                # import pdb; pdb.set_trace()
                seg_index = np.argmax(seg_, axis = 1)
                for i in range(26):
                    seg_this = seg_index[0] == i
                    if pcd_show[0,seg_this].shape[0] > 0:
                        ax.scatter(pcd_show[0,seg_this,0], pcd_show[0,seg_this,1], pcd_show[0,seg_this,2], c = color[i])
                    seg_label_this = sample['seg_label'][0] == i
                    if pcd_show[0,seg_label_this].shape[0] > 0:
                        ax1.scatter(pcd_show[0,seg_label_this,0], pcd_show[0,seg_label_this,1], pcd_show[0,seg_label_this,2], c = color[i])
                # ax1.scatter(pred_show[0,:,0], pred_show[0,:,1], pred_show[0,:,2])
                # plt.show()
                plt.show()
    val_mPVE = all_gather(float(mPVE.avg))
    val_mPVE = sum(val_mPVE)/len(val_mPVE)
    val_mPJPE = all_gather(float(mPJPE.avg))
    val_mPJPE = sum(val_mPJPE)/len(val_mPJPE)
    val_PAmPJPE = all_gather(float(PAmPJPE.avg))
    val_PAmPJPE = sum(val_PAmPJPE)/len(val_PAmPJPE)
    return val_mPVE, val_mPJPE, val_PAmPJPE

args = parse_args()
setup_seed(10)
dataset_task = args.dataset #'sloper4d', 'waymov2', 'collect'
model_type = args.model #'graphormer' #'graphormer_smpl' #'lpformer', 'v2v'
if model_type == 'graphormer':
    model = graphormer_model().cuda()
    bs = 2
    interval = 4
    PRINT_FREQ = 500

elif model_type == 'graphormer_smpl':
    model = graphormer_model_smpl().cuda()
    bs = 64
    interval = 1
    PRINT_FREQ = 100

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
    num_keypoints = 14

elif dataset_task == 'humanm3':
    train_dataset = HumanM3_Dataset(is_train = True,
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, augmentation = True, interval = 5)
    test_dataset = HumanM3_Dataset(is_train = False,
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, interval = 5)
    num_keypoints = 15

train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

# optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr = 3e-3) #, betas = (0.85. 0.95)
# optimizer = optim.AdamW(model.parameters(), lr = 1e-4, betas = (0.85, 0.95)) #
# optimizer = optim.SGD(model.parameters(), lr = 1e-4)
optimizer = torch.optim.Adam(params=list(model.parameters()),
                                           lr=1e-4,
                                           betas=(0.9, 0.999),
                                           weight_decay=0)
import datetime
now = datetime.datetime.now()
time_str = now.strftime("%Y-%m-%d %H:%M:%S")
epoch_now = 0
if args.state_dict != '':
    state_dict = torch.load(args.state_dict)
    # import pdb; pdb.set_trace()
    model.load_state_dict(state_dict['net'])
    epoch_now = int(args.state_dict.split('/')[-1].split('_')[0].replace('epoch', ''))

for epoch in range(50):
    if epoch <= epoch_now:
        continue
    train(model, train_loader, optimizer, epoch, PRINT_FREQ)
    state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
    save_dir = os.path.join('save_state', dataset_task, 'mesh', model_type, time_str)
    os.makedirs(save_dir, exist_ok=True)
    val_mPVE, val_mPJPE, val_PAmPJPE = run_validate(test_loader, model, model.smpl, model.mesh)
    msg = 'epoch{0}_{mpjpe:.5f}_{mpvpe:.5f}.pth'.format(epoch, mpjpe = val_mPJPE, mpvpe = val_mPVE)
    torch.save(state, os.path.join(save_dir, msg))
    print('mPVE: '+str(val_mPVE) + '; MPJPE:' + str(val_mPJPE) + '; PA-MPJPE:' + str(val_PAmPJPE))
