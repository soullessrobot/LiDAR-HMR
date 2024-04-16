import faulthandler
faulthandler.enable()
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
from models.pose_mesh_net import LiDAR_HMR
# from models.v2v_posenet import V2VPoseNet
from tqdm import tqdm
import torch.optim as optim
import logging
import torch.nn.functional as F
import argparse
from models.pmg_config import config, update_config
from scripts.eval_utils import mean_per_vertex_error, setup_seed, mean_per_edge_error, smpl_model, compute_error_accel
from models.graphormer.data.config import H36M_J17_NAME, H36M_J17_TO_J14, J24_NAME, J24_TO_J14
from fvcore.nn import FlopCountAnalysis, parameter_count_table
# from thop import profile, clever_format
from models.pct_config import pct_config, update_pct_config
import smplx

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--model', default='lpformer', required=False, type=str)
    parser.add_argument(
        '--dataset', default='sloper4d', required=False, type=str)
    parser.add_argument(
        '--state_dict', default='', required=False, type=str)
    parser.add_argument(
        '--resume_state_dict', default='', required=False, type=str)
    parser.add_argument(
        '--prn_state_dict', default='', required=False, type=str)
    parser.add_argument(
        '--cfg', default='configs/mesh/adj.yaml', required=False, type=str)
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
    loss_shift = AverageMeter()
    loss_joint = AverageMeter()
    loss_vert = AverageMeter()
    PRINT_FREQ = 500
    for i, sample in enumerate(dataloader):
        for key in sample:
            if type(sample[key]) is not dict and type(sample[key]) is not list:
                sample[key] = sample[key].cuda()
        pcd = sample['human_points_local']
        torch.autograd.set_detect_anomaly(True)
        optimizer.zero_grad()

        # flops = FlopCountAnalysis(model, pcd)
        # print("FLOPs: ", flops.total() / 1e9)

        # flops, params = profile(model, inputs = (pcd,))
        # flops, params = clever_format([flops, params], "%.3f")
        # print(flops, params)
        # import pdb; pdb.set_trace()
        
        ret_dict = model(pcd)
        loss_dict = model.all_loss(ret_dict, sample)
        all_loss = loss_dict['loss']
        if epoch >= 7:
            all_loss += loss_dict['edge_loss']
            loss_dict['loss_vert'] += loss_dict['edge_loss']
            # all_loss += loss_dict['pen_loss']
            # loss_dict['loss_vert'] += loss_dict['pen_loss']
            
        all_loss.backward()
        optimizer.step()
        loss_all.update(loss_dict['loss'].item())
        loss_joint.update(loss_dict['loss_joint'].item())
        loss_shift.update(loss_dict['loss_shift'].item())
        # if 'loss_vert' in loss_dict.keys():
        # import pdb; pdb.set_trace()
        loss_vert.update(loss_dict['loss_vert'].item())
        # print(loss_dict['loss_vert'])
        if i % PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Loss_all: {loss_all.val:.3f} ({loss_all.avg:.3f})\t' \
                  'loss_shift: {loss_shift.val:.3f} ({loss_shift.avg:.3f})\t' \
                    'loss_joint: {loss_joint.val:.3f} ({loss_joint.avg:.3f})\t' \
                    'loss_vert: {loss_vert.val:.3f} ({loss_vert.avg:.3f})'.format(
                    epoch, i, len(dataloader), loss_all=loss_all, loss_shift=loss_shift, \
                        loss_joint = loss_joint, loss_vert = loss_vert)
            print(msg)
            # logger.info(msg)

def test(model):
    model.eval()
    mpjpe = 0
    mpvpe = 0
    ae = 0
    J_r = smpl_model.J_regressor.cuda()
    file_name_list = ['lidarcap_test32']
    #['sloper4d_test32', 'sloper4d_train32', 'lidarcap_test32', 'lidarcap_train32', 'cimi4d_train32', 'cimi4d_test32']
    for file_name in file_name_list:
        all_data = np.load('/Extra/fanbohao/fbh_code/mesh_sequence/LiDARCap/lidarcap_pred/'+file_name+'.npy', allow_pickle=True).tolist()
        point_cloud = torch.tensor(all_data['pointcloud']).cuda().float()
        gt_pose = torch.tensor(all_data['gt_pose']).cuda().float()
        gt_trans = torch.tensor(all_data['gt_trans']).cuda().float()
        gt_shape = torch.tensor(all_data['gt_shape']).cuda().float()
        
        pred_trans_ = []
        pred_pose_ = []
        pred_shape_ = []
        
        n_seq, seq_len = point_cloud.shape[:2]
        number = 0
        gt_valid_index = np.ones([gt_pose.shape[0]]) > 0
        for i in tqdm(range(n_seq)):
            human_points = point_cloud[i] #[32,N,3]
            # for j in range(seq_len):
            max_, min_ = human_points.max(dim = 1)[0], human_points.min(dim = 1)[0]
            if (max_ == min_).sum() > 0:
                gt_valid_index[i] = False
                continue
            root = (max_ + min_) / 2 #[32,3]
            human_points_local = human_points - root[:,None,:]
            with torch.no_grad():
                ret_dict = model(human_points_local)
            pred_pose = ret_dict['pose_theta']
            pred_betas = ret_dict['pose_beta']
            pred_trans = ret_dict['trans']
            
            trans_all = root + pred_trans
            pred_trans_.append(trans_all.cpu().numpy())
            pred_pose_.append(pred_pose.cpu().numpy())
            pred_shape_.append(pred_betas.cpu().numpy())
            # import pdb; pdb.set_trace()
            
            pred_smpl_output = smpl_model(transl = trans_all, betas = pred_betas, body_pose = pred_pose[...,3:], global_orient = pred_pose[...,:3])
            # import pdb; pdb.set_trace()
            gt_smpl_output = smpl_model(transl = gt_trans[i], betas = gt_shape[i], body_pose = gt_pose[i,:,3:], global_orient = gt_pose[i,:,:3])
            mpvpe_ = mean_per_vertex_error(pred_smpl_output.vertices, gt_smpl_output.vertices).mean()
            mpjpe_ = (pred_smpl_output.joints - gt_smpl_output.joints).norm(dim = -1).mean()
            acc_error = compute_error_accel(pred_smpl_output.joints, gt_smpl_output.joints).mean()
            # if (pred_smpl_output.joints - gt_smpl_output.joints).norm(dim = -1).mean() > 1:
            #     import pdb; pdb.set_trace()
            # print((pred_smpl_output.joints - gt_smpl_output.joints).norm(dim = -1).mean(), mean_per_vertex_error(pred_smpl_output.vertices, gt_smpl_output.vertices).mean())
            mpjpe += mpjpe_
            mpvpe += mpvpe_
            ae += acc_error
            number += 1
        mpjpe = mpjpe.item() / number    
        mpvpe = mpvpe.item() / number
        ae = ae.item() / number
        print(mpjpe,mpvpe,ae)
        
        pred_trans_ = np.stack(pred_trans_, axis = 0)
        pred_pose_ = np.stack(pred_pose_, axis = 0)
        pred_shape_ = np.stack(pred_shape_, axis = 0)
        print(gt_trans.shape[0], pred_trans_.shape[0])
        all_data.update({'pred_pose': pred_pose_, 'pred_betas': pred_shape_, 'pred_trans': pred_trans_}) 
        all_data['gt_pose'] = all_data['gt_pose'][gt_valid_index]
        all_data['gt_trans'] = all_data['gt_trans'][gt_valid_index]
        all_data['gt_shape'] = all_data['gt_shape'][gt_valid_index]
        all_data['pointcloud'] = all_data['pointcloud'][gt_valid_index]
        np.save('pred_seq/'+file_name+'.npy', np.array(all_data))
    return mpjpe, mpvpe, ae

args = parse_args()
setup_seed(10)

config_name = args.cfg.split('/')[-1].split('.')[0]
update_config(args.cfg)
model = LiDAR_HMR(pmg_cfg = config, train_pmg = True).cuda()

if args.state_dict != '':
    state_dict = torch.load(args.state_dict)
    model.load_state_dict(state_dict['net'])
    
mpjpe, mpvpe, ae = test(model)
print('MPJPE: '+str(mpjpe) + '; MPVPE:' + str(mpvpe) + '; AE:'+str(ae) + '; ')
