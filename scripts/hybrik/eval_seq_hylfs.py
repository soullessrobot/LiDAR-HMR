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
from scripts.eval_utils import mean_per_vertex_error, setup_seed, mean_per_edge_error
from models.graphormer.data.config import H36M_J17_NAME, H36M_J17_TO_J14, J24_NAME, J24_TO_J14
from fvcore.nn import FlopCountAnalysis, parameter_count_table
# from thop import profile, clever_format
from models.pct_config import pct_config, update_pct_config
import smplx
import h5py
import pickle

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

def test(model, data_file):
    model.eval()
    data = h5py.File(data_file, 'r')
    human_points = torch.tensor(data['point_clouds']).float()
    num_seq, seq_len = human_points.shape[:2]
    pred_pose = torch.zeros(num_seq, seq_len, 72)
    pred_trans = torch.zeros(num_seq, seq_len, 3)
    pred_betas = torch.zeros(num_seq, seq_len, 10)
    for i in tqdm(range(num_seq)):
        pcd_this = torch.tensor(human_points[i]).cuda() #[32,N,3]
        max_, min_ = pcd_this.max(dim = 1)[0], pcd_this.min(dim = 1)[0]
        global_trans = (max_ + min_) / 2
        human_points_local = pcd_this - global_trans.unsqueeze(1)
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            ret_dict = model(human_points_local)
        beta_, theta_, trans_ = ret_dict['pose_beta'], ret_dict['pose_theta'], ret_dict['trans']
        trans_ = trans_ + global_trans
        pred_pose[i] = theta_
        pred_trans[i] = trans_
        pred_betas[i] = beta_
    return pred_pose, pred_trans, pred_betas
    
args = parse_args()
setup_seed(10)
dataset_task = args.dataset #'sloper4d', 'waymov2', 'collect'
model_type = 'meshik' #'lpformer', 'v2v'

# model = pose_meshformer().cuda()
config_name = args.cfg.split('/')[-1].split('.')[0]
update_config(args.cfg)
model = LiDAR_HMR(pmg_cfg = config, train_pmg = True).cuda()

state_dict = torch.load(args.state_dict)
model.load_state_dict(state_dict['net'])
data_file = os.path.join('/mnt/data1/fbh/DnD_template/LiDARCapSource/LiDARCap/dataset_files/', 'sloper4d_test32.hdf5')

pred_pose, pred_trans, pred_betas = test(model, data_file)
save_dict = {'pose':pred_pose.cpu().numpy(), 'trans':pred_trans.cpu().numpy(), 'betas':pred_betas.cpu().numpy()}
data_save_file = os.path.join('/mnt/data1/fbh/Pcd_Sequence/data_files/', 'sloper4d_test32.pkl')
with open(data_save_file, 'wb') as file:
    pickle.dump(save_dict, file)