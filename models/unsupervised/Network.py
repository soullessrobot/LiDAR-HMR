import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
# used for 6D vector
from pytorch3d.transforms import (matrix_to_rotation_6d,
                        				  rotation_6d_to_matrix,
                                  matrix_to_quaternion,
                                  quaternion_to_axis_angle)

from .utils import (PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetEncoder, 
                    feature_transform_reguliarzer, PointNetFeaturePropagation,
                    PointConvDensitySetAbstraction, 
                    knn, get_graph_feature, DGCNN_feature,
                    SA_Layer, Local_op, sample_and_group)
import smplx

model_params = dict(model_path='/disk1/fanbohao/fbh_code/SMPL/smplx/models/',
                    model_type = 'smpl')
smpl_model = smplx.create(gender='neutral', **model_params).cuda()

class point_net_ssg(nn.Module):
    def __init__(self,
                 shape_classes=10,
                 pose_classes=72-3,
                 trans_classes=3,
                 gRT_classes=6,
                 normal_channel=False,
                 smpl_mean_file="/disk1/fanbohao/fbh_code/mesh/FaceFormer/models/unsupervised/smpl_models/neutral_smpl_mean_params.h5",
                 device="cuda"):
                 
        super(point_net_ssg, self).__init__()
        self.smpl_mean_file = smpl_mean_file
        self.device = device

        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.1, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False) # change 0.1
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.2, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False) #change 0.2
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128+16+6+additional_channel, mlp=[128, 128, 128])

        self.fc1 = nn.Linear(1024 + 72 + 10 + 3 -3 , 1024)
        self.fc1_p = nn.Linear(1024, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.drop2 = nn.Dropout(0.5)

        self.decpose = nn.Linear(1024, pose_classes)
        self.decshape = nn.Linear(1024, shape_classes)
        self.dectrans = nn.Linear(1024, trans_classes)
        self.decR = nn.Linear(1024, gRT_classes)

        file = h5py.File(self.smpl_mean_file, 'r')
        init_pose = torch.from_numpy(file['pose'][:]).unsqueeze(0).float()
        init_shape = torch.from_numpy(file['shape'][:]).unsqueeze(0).float()
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.JOINTS_IDX = [8, 5, 29, 30, 4, 7, 21, 19, 17, 16, 18, 20, 31, 32, 33, 34, 35, 36, 37, 24, 26, 25, 28, 27]
        #  = smplx.create(model_path = )
        self.criterion_parameters = torch.nn.L1Loss()


    def forward(self,
                xyz,
                init_pose=None,
                init_shape=None,
                init_trans=None,
                n_iter=3):

        batch_size, _, _ = xyz.shape

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)[:, 3:]
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_trans is None:
            init_trans = torch.mean(xyz, dim = 1)
            # init_trans = torch.mean([batch_size, 3]).to(self.device)

        #xf, trans, trans_feat = self.feat(xyz)
        norm = None
        # import pdb; pdb.set_trace()
        xyz = xyz.permute(0,2,1)
        # import pdb; pdb.set_trace()
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        xf = l3_points.view(batch_size, 1024)
        
        
        # final Linear part
        xr = self.drop1(F.relu(self.bn1(self.fc1_p(xf))))
        xr = self.drop2(F.relu(self.bn2(self.fc2(xr))))
        pred_R6D = self.decR(xr)
        
        pred_pose = init_pose
        pred_shape = init_shape
        pred_trans = init_trans

        for i in range(n_iter):
            # import pdb; pdb.set_trace()
            xc = torch.cat([xf, pred_pose, pred_shape, pred_trans], 1)

            # add relu and BN  here
            xc = self.drop1(F.relu(self.bn1(self.fc1(xc))))
            xc = self.drop2(F.relu(self.bn2(self.fc2(xc))))

            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_trans = self.dectrans(xc) + pred_trans
        pred_R6D_3D = quaternion_to_axis_angle(matrix_to_quaternion((rotation_6d_to_matrix(pred_R6D))))
        ret_dict = {'pred_shape':pred_shape, 'pred_pose':pred_pose, 'pred_trans':pred_trans, 'pred_R6D_3D':pred_R6D_3D}
        return ret_dict

    def all_loss(self, ret_dict, sample):
        pred_shape, pred_pose, pred_trans, pred_R6D_3D = \
            ret_dict['pred_shape'], ret_dict['pred_pose'], ret_dict['pred_trans'], ret_dict['pred_R6D_3D']
        gt_shape, gt_pose, gt_trans, gt_joints = sample['betas'], sample['smpl_pose'], sample['global_trans'], sample['smpl_joints']
        # import pdb; pdb.set_trace()
        pred_pose_all = torch.cat([pred_R6D_3D, pred_pose], dim = 1)
        loss_shape = self.criterion_parameters(pred_shape, gt_shape)
        loss_pose = self.criterion_parameters(pred_pose_all, gt_pose)
        loss_trans = (pred_trans - gt_trans).norm(dim = -1).mean()
        # import pdb; pdb.set_trace()
        smpl_out = smpl_model(betas = pred_shape, body_pose = pred_pose, global_orient = pred_R6D_3D, transl = pred_trans)
        smpl_out_joints = smpl_out.joints[:,self.JOINTS_IDX]
        loss_joint = (gt_joints - smpl_out_joints).norm(dim = -1).mean()
        loss = loss_shape + loss_pose + loss_trans + loss_joint
        return {'loss':loss, 'loss_shape':loss_shape, 'loss_pose':loss_pose, 'loss_trans': loss_trans, 'loss_joint': loss_joint}
# --- point net encoder ---- for baseline ----
class point_net_encoder(nn.Module):
    def __init__(self,
                 shape_classes=10,
                 pose_classes=72,
                 trans_classes=3,
                 normal_channel=False,
                 smpl_mean_file="/home/xinxin/code/data/SMPLmodel/neutral_smpl_mean_params.h5",
                 device="cpu"
                 ):
        super(point_net_encoder, self).__init__()
        self.smpl_mean_file = smpl_mean_file
        self.device = device

        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel

        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=in_channel, device=device)

        self.fc1 = nn.Linear(1024  + 72 + 10 + 3, 1024)
        self.fc1_p = nn.Linear(1024, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.drop2 = nn.Dropout(0.5)

        self.decpose = nn.Linear(1024, pose_classes)
        self.decshape = nn.Linear(1024, shape_classes)
        self.dectrans = nn.Linear(1024, trans_classes)

        file = h5py.File(self.smpl_mean_file, 'r')
        init_pose = torch.from_numpy(file['pose'][:]).unsqueeze(0).float()
        init_shape = torch.from_numpy(file['shape'][:]).unsqueeze(0).float()
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)

    def forward(self,
                xyz,
                init_pose=None,
                init_shape=None,
                init_trans=None,
                n_iter=3):

        batch_size, _, _ = xyz.shape

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_trans is None:
            init_trans = torch.zeros([batch_size, 3]).to(self.device)

        xf, trans, trans_feat = self.feat(xyz)


        pred_pose = init_pose
        pred_shape = init_shape
        pred_trans = init_trans

        for i in range(n_iter):
            xc = torch.cat([xf, pred_pose, pred_shape, pred_trans], 1)

            # add relu and BN here
            xc = self.drop1(F.relu(self.bn1(self.fc1(xc))))
            xc = self.drop2(F.relu(self.bn2(self.fc2(xc))))

            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_trans = self.dectrans(xc) + pred_trans

        return pred_shape, pred_pose, pred_trans
    