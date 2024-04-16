from .pct.pct_pose import PCTv2_SegReg
from .pose2mesh.pointcloud2mesh_net import pointcloud2mesh_net, p2m_svd, JOINT_REGRESSOR_H36M_correct, H36M_J17_TO_J15, flip_pairs_15, skeleton_15
from ._smpl import SMPL
import torch
import torch.nn as nn
from .lpformer.lpformer import BertSelfAttention
from .pose2mesh.coarsening import HEM, lmax_L, rescale_L
from .pose2mesh.graph_utils import build_graph
from .pose2mesh.loss import get_loss
import numpy as np
from .graphormer.modeling_bert import BertConfig
from .graphormer.modeling_graphormer import Graphormer, GraphConvolution, GraphResBlock
import copy
from .smpl_param_regressor import SMPLParamRegressor
from .pct_config import pct_config
from .smpl_hybrik.regressor import Hybrik
import pickle
from .smpl_hybrik.SMPL import SMPL_layer
from .VoteHMR import GlobalParamRegressor, LocalParamRegressor
from .LiDARCap.modules.regressor import Regressor
import smplx

class pose_mesh_net(nn.Module):
    # PRN + Pose2Mesh: https://arxiv.org/abs/2008.09047
    def __init__(self, pose_dim = 15, **kwargs):
        super().__init__()
        self.pct_pose = PCTv2_SegReg(pct_config)
        self.p2m_ = pointcloud2mesh_net()

    def forward(self, pcd):
        ret_dict = self.pct_pose(pcd)
        pose = ret_dict['pose']
        
        mesh_dict = self.p2m_(pose)
        ret_dict.update(mesh_dict)
        return ret_dict

    def all_loss(self, ret_dict, sample):
        loss_dict_pose = self.pct_pose.all_loss(ret_dict, sample)
        loss_dict_mesh = self.p2m_.all_loss(ret_dict, sample)
        # loss3 = self.criterion[2](pred_mesh, gt_mesh) * self.edge_weight
        for key in loss_dict_mesh.keys():
            if key in loss_dict_pose.keys():
                loss_dict_pose[key] += loss_dict_mesh[key]
            else:
                loss_dict_pose[key] = loss_dict_mesh[key]
        return loss_dict_pose

class pose_mesh_net_svd(nn.Module):
    # PRN + Pose2Mesh + MeshIK
    def __init__(self, train_p2m = False, **kwargs):
        super().__init__()
        self.p2m_ = pose_mesh_net()
        j36m = np.load('models/data/J_regressor_h36m_correct.npy')
        self.smpl_ = SMPL_layer('smplx_models/smpl/SMPL_NEUTRAL.pkl',num_joints = 24, h36m_jregressor=j36m)
        self.beta_regressor = nn.Sequential(
            nn.Linear(23+23+23, 96),
            nn.Linear(96, 24),
            nn.Linear(24, 10)
        )
        self.train_p2m = train_p2m
        if not self.train_p2m:
            for param in self.p2m_.parameters():
                param.requires_grad = False

    def forward(self, pcd):
        ret_dict = self.p2m_(pcd)
        mesh_ = ret_dict['mesh_out'].detach()
        batch_size = mesh_.shape[0]
        pose_skeleton = self.smpl_.J_regressor @ mesh_
        rel_ps = (pose_skeleton[:,1:] - pose_skeleton[:, self.smpl_.parents[1:]].clone()).norm(dim = -1)
        template_skeleton = self.smpl_.J_regressor @ self.smpl_.v_template
        rel_ts = (template_skeleton[1:] - template_skeleton[self.smpl_.parents[1:]].clone()).norm(dim = -1)
        rel_ts = rel_ts.unsqueeze(0).repeat(batch_size, 1).to(rel_ps.device) #[B,23,3]
        in_ds = rel_ps - rel_ts
        
        inp_feat = torch.cat([rel_ps, rel_ts, in_ds], dim = -1)
        betas = self.beta_regressor(inp_feat)
        output = self.smpl_.meshik(
            pred_verts=mesh_,
            betas=betas
        )
        verts = output.vertices
        vert_shift = [verts - mesh_]
        ret_dict.update({'mesh_refine':verts, 'vert_shift':vert_shift, 'pose_theta': output.theta, 'pose_beta':betas, 'trans': output.trans})
        return ret_dict

    def all_loss(self, ret_dict, sample):
        before_mesh = ret_dict['mesh_out']
        final_mesh = ret_dict['mesh_refine']
        gt_mesh = sample['smpl_verts_local']
        before_sk = self.smpl_.J_regressor @ before_mesh
        after_sk = self.smpl_.J_regressor @ final_mesh
        before_len = (before_sk[:,1:] - before_sk[:, self.smpl_.parents[1:]].clone()).norm(dim = -1)
        after_len = (after_sk[:,1:] - after_sk[:, self.smpl_.parents[1:]].clone()).norm(dim = -1)
        loss_sk = (before_len - after_len).abs().mean()
        
        if not self.train_p2m:
            loss_dict_pose = self.p2m_.p2m_.get_final_mesh_loss(final_mesh, gt_mesh)
        else:
            loss_dict_pose = self.p2m_.p2m_.all_loss(ret_dict, sample)

        loss_dict_pose['loss_refine'] = loss_sk
        if 'vert_shift' in ret_dict.keys():
            consist_loss = 0
            for vs in ret_dict['vert_shift']:
                consist_loss += vs.norm(dim = -1).mean()
            loss_dict_pose['loss_refine'] += consist_loss
            # loss_dict_pose['loss_shift'] = consist_loss
            # loss_dict_pose['loss'] += consist_loss
        return loss_dict_pose

class pose_mesh_hybrik(nn.Module):
    # PRN + HybrIK: https://arxiv.org/abs/2011.14672
    def __init__(self, pct_config, **kwargs):
        super().__init__()
        self.pct_pose = PCTv2_SegReg(pct_config)
        self.p2m_ = Hybrik()
        self.mesh_faces = self.p2m_.smpl_.faces.astype(np.int32)

    def forward(self, pcd):
        ret_dict = self.pct_pose(pcd)
        pose = ret_dict['pose']
        mesh_dict = self.p2m_(pose)
        ret_dict.update(mesh_dict)
        return ret_dict

    def all_loss(self, ret_dict, sample):
        loss_dict_pose = self.pct_pose.all_loss(ret_dict, sample)
        loss_dict_mesh = self.p2m_.all_loss(ret_dict, sample)
        for key in loss_dict_mesh.keys():
            if key in loss_dict_pose.keys():
                loss_dict_pose[key] += loss_dict_mesh[key]
            else:
                loss_dict_pose[key] = loss_dict_mesh[key]
        return loss_dict_pose

class votehmr(nn.Module):
    # VoteHMR: https://arxiv.org/abs/2110.08729
    def __init__(self, pct_cfg = None, **kwargs):
        super().__init__()
        if pct_cfg is not None:
            self.pct_pose = PCTv2_SegReg(pct_cfg)
        else:
            self.pct_pose = PCTv2_SegReg(pct_config)
        self.gpr = GlobalParamRegressor()
        graph_adj = torch.eye(23)
        bone_link = [[1,4], [2,5], [3,6], [4,7], [5,8], [6,9], [7,10], [8,11], [9,12], [9,13], [9,14], [12,15], [13,16], \
                      [14,17], [16,18], [17,19], [18,20], [19,21], [20,22], [21,23]]
        for bl in bone_link:
            x,y = bl
            graph_adj[x-1, y-1] = 1
            graph_adj[y-1, x-1] = 1
        self.global_linear = nn.Linear(96, 48)
        graph_adj = graph_adj / graph_adj.sum(dim = -1, keepdim = True)
        self.lpr = LocalParamRegressor(graph_adj)
        self.body_model = smplx.create(gender='neutral', model_type = 'smpl',\
                          model_path='./smplx_models/')
        self.joint_24_regressor = self.body_model.J_regressor.clone().float()
        self.JOINTS_IDX = [0, 1, 2, 4, 5, 7, 8, 12, 15, 16, 17, 18, 19, 20, 21]
        self.criterion_beta = torch.nn.L1Loss('mean')
        self.criterion_rotmat = torch.nn.L1Loss('mean')
        from ChamferDistancePytorch.chamfer3D import dist_chamfer_3D
        self.chd = dist_chamfer_3D.chamfer_3DDist()

    def forward(self, pcd):
        ret_dict = self.pct_pose(pcd)
        pose = ret_dict['pose']
        pose_feat = ret_dict['skeleton_feat'].permute(0,2,1)
        point_feat = ret_dict['feat']
        pose_feat_mean = point_feat.mean(dim = 1, keepdim = True)
        pose_feat_max = point_feat.max(dim = 1, keepdim = True)[0]
        global_feat = torch.cat([pose_feat_max, pose_feat_mean], dim = -1)
        global_feat = self.global_linear(global_feat).permute(0,2,1)
        global_dict = self.gpr(global_feat, pose_feat)
        betas, g_o, trans = global_dict['betas'], global_dict['global_orient'], global_dict['trans']
        local_dict = self.lpr(pose_feat[...,1:])
        l_o = local_dict['local_orient']
        verts = self.body_model(betas = betas, body_pose = l_o, global_orient = g_o, transl = trans, pose2rot = False).vertices
        ret_dict.update(global_dict)
        ret_dict.update(local_dict)
        ret_dict['mesh_out'] = verts
        ret_dict['joints'] = (self.joint_24_regressor.to(verts.device) @ verts)[:,self.JOINTS_IDX]
        return ret_dict

    def batch_rodrigues(self,
        rot_vecs: torch.Tensor,
        epsilon: float = 1e-8,
        ) -> torch.Tensor:
        ''' Calculates the rotation matrices for a batch of rotation vectors
            Parameters
            ----------
            rot_vecs: torch.tensor Nx3
                array of N axis-angle vectors
            Returns
            -------
            R: torch.tensor Nx3x3
                The rotation matrices for the given axis-angle parameters
        '''

        batch_size = rot_vecs.shape[0]
        device, dtype = rot_vecs.device, rot_vecs.dtype

        angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
        rot_dir = rot_vecs / angle

        cos = torch.unsqueeze(torch.cos(angle), dim=1)
        sin = torch.unsqueeze(torch.sin(angle), dim=1)

        # Bx1 arrays
        rx, ry, rz = torch.split(rot_dir, 1, dim=1)
        K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

        zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
        K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
            .view((batch_size, 3, 3))

        ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
        rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
        return rot_mat

    def all_loss(self, ret_dict, sample):
        batch_size = ret_dict['pose'].shape[0]
        loss_dict_pose = self.pct_pose.all_loss(ret_dict, sample)
        gt_pose, gt_beta, gt_verts = sample['smpl_pose'], sample['betas'], sample['smpl_verts_local']
        gt_rot_mats = self.batch_rodrigues(gt_pose.view(-1, 3)).view(
            [batch_size, -1, 3, 3])
        pred_rot_global, pred_rot_local, pred_beta, pred_verts = ret_dict['global_orient'], ret_dict['local_orient'], ret_dict['betas'], ret_dict['mesh_out']
        pred_rot = torch.cat([pred_rot_global, pred_rot_local], dim = 1) #[B,N,3,3]
        loss_smpl = self.criterion_beta(gt_beta, pred_beta) + self.criterion_rotmat(gt_rot_mats, pred_rot)
        diff_orth = torch.einsum('bnjk,bnkl->bnjl', [pred_rot, pred_rot.permute(0,1,3,2)]) - torch.eye(3).to(pred_rot.device).unsqueeze(0).unsqueeze(0)
        loss_orth = (diff_orth ** 2).sum(dim = [2,3]).sqrt().mean()
        loss_joints = (ret_dict['joints'] - sample['smpl_joints_local']).norm(dim = -1).mean()
        loss_vert = (gt_verts - pred_verts).norm(dim = -1).mean()
        pcd = sample['human_points_local']
        dist1, dist2, idx1, idx2 = self.chd(pred_verts,pcd)
        chamfer_loss = (torch.mean(dist1)) + (torch.mean(dist2))
        loss_dict_pose['loss'] += (loss_smpl + loss_orth + loss_vert + loss_joints + chamfer_loss) * 10
        # chd = chamfer_dist()
        return loss_dict_pose

def graph_upsample(x, p_):
    if p_ > 1:
        x = x.permute(0, 2, 1).contiguous()  # x = B x F x V
        x = nn.Upsample(scale_factor=p_)(x)  # B x F x (V*p)
        x = x.permute(0, 2, 1).contiguous()  # x = B x (V*p) x F
        return x
    else:
        return x

class pose_dir(nn.Module):
    def __init__(self, device = 'cuda'):
        super().__init__()
        self.pct_pose = PCTv2_SegReg(pct_config)
        self.smpl = SMPL()
        num_heads = 3
        self.emb_dim = 48
        self.device = device
        mesh_face = self.smpl.faces
        self.criterion = get_loss(faces=self.smpl.faces)
        self.mesh_face = mesh_face
        self.criterion_regr = nn.MSELoss().to(self.device)
        self.fc = nn.Sequential(
            nn.Linear(15 * 3, 1024),
            nn.Linear(1024, 24 * self.emb_dim)
        )
        self.pose_decoder = nn.Sequential(
            nn.Linear(self.emb_dim, 16),
            nn.Linear(16, 3)
        )
        self.trans_decoder = nn.Sequential(
            nn.Linear(96, 48),
            nn.Linear(48, 16),
            nn.Linear(16, 3)
        )
        self.beta_decoder = nn.Sequential(
            nn.Linear(self.emb_dim * 2, 48),
            nn.Linear(48, 10)
        )
        self.joint_regressor = torch.tensor(np.load(JOINT_REGRESSOR_H36M_correct))
        self.valid_kpts_indx = H36M_J17_TO_J15
        self.normal_weight = 0.1 #0.1
        self.joint_weight = 1.0
        self.edge_weight = 20
        self.final_mesh_weight = 1#3
        self.body_model = smplx.create(gender='neutral', model_type = 'smpl',\
                          model_path='./smplx_models/')

    def forward(self, pcd):
        ret_dict = self.pct_pose(pcd)
        B, N = pcd.shape[:2]
        pose = ret_dict['pose'].view(B,-1)
        pose_feat = self.fc(pose).view(B,24,-1)
        smpl_pose = self.pose_decoder(pose_feat).view(B,-1)
        mean_pose_feat = pose_feat.mean(dim = 1, keepdim = True)
        max_pose_feat = pose_feat.max(dim = 1, keepdim = True)[0]
        beta_feat = torch.cat([mean_pose_feat, max_pose_feat], dim = -1).squeeze(1)
        beta_pose = self.beta_decoder(beta_feat) #[B,10]
        pcd_feat = ret_dict['feat'] #[B,N,C]
        mean_pcd_feat, max_pcd_feat = pcd_feat.mean(dim = 1, keepdim = True), pcd_feat.max(dim = 1, keepdim = True)[0]
        pcd_f = torch.cat([mean_pcd_feat, max_pcd_feat], dim = -1).squeeze(1)
        trans = self.trans_decoder(pcd_f)
        verts = self.body_model(betas = beta_pose, body_pose = smpl_pose[...,3:], global_orient = smpl_pose[...,:3], transl = trans).vertices
        ret_dict.update({'mesh_out':verts})
        return ret_dict

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl = None):
        """Compute SMPL parameter loss for the examples that SMPL annotations are available."""
        if has_smpl is not None:
            pred_rotmat_valid = pred_rotmat[has_smpl == 1].view(-1, 3, 3)
            gt_rotmat_valid = rodrigues(gt_pose[has_smpl == 1].view(-1,3))
            pred_betas_valid = pred_betas[has_smpl == 1]
            gt_betas_valid = gt_betas[has_smpl == 1]
        else:
            pred_rotmat_valid = pred_rotmat.view(-1, 3, 3)
            gt_rotmat_valid = rodrigues(gt_pose.view(-1,3))
            pred_betas_valid = pred_betas
            gt_betas_valid = gt_betas
        # import pdb; pdb.set_trace()
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)
        return loss_regr_pose, loss_regr_betas

    def edge_loss(self, coord_out, coord_gt, mesh_face):
        face = torch.LongTensor(mesh_face).cuda()

        d1_out = torch.sqrt(
            torch.sum((coord_out[:, face[:, 0], :] - coord_out[:, face[:, 1], :]) ** 2 + 1e-8, 2, keepdim=True))
        d2_out = torch.sqrt(
            torch.sum((coord_out[:, face[:, 0], :] - coord_out[:, face[:, 2], :]) ** 2 + 1e-8, 2, keepdim=True))
        d3_out = torch.sqrt(
            torch.sum((coord_out[:, face[:, 1], :] - coord_out[:, face[:, 2], :]) ** 2 + 1e-8, 2, keepdim=True))

        d1_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 0], :] - coord_gt[:, face[:, 1], :]) ** 2, 2, keepdim=True))
        d2_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 0], :] - coord_gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))
        d3_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 1], :] - coord_gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))
 
        diff1 = torch.abs(d1_out - d1_gt)
        diff2 = torch.abs(d2_out - d2_gt)
        diff3 = torch.abs(d3_out - d3_gt)
        # import pdb; pdb.set_trace()
        loss = torch.cat((diff1, diff2, diff3), 1)# + torch.cat((diff1 / d1_gt, diff2 / d2_gt, diff3 / d3_gt), 1)
        return loss.mean()

    def get_final_mesh_loss(self, pred_mesh, gt_mesh, val_mesh = None):
        # pred_mesh = ret_dict['mesh_out']
        # gt_mesh = sample['smpl_verts_local'].to(pred_mesh.device)
        # import pdb; pdb.set_trace()
        j_r = self.joint_regressor.to(pred_mesh.device).float()
        pred_pose = (j_r @ pred_mesh)[:,self.valid_kpts_indx]
        gt_reg3dpose = (j_r @ gt_mesh)[:,self.valid_kpts_indx]
        # 
        loss1, loss2, loss4 = self.criterion[0](pred_mesh, gt_mesh, val_mesh) * self.final_mesh_weight,  \
                                                self.normal_weight * self.criterion[1](pred_mesh, gt_mesh) * self.final_mesh_weight, \
                                                self.joint_weight * self.criterion[3](pred_pose,  gt_reg3dpose, val_mesh) * self.final_mesh_weight
        loss3 = self.criterion[2](pred_mesh, gt_mesh) * self.edge_weight * self.final_mesh_weight
        # loss5 = self.criterion[5](pred_mesh, gt_mesh) * self.edge_weight * 10
        loss5 = 0
        loss = loss1 + loss4 + loss2 
        # print(loss1, loss2, loss3, loss5)
        # import pdb; pdb.set_trace()
        return {'loss': loss, 'loss_joint':loss4, 'loss_vert':loss1 + loss2, 'edge_loss': loss3 + loss5}

    def get_mid_mesh_loss(self, ret_dict, sample):
        pred_meshes = ret_dict['pose_lister']
        loss_mesh = 0
        loss_edge = 0
        gt_mesh = sample['smpl_verts_local']
        gt_meshes = []
        gt_meshes.append(gt_mesh)
        
        for ind, indx in enumerate(self.multi_index):
            sub_mesh_this = (gt_meshes[-1][:,indx[:,0]] + gt_meshes[-1][:,indx[:,1]] ) / 2
            gt_meshes.append(sub_mesh_this)
            # import pdb; pdb.set_trace()
            loss1 = self.criterion[0](pred_meshes[-ind-1], gt_meshes[ind], None)
            loss_mesh += loss1
            if self.cfg.LOSS.MID_EDGE_LOSS:
                loss2 = self.edge_loss(pred_meshes[-ind-1], gt_meshes[ind], self.mesh_faces[ind])
                loss_edge += loss2
        return {'loss_vert': loss_mesh, 'edge_loss' : loss_edge}

    def get_smpl_mesh_loss(self, ret_dict, sample, val_mesh = None):
        pred_mesh = ret_dict['mesh_smpl']
        gt_mesh = sample['smpl_verts_local'].to(pred_mesh.device)
        gt_pose = sample['smpl_pose']
        gt_betas = sample['betas']
        # gt_trans = sample['global_trans']
        pred_rotmat = ret_dict['pred_rot']
        pred_shape = ret_dict['pred_shape']
        pred_trans = ret_dict['pred_trans']
        j_r = self.joint_regressor.to(pred_mesh.device).float()
        pred_pose = (j_r @ pred_mesh)[:,self.valid_kpts_indx]
        gt_reg3dpose = (j_r @ gt_mesh)[:,self.valid_kpts_indx]
        # import pdb; pdb.set_trace()
        loss_regr_pose, loss_regr_betas = self.smpl_losses(pred_rotmat, pred_shape, gt_pose, gt_betas)
        # import pdb; pdb.set_trace()
        loss_smpl = loss_regr_pose + 0.1 * loss_regr_betas
        loss1, loss2, loss4 = self.criterion[0](pred_mesh, gt_mesh, val_mesh),  \
                                                self.criterion[1](pred_mesh, gt_mesh), \
                                                self.joint_weight * self.criterion[3](pred_pose,  gt_reg3dpose, val_mesh)
        loss3 = 0
        loss5 = 0
        loss = loss1 + loss2 + loss4 + loss_smpl
        # print(loss_regr_pose, loss_regr_betas, self.criterion_regr(pred_trans, gt_trans))
        return {'loss': loss, 'loss_joint':loss4, 'loss_vert':loss1 + loss2 + loss_smpl, 'edge_loss': loss3 + loss5}

    def all_loss(self, ret_dict, sample):
        loss_dict_pose = self.pct_pose.all_loss(ret_dict, sample)
        
        pred_mesh = ret_dict['mesh_out']
        gt_mesh = sample['smpl_verts_local'].to(pred_mesh.device)
        loss_dict_mesh = self.get_final_mesh_loss(pred_mesh, gt_mesh)
        for key in loss_dict_mesh.keys():
            if key in loss_dict_pose.keys():
                loss_dict_pose[key] += loss_dict_mesh[key]
            else:
                loss_dict_pose[key] = loss_dict_mesh[key].clone()

        return loss_dict_pose

class pose_meshgraphormer(nn.Module):
    # PRN+MRN
    def __init__(self, pose_dim = 15, device = 'cuda', pmg_cfg = None, use_collision = False, **kwargs):
        super().__init__()
        self.cfg = pmg_cfg
        self.pct_pose = PCTv2_SegReg(pct_config)
        if not self.cfg.NETWORK.PRN_TRAINED:
            for p in self.pct_pose.parameters():
                p.requires_grad = False
        self.smpl = SMPL()
        num_heads = 3
        self.emb_dim = 48
        # self.mesh_model = SMPL()
        self.device = device
        mesh_face = self.smpl.faces
        self.criterion = get_loss(faces=self.smpl.faces)
        mesh_adj = build_graph(mesh_face, mesh_face.max() + 1)
        self.mesh_face = mesh_face
        self.criterion_regr = nn.MSELoss().to(self.device)
        levels = 9
        self.graphs, self.parents = HEM(mesh_adj, levels)
        # self.multi_index, self.par_index, self.indexes, self.mesh_faces = 
        graphs_ = []
        for L in self.graphs:
            lm = lmax_L(L)
            gra_ = rescale_L(L, lm)
            gra_ = self.sparse_python_to_torch(gra_).to(device)
            graphs_.append(gra_)
        self.graphs = graphs_
        self.get_multilevel_index()

        self.fc = nn.Sequential(
            nn.Linear(15 * 3, 1024),
            nn.Linear(1024, 24 * self.emb_dim)
        )
        self.location_decoder = nn.Sequential(
            nn.Linear(self.emb_dim, 16),
            nn.Linear(16, 3)
        )
        if self.cfg.NETWORK.UP_TYPE == 'graph_mlp':
            self.nl1 = nn.Sequential(
                nn.Linear(self.emb_dim, self.emb_dim * 2),
                nn.Linear(self.emb_dim * 2, self.emb_dim)
            )
            self.nl2 = nn.Sequential(
                nn.Linear(self.emb_dim, self.emb_dim * 2),
                nn.Linear(self.emb_dim * 2, self.emb_dim)
            )
        # elif self.cfg.NETWORK.UP_TYPE == 'upsample':

        p2m_ = []
        for i in range(len(self.graphs) - 3):
            config1 = BertConfig.from_pretrained('./models/graphormer')
            config1.output_attentions = False
            config1.img_feature_dim = self.emb_dim
            config1.output_feature_dim = self.emb_dim
            config1.hidden_size = 24
            config1.graph_conv = True
            config1.mesh_type = 'body'
            config1.intermediate_size = 64
            config1.num_attention_heads = 3
            config1.num_hidden_layers = 2
            if self.cfg.NETWORK.USE_PCD == True:
                config1.max_position_embeddings = 1024 + self.graphs[-i-1].shape[0]
            else:
                config1.max_position_embeddings = self.graphs[-i-1].shape[0]
            gphmr_ = Graphormer(config1, adj_mat = self.graphs[-i-1])
            p2m_.append(gphmr_)
        self.p2m_ = nn.ModuleList(p2m_)
        
        config1.max_position_embeddings = 1024 + self.graphs[2].shape[0] if self.cfg.NETWORK.PCD_LAST else self.graphs[2].shape[0]
        self.mesh_layer = Graphormer(config1, adj_mat = self.graphs[2])
        self.smpl_param_regressor = SMPLParamRegressor()
        self.up1 = nn.Linear(1946, 3679)
        self.up2 = nn.Linear(3679, 6890)

        self.joint_regressor = torch.tensor(np.load(JOINT_REGRESSOR_H36M_correct))
        self.valid_kpts_indx = H36M_J17_TO_J15
        self.normal_weight = 0.1 #0.1
        self.joint_weight = 1.0
        self.edge_weight = 20
        self.final_mesh_weight = 1#3

        self.use_collision = use_collision
        if self.use_collision:
            max_collisions = 8
            self.search_tree = BVH(max_collisions=max_collisions)
            df_cone_height = 0.0001
            point2plane = False
            penalize_outside = True
            self.pen_distance = \
                collisions_loss.DistanceFieldPenetrationLoss(
                    sigma=df_cone_height, point2plane=point2plane,
                    vectorized=True, penalize_outside=penalize_outside)
            self.coll_loss_weight = 0.01

    @staticmethod
    def sparse_python_to_torch(sp_python):
        L = sp_python.tocoo()
        indices = np.column_stack((L.row, L.col)).T
        indices = indices.astype(np.int64)
        indices = torch.from_numpy(indices)
        indices = indices.type(torch.LongTensor)
        L_data = L.data.astype(np.float32)
        L_data = torch.from_numpy(L_data)
        L_data = L_data.type(torch.FloatTensor)
        L = torch.sparse.FloatTensor(indices, L_data, torch.Size(L.shape))
        # import pdb; pdb.set_trace()
        return L

    def get_multilevel_index(self):
        all_index = []
        for ind, parent in enumerate(self.parents):
            # import pdb; pdb.set_trace()
            ind_this = []
            for indx in range(self.graphs[ind+1].shape[0]):
                indexs = np.where(parent == indx)[0]
                ind_this.append(indexs) if len(indexs) == 2 else ind_this.append(np.array([indexs[0], indexs[0]]))
            all_index.append(np.stack(ind_this, axis = 0))
        par_index = []
        indexes = []
        mesh_face = []
        mesh_face.append(self.mesh_face)
        for par in self.parents:
            par_this = []
            index_this = []
            mef_this = copy.deepcopy(mesh_face[-1])
            for ind_par, pk in enumerate(par):
                index_this.append(0) if pk in par_this else index_this.append(1)
                par_this.append(pk)
                mef_this[mesh_face[-1]==ind_par] = pk
            mef_indx = (mef_this[:,0] != mef_this[:,1]) & (mef_this[:,0] != mef_this[:,2]) & (mef_this[:,1] != mef_this[:,2])
            mef_this = mef_this[mef_indx,:]
            mesh_face.append(mef_this)
            par_index.append(np.array(par_this))
            indexes.append(np.array(index_this))
        graphs = []
        for iks in range(len(mesh_face)):
            g0 = self.graphs[iks]
            gxo = torch.eye(g0.shape[0])
            for ind in mesh_face[iks]:
                x,y,z = ind
                gxo[x,y] = 1 
                gxo[x,z] = 1
                gxo[y,z] = 1
                gxo[y,x] = 1 
                gxo[z,x] = 1
                gxo[z,y] = 1
            gxo /= gxo.sum(dim = 1, keepdim = True)
            graphs.append(gxo.to_sparse().to(self.device))
        self.multi_index = all_index
        self.par_index = par_index
        self.indexes = indexes
        self.mesh_faces = mesh_face
        # print(self.cfg.NETWORK.ADJ_TYPE)
        if self.cfg.NETWORK.ADJ_TYPE == 'adj':
            self.graphs = graphs

        # self.graphs = graphs
        # return all_index, par_index, indexes, mesh_face

    def get_multilevel_points(self, full_verts, parents, graphs):
        all_verts = []
        all_verts.append(full_verts)
        for ind, parent in enumerate(parents):
            verts_parent = np.zeros([graphs[ind+1].shape[0], 3])
            # graph_index = {}
            for indx in range(verts_parent.shape[0]):
                indexs = np.where(parent == indx)[0]
                verts_parent[indx] = np.mean(all_verts[-1][indexs], axis = 0)
            all_verts.append(verts_parent)
        # import pdb; pdb.set_trace()
        return all_verts

    def forward(self, pcd):
        ret_dict = self.pct_pose(pcd)
        B, N = pcd.shape[:2]
        pose = ret_dict['pose']
        
        pose_lister = []
        pose_lister.append(pose)
        ori_feat = self.fc(pose.reshape(B,-1)).reshape(B,-1,self.emb_dim)
        # pose_24 = self.location_decoder(ori_feat)
        # pose_lister.append(pose_24)
        
        all_feat = torch.cat([ori_feat, ret_dict['feat']], dim = 1) if self.cfg.NETWORK.USE_PCD else ori_feat
        num_verts = [self.graphs[-ind-1].shape[0] for ind in range(len(self.graphs))]
        for ind, p2m_layer in enumerate(self.p2m_):
            #自注意力
            out_feat = p2m_layer(all_feat)
            mid_feat = out_feat[:,:-N].clone() if self.cfg.NETWORK.USE_PCD else out_feat
            pose_pred = self.location_decoder(mid_feat)
            pose_lister.append(pose_pred)
            
            #上采样
            if self.cfg.NETWORK.USE_PCD:
                pcd_feat = out_feat[:,-N:].clone()
            if self.cfg.NETWORK.UP_TYPE == 'graph_mlp':
                mig_feat = out_feat[:,self.par_index[-ind - 1]].clone()
                nl1 = self.nl1(mig_feat)
                nl2 = self.nl2(mig_feat)
                index_this = self.indexes[-ind - 1]
                mid_feat = mig_feat.clone()
                mid_feat[:,index_this==0] = nl1[:,index_this==0]
                mid_feat[:,index_this==1] = nl2[:,index_this==1]
            elif self.cfg.NETWORK.UP_TYPE == 'upsamle':
                mid_feat = out_feat[:,:num_verts[ind]]
                # import pdb; pdb.set_trace()
                mid_feat = graph_upsample(mid_feat, num_verts[ind+1] / num_verts[ind])
            
            all_feat = torch.cat([mid_feat, pcd_feat], dim = 1) if self.cfg.NETWORK.USE_PCD else mid_feat
            if ind == len(self.p2m_) - 1:
                if self.cfg.NETWORK.PCD_LAST:
                    mid_feat = self.mesh_layer(all_feat)
                    mid_feat = mid_feat[:,:num_verts[ind + 1]]
                else:
                    mid_feat = self.mesh_layer(mid_feat)
                pose_pred = self.location_decoder(mid_feat)
                pose_lister.append(pose_pred)
        # import pdb; pdb.set_trace()
        # local_mesh = pose_lister[-1]
        # pred_rotmat, pred_shape, pred_trans = self.smpl_param_regressor(local_mesh.view(B,-1).detach())
        # pred_vertices_smpl = self.smpl(pred_rotmat, pred_shape)
        # pred_vertices_smpl += pred_trans.unsqueeze(1)
        sub_feat = self.up1(mid_feat.permute(0,2,1))
        pose_lister.append(self.location_decoder(sub_feat.permute(0,2,1)))
        sub2_feat = self.up2(sub_feat)
        final_vert_feat = sub2_feat.permute(0,2,1)
        pose_lister.append(self.location_decoder(final_vert_feat))

        ret_dict.update({'pose_lister':pose_lister, 'mesh_out':pose_lister[-1], 'vert_feat':final_vert_feat})
        # 'mesh_smpl':pred_vertices_smpl, 'pred_rot':pred_rotmat, 'pred_shape':pred_shape, 'pred_trans':pred_trans})
        return ret_dict

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl = None):
        """Compute SMPL parameter loss for the examples that SMPL annotations are available."""
        if has_smpl is not None:
            pred_rotmat_valid = pred_rotmat[has_smpl == 1].view(-1, 3, 3)
            gt_rotmat_valid = rodrigues(gt_pose[has_smpl == 1].view(-1,3))
            pred_betas_valid = pred_betas[has_smpl == 1]
            gt_betas_valid = gt_betas[has_smpl == 1]
        else:
            pred_rotmat_valid = pred_rotmat.view(-1, 3, 3)
            gt_rotmat_valid = rodrigues(gt_pose.view(-1,3))
            pred_betas_valid = pred_betas
            gt_betas_valid = gt_betas
        # import pdb; pdb.set_trace()
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)
        return loss_regr_pose, loss_regr_betas

    def edge_loss(self, coord_out, coord_gt, mesh_face):
        face = torch.LongTensor(mesh_face).cuda()

        d1_out = torch.sqrt(
            torch.sum((coord_out[:, face[:, 0], :] - coord_out[:, face[:, 1], :]) ** 2 + 1e-8, 2, keepdim=True))
        d2_out = torch.sqrt(
            torch.sum((coord_out[:, face[:, 0], :] - coord_out[:, face[:, 2], :]) ** 2 + 1e-8, 2, keepdim=True))
        d3_out = torch.sqrt(
            torch.sum((coord_out[:, face[:, 1], :] - coord_out[:, face[:, 2], :]) ** 2 + 1e-8, 2, keepdim=True))

        d1_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 0], :] - coord_gt[:, face[:, 1], :]) ** 2, 2, keepdim=True))
        d2_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 0], :] - coord_gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))
        d3_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 1], :] - coord_gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))
 
        diff1 = torch.abs(d1_out - d1_gt)
        diff2 = torch.abs(d2_out - d2_gt)
        diff3 = torch.abs(d3_out - d3_gt)
        # import pdb; pdb.set_trace()
        loss = torch.cat((diff1, diff2, diff3), 1)# + torch.cat((diff1 / d1_gt, diff2 / d2_gt, diff3 / d3_gt), 1)
        return loss.mean()

    def get_final_mesh_loss(self, pred_mesh, gt_mesh, val_mesh = None):
        # pred_mesh = ret_dict['mesh_out']
        # gt_mesh = sample['smpl_verts_local'].to(pred_mesh.device)
        # import pdb; pdb.set_trace()
        j_r = self.joint_regressor.to(pred_mesh.device).float()
        pred_pose = (j_r @ pred_mesh)[:,self.valid_kpts_indx]
        gt_reg3dpose = (j_r @ gt_mesh)[:,self.valid_kpts_indx]
        # 
        loss1, loss2, loss4 = self.criterion[0](pred_mesh, gt_mesh, val_mesh) * self.final_mesh_weight,  \
                                                self.normal_weight * self.criterion[1](pred_mesh, gt_mesh) * self.final_mesh_weight, \
                                                self.joint_weight * self.criterion[3](pred_pose,  gt_reg3dpose, val_mesh) * self.final_mesh_weight
        loss3 = self.criterion[2](pred_mesh, gt_mesh) * self.edge_weight * self.final_mesh_weight
        # loss5 = self.criterion[5](pred_mesh, gt_mesh) * self.edge_weight * 10
        loss5 = 0
        loss = loss1 + loss4 + loss2 
        # print(loss1, loss2, loss3, loss5)
        # import pdb; pdb.set_trace()
        return {'loss': loss, 'loss_joint':loss4, 'loss_vert':loss1 + loss2, 'edge_loss': loss3 + loss5}

    def get_mid_mesh_loss(self, ret_dict, sample):
        pred_meshes = ret_dict['pose_lister']
        loss_mesh = 0
        loss_edge = 0
        gt_mesh = sample['smpl_verts_local']
        gt_meshes = []
        gt_meshes.append(gt_mesh)
        
        for ind, indx in enumerate(self.multi_index):
            sub_mesh_this = (gt_meshes[-1][:,indx[:,0]] + gt_meshes[-1][:,indx[:,1]] ) / 2
            gt_meshes.append(sub_mesh_this)
            # import pdb; pdb.set_trace()
            loss1 = self.criterion[0](pred_meshes[-ind-1], gt_meshes[ind], None)
            loss_mesh += loss1
            if self.cfg.LOSS.MID_EDGE_LOSS:
                loss2 = self.edge_loss(pred_meshes[-ind-1], gt_meshes[ind], self.mesh_faces[ind])
                loss_edge += loss2
        return {'loss_vert': loss_mesh, 'edge_loss' : loss_edge}

    def get_smpl_mesh_loss(self, ret_dict, sample, val_mesh = None):
        pred_mesh = ret_dict['mesh_smpl']
        gt_mesh = sample['smpl_verts_local'].to(pred_mesh.device)
        gt_pose = sample['smpl_pose']
        gt_betas = sample['betas']
        # gt_trans = sample['global_trans']
        pred_rotmat = ret_dict['pred_rot']
        pred_shape = ret_dict['pred_shape']
        pred_trans = ret_dict['pred_trans']
        j_r = self.joint_regressor.to(pred_mesh.device).float()
        pred_pose = (j_r @ pred_mesh)[:,self.valid_kpts_indx]
        gt_reg3dpose = (j_r @ gt_mesh)[:,self.valid_kpts_indx]
        # import pdb; pdb.set_trace()
        loss_regr_pose, loss_regr_betas = self.smpl_losses(pred_rotmat, pred_shape, gt_pose, gt_betas)
        # import pdb; pdb.set_trace()
        loss_smpl = loss_regr_pose + 0.1 * loss_regr_betas
        loss1, loss2, loss4 = self.criterion[0](pred_mesh, gt_mesh, val_mesh),  \
                                                self.criterion[1](pred_mesh, gt_mesh), \
                                                self.joint_weight * self.criterion[3](pred_pose,  gt_reg3dpose, val_mesh)
        loss3 = 0
        loss5 = 0
        loss = loss1 + loss2 + loss4 + loss_smpl
        # print(loss_regr_pose, loss_regr_betas, self.criterion_regr(pred_trans, gt_trans))
        return {'loss': loss, 'loss_joint':loss4, 'loss_vert':loss1 + loss2 + loss_smpl, 'edge_loss': loss3 + loss5}

    def all_loss(self, ret_dict, sample):
        if self.cfg.NETWORK.PRN_TRAINED:
            loss_dict_pose = self.pct_pose.all_loss(ret_dict, sample)
        else:
            loss_dict_pose = {}
        pred_mesh = ret_dict['mesh_out']
        gt_mesh = sample['smpl_verts_local'].to(pred_mesh.device)
        loss_dict_mesh = self.get_final_mesh_loss(pred_mesh, gt_mesh)
        # loss_dict_smpl = self.get_smpl_mesh_loss(ret_dict, sample)
        ldm = self.get_mid_mesh_loss(ret_dict, sample)
        for key in loss_dict_mesh.keys():
            if key in loss_dict_pose.keys():
                loss_dict_pose[key] += loss_dict_mesh[key]
            else:
                loss_dict_pose[key] = loss_dict_mesh[key].clone()

        # for key in loss_dict_smpl.keys():
        #     if key in loss_dict_pose.keys():
        #         loss_dict_pose[key] += loss_dict_smpl[key]
        #     else:
        #         loss_dict_pose[key] = loss_dict_smpl[key].clone()
        if self.cfg.LOSS.MID_LOSS:
            loss_dict_pose['loss_vert'] += ldm['loss_vert']
            loss_dict_pose['loss'] += ldm['loss_vert']
            loss_dict_pose['edge_loss'] += ldm['edge_loss']
            
        if self.use_collision:
            batch_size = pred_mesh.shape[0]
            # import pdb; pdb.set_trace()
            triangles = torch.index_select(
                pred_mesh, 1,
                self.mesh_face.view(-1).to(pred_mesh.device)).view(batch_size, -1, 3, 3)

            with torch.no_grad():
                collision_idxs = search_tree(triangles)

            if collision_idxs.ge(0).sum().item() > 0:
                pen_loss = torch.sum(
                    self.coll_loss_weight *
                    pen_distance(triangles, collision_idxs))
                # loss_dict_pose['loss'] += pen_loss
                # loss_dict_pose['loss_vert'] += pen_loss
                loss_dict_pose['pen_loss'] = pen_loss
        return loss_dict_pose

def rodrigues(theta):
    """Convert axis-angle representation to rotation matrix.
    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    return quat2mat(quat)

def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """ 
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat 

class LiDAR_HMR(nn.Module):
    def __init__(self, pose_dim = 15, device = 'cuda', pmg_cfg = None, train_pmg = True, **kwargs):
        super().__init__()
        self.pmg = pose_meshgraphormer(pose_dim=pose_dim, device=device, pmg_cfg=pmg_cfg)
        j36m = np.load('./models/data/J_regressor_h36m_correct.npy')
        self.smpl_ = SMPL_layer('./smplx_models/smpl/SMPL_NEUTRAL.pkl',num_joints = 24, h36m_jregressor=j36m)
        self.beta_regressor = nn.Sequential(
            nn.Linear(48+48+23+23+23, 96),
            nn.Linear(96, 24),
            nn.Linear(24, 10)
        )
        self.train_pmg = train_pmg
        if not self.train_pmg:
            for param in self.pmg.parameters():
                param.requires_grad = False

    def forward(self, pcd):
        ret_dict = self.pmg(pcd)
        mesh_, feat_ = ret_dict['mesh_out'].detach(), ret_dict['vert_feat'].detach()
        batch_size = mesh_.shape[0]
        pose_skeleton = self.smpl_.J_regressor @ mesh_
        rel_ps = (pose_skeleton[:,1:] - pose_skeleton[:, self.smpl_.parents[1:]].clone()).norm(dim = -1)
        template_skeleton = self.smpl_.J_regressor @ self.smpl_.v_template
        rel_ts = (template_skeleton[1:] - template_skeleton[self.smpl_.parents[1:]].clone()).norm(dim = -1)
        rel_ts = rel_ts.unsqueeze(0).repeat(batch_size, 1).to(rel_ps.device) #[B,23,3]
        in_ds = rel_ps - rel_ts
        mean_, max_ = feat_.mean(dim = 1), feat_.max(dim = 1).values
        inp_feat = torch.cat([rel_ps, rel_ts, in_ds, mean_, max_], dim = -1)
        betas = self.beta_regressor(inp_feat)
        output = self.smpl_.meshik(
            pred_verts=mesh_,
            betas=betas
        )
        verts = output.vertices
        vert_shift = [verts - mesh_]
        ret_dict.update({'mesh_refine':verts, 'vert_shift':vert_shift, 'pose_theta': output.theta, 'pose_beta':betas, 'trans': output.trans})
        return ret_dict

    def all_loss(self, ret_dict, sample):
        before_mesh = ret_dict['mesh_out']
        final_mesh = ret_dict['mesh_refine']
        gt_mesh = sample['smpl_verts_local']
        before_sk = self.smpl_.J_regressor @ before_mesh
        after_sk = self.smpl_.J_regressor @ final_mesh
        before_len = (before_sk[:,1:] - before_sk[:, self.smpl_.parents[1:]].clone()).norm(dim = -1)
        after_len = (after_sk[:,1:] - after_sk[:, self.smpl_.parents[1:]].clone()).norm(dim = -1)
        loss_sk = (before_len - after_len).abs().mean()
        
        if not self.train_pmg:
            loss_dict_pose = self.pmg.get_final_mesh_loss(final_mesh, gt_mesh)
        else:
            loss_dict_pose = self.pmg.all_loss(ret_dict, sample)

        loss_dict_pose['loss_refine'] = loss_sk
        if 'vert_shift' in ret_dict.keys():
            consist_loss = 0
            for vs in ret_dict['vert_shift']:
                consist_loss += vs.norm(dim = -1).mean()
            loss_dict_pose['loss_refine'] += consist_loss
        return loss_dict_pose

class LiDARCap_(nn.Module):
    # LiDARCap: https://arxiv.org/abs/2203.14698
    def __init__(self, **kwargs):
        super().__init__()
        self.regressor = Regressor()
        self.criterion_betas = nn.MSELoss()
        self.criterion_trans = nn.MSELoss()
        self.criterion_pose = nn.MSELoss()
        self.criterion_verts = nn.L1Loss(reduction='mean')
        with open('smplx_models/smpl/SMPL_NEUTRAL.pkl', 'rb') as smpl_file:
            smpl_data = pickle.load(smpl_file, encoding='latin1')
        self.joint_24_regressor = torch.tensor(smpl_data['J_regressor'].todense()).float()
        self.JOINTS_IDX = [0, 1, 2, 4, 5, 7, 8, 12, 15, 16, 17, 18, 19, 20, 21]

    def forward(self, data):
        pred = self.regressor(data)
        pred_mesh = pred['mesh_out']
        pred_joints = (self.joint_24_regressor.to(pred_mesh.device) @ pred_mesh) [:, self.JOINTS_IDX]
        # import pdb; pdb.set_trace()
        pred.update({'pred_joints':pred_joints})
        return pred

    def all_loss(self, pred, sample):
        pred_mesh = pred['mesh_out']
        pred_pose = pred['pred_pose']
        pred_joints = pred['pred_joints']
        gt_mesh = sample['smpl_verts_local'].to(pred_mesh.device)
        gt_joints = sample['smpl_joints_local'].to(pred_mesh.device)
        gt_pose = sample['smpl_pose'].to(pred_mesh.device)

        loss_verts = self.criterion_verts(pred_mesh, gt_mesh)
        loss_pose = self.criterion_pose(pred_pose, gt_pose)
        gt_trans, pred_trans = sample['gt_trans'], pred['pred_trans']
        gt_betas, pred_betas = sample['betas'], pred['pred_beta']
        loss_param = self.criterion_betas(gt_betas, pred_betas) + self.criterion_trans(gt_trans, pred_trans)
        loss_joints = (pred_joints - gt_joints).norm(dim = -1).mean()
        loss = loss_verts + loss_pose + loss_param + loss_joints
        loss_dict_mesh = {}
        loss_dict_mesh['loss'] = loss
        loss_dict_mesh['loss_vert'] = loss
        
        return loss_dict_mesh