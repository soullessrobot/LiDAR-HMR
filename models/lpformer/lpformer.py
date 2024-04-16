import torch
import torch.nn as nn
import sys
import os
from ..v2v.v2v_net import V2VNet
import torch.nn.functional as F
# torch.backends.cudnn.enabled = False
import math

class BertSelfAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, output_attentions = False):
        super(BertSelfAttention, self).__init__()
        if emb_dim % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (emb_dim, num_heads))
        self.output_attentions = output_attentions

        self.num_attention_heads = num_heads
        self.attention_head_size = int(emb_dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(emb_dim, self.all_head_size)
        self.key = nn.Linear(emb_dim, self.all_head_size)
        self.value = nn.Linear(emb_dim, self.all_head_size)

        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask = None, head_mask=None,
            history_state=None):
        if history_state is not None:
            x_states = torch.cat([history_state, hidden_states], dim=1)
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(x_states)
            mixed_value_layer = self.value(x_states)
        else:
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask if attention_mask is not None else attention_scores

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape) + hidden_states

        outputs = (context_layer, attention_probs) if self.output_attentions else context_layer
        return outputs

class LPFormer(nn.Module):
    # LPFormer: https://arxiv.org/abs/2306.12525
    def __init__(self, point_dim = 3, voxel_dim = 32, emb_dim = 256, num_heads = 8, num_layers = 4, \
        num_keypoints = 14, bev_dim = 512, compress_dim = 32):
        super(LPFormer, self).__init__()
        self.kpt_embedding = nn.Embedding(num_keypoints, emb_dim)
        self.v2v_part = V2VNet(1, 32)
        self.voxel_length = 2 #
        self.voxel_offset = -1
        self.voxel_size = (64, 64, 64)
        self.num_kp = num_keypoints
        self.mlp_ = nn.Sequential(
            nn.Linear(voxel_dim + point_dim + 3, emb_dim * 4),
            nn.Linear(emb_dim * 4, emb_dim * 2),
            nn.Linear(emb_dim * 2, emb_dim),
        )
        self.KPTR = nn.Sequential(
            BertSelfAttention(emb_dim, num_heads),
            BertSelfAttention(emb_dim, num_heads),
            BertSelfAttention(emb_dim, num_heads),
            BertSelfAttention(emb_dim, num_heads)
        )
        self.mlp_xy = nn.Sequential(
            nn.Linear(emb_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.Linear(8, 2),
        )
        self.mlp_z = nn.Sequential(
            nn.Linear(emb_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.Linear(8, 1),
        )
        self.mlp_vis = nn.Sequential(
            nn.Linear(emb_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.Linear(8, 2),
        )
        self.softmax_ = nn.Softmax(dim = -1)
        self.mlp_seg = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 16),
            nn.Linear(16, num_keypoints + 1),
        )
        # losses
        self.lambda1 = 5
        self.lambda2 = 1
        self.lambda3 = 1
        self.lambda4 = 1

    def forward(self, points_input, center_position):
        #[B,N,3], [B,3] in meter
        bs = points_input.shape[0]
        center_ = center_position.unsqueeze(1) if len(center_position.shape) < 3 else center_position
        local_points = points_input - center_
        points_feat = torch.cat([points_input, local_points], dim = -1)
        voxel_feat = self.get_voxel_(local_points)
        feat_list = []
        feat0 = self.mlp_(torch.cat([points_feat, voxel_feat], dim = -1)) #.float()
        query = self.kpt_embedding(torch.arange(self.num_kp).to(points_input.device)).unsqueeze(0).repeat(bs, 1, 1)#self.kpt_querys.unsqueeze(0).repeat(bs, 1, 1)
        featx = torch.cat([query, feat0], dim = 1) #[B,N_kp+N_points,emb_dim]
        
        featx = self.KPTR(featx)
        feat_kp = featx[...,:self.num_kp,:]
        feat_p = featx[...,self.num_kp:,:]
        xy = self.mlp_xy(feat_kp)
        z = self.mlp_z(feat_kp)
        xyz = torch.cat([xy,z], dim = -1)
        vis = self.softmax_(self.mlp_vis(feat_kp))[...,0]
        seg = self.softmax_(self.mlp_seg(feat_p))
        return {'xyz':xyz, 'vis':vis, 'seg':seg}

    def get_voxel_(self, local_points):
        #[B,N,3]
        bs,N = local_points.shape[:2]
        voxel_input = torch.zeros((bs,) + (1,) + self.voxel_size).to(local_points.device)
        points_loc = ((local_points - self.voxel_offset) / self.voxel_length) #[B,N,3]
        points_index = (points_loc * self.voxel_size[0]).type(torch.int64)
        points_index[points_index > self.voxel_size[0] - 1] = self.voxel_size[0] - 1
        points_index[points_index < 0] = 0
        for b in range(bs):
            voxel_input[b, :, points_index[b,:,0], points_index[b,:,1], points_index[b,:,2]] = 1
        voxel_feat = self.v2v_part(voxel_input)
        sample_grid = points_loc * 2 - 1
        sample_grid = torch.clamp(sample_grid, -1.1, 1.1)
        sample_grid[...,[0,2]] = sample_grid[...,[2,0]] #[B,N,3]
        sample_grid = sample_grid.unsqueeze(1).unsqueeze(1) #[B,1,1,N,3]
        sampled_features = F.grid_sample(voxel_feat, sample_grid).squeeze(3).squeeze(2) #[B,32,N] #.float()
        sampled_features = sampled_features.permute(0,2,1)
        return sampled_features
    
    def xyz_loss(self, xyz, gt_xyz_, flag = None):
        if flag is not None:
            return torch.mean(torch.norm(self.lambda2 * (xyz - gt_xyz_), dim = -1) * flag)
        else:
            return torch.mean(torch.norm(self.lambda2 * (xyz - gt_xyz_), dim = -1))

    def vis_loss(self, vis, gt_vis_):
        return self.lambda3 * F.binary_cross_entropy(vis, gt_vis_)
    
    def seg_loss(self, seg, gt_seg_):
        #[B,Np,Nkp]
        #[B,Np]
        loss_func = nn.CrossEntropyLoss(reduction='mean')     
        return self.lambda4 * loss_func(seg.permute(0,2,1), gt_seg_.to(seg.device))

    def all_loss(self, ret_dict, sample):
        xyz, vis, seg = ret_dict['xyz'], ret_dict['vis'], ret_dict['seg']
        gt_xyz = sample['smpl_joints_local']
        vis_label = sample['vis_label']
        seg_label = sample['seg_label']
        flag = sample['valid_flag'] if 'valid_flag' in sample else None
        xyz_loss = self.xyz_loss(xyz, gt_xyz, flag)
        vis_loss = torch.tensor(0).cuda()
        seg_loss = self.seg_loss(seg, seg_label)
        all_loss = xyz_loss + vis_loss + seg_loss
        return {'loss': all_loss, 'xyz_loss':xyz_loss, 'vis_loss':vis_loss, 'seg_loss':seg_loss}
