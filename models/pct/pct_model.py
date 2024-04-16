import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from pointnet2_ops import pointnet2_utils
# from pointnet2.pointnet2_modules import *
# from data.nn_distance import *
from random import shuffle

class Embedding(nn.Module):
    """
    Input Embedding layer which consist of 2 stacked LBR layer.
    """

    def __init__(self, in_channels=3, out_channels=128):
        super(Embedding, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        """
        Input
            x: [B, in_channels, N]
        
        Output
            x: [B, out_channels, N]
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class SA(nn.Module):
    """
    Self Attention module.
    """

    def __init__(self, channels):
        super(SA, self).__init__()

        self.da = channels // 4

        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """
        Input
            x: [B, de, N]
        
        Output
            x: [B, de, N]
        """
        # compute query, key and value matrix
        x_q = self.q_conv(x).permute(0, 2, 1)  # [B, N, da]
        x_k = self.k_conv(x)                   # [B, da, N]        
        x_v = self.v_conv(x)                   # [B, de, N]

        # compute attention map and scale, the sorfmax
        energy = torch.bmm(x_q, x_k) / (math.sqrt(self.da))   # [B, N, N]
        attention = self.softmax(energy)                      # [B, N, N]

        # weighted sum
        x_s = torch.bmm(x_v, attention)  # [B, de, N]
        x_s = self.act(self.after_norm(self.trans_conv(x_s)))
        
        # residual
        x = x + x_s

        return x

class SG(nn.Module):
    """
    SG(sampling and grouping) module.
    """

    def __init__(self, s, in_channels, out_channels):
        super(SG, self).__init__()

        self.s = s

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
    
    def forward(self, x, coords):
        """
        Input:
            x: features with size of [B, in_channels//2, N]
            coords: coordinates data with size of [B, N, 3]
        """
        x = x.permute(0, 2, 1)           # (B, N, in_channels//2)
        new_xyz, new_feature = sample_and_knn_group(s=self.s, k=32, coords=coords, features=x)  # [B, s, 3], [B, s, 32, in_channels]
        b, s, k, d = new_feature.size()
        new_feature = new_feature.permute(0, 1, 3, 2)
        new_feature = new_feature.reshape(-1, d, k)                               # [Bxs, in_channels, 32]
        batch_size = new_feature.size(0)
        new_feature = F.relu(self.bn1(self.conv1(new_feature)))                   # [Bxs, in_channels, 32]
        new_feature = F.relu(self.bn2(self.conv2(new_feature)))                   # [Bxs, in_channels, 32]
        new_feature = F.adaptive_max_pool1d(new_feature, 1).view(batch_size, -1)  # [Bxs, in_channels]
        new_feature = new_feature.reshape(b, s, -1).permute(0, 2, 1)              # [B, in_channels, s]
        return new_xyz, new_feature

class SG_K(nn.Module):
    """
    SG(sampling and grouping) module.
    """

    def __init__(self, s, in_channels, out_channels, k_neighbors = 4):
        super(SG_K, self).__init__()

        self.s = s

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.k_neighbors = k_neighbors
    
    def forward(self, x, coords):
        """
        Input:
            x: features with size of [B, in_channels//2, N]
            coords: coordinates data with size of [B, N, 3]
        """
        x = x.permute(0, 2, 1)           # (B, N, in_channels//2)
        new_xyz, new_feature = knn_group(k=self.k_neighbors, coords=coords, features=x)  # [B, s, 3], [B, s, 32, in_channels]
        b, s, k, d = new_feature.size()
        new_feature = new_feature.permute(0, 1, 3, 2)
        new_feature = new_feature.reshape(-1, d, k)                               # [Bxs, in_channels, 32]
        batch_size = new_feature.size(0)
        new_feature = F.relu(self.bn1(self.conv1(new_feature)))                   # [Bxs, in_channels, 32]
        new_feature = F.relu(self.bn2(self.conv2(new_feature)))                   # [Bxs, in_channels, 32]
        new_feature = F.adaptive_max_pool1d(new_feature, 1).view(batch_size, -1)  # [Bxs, in_channels]
        new_feature = new_feature.reshape(b, s, -1).permute(0, 2, 1)              # [B, in_channels, s]
        return new_xyz, new_feature

class NeighborEmbedding(nn.Module):
    def __init__(self, samples=[512, 256], module = 'SG', out_channels = 256):
        super(NeighborEmbedding, self).__init__()

        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        if module == 'SG':
          self.sg1 = SG(s=samples[0], in_channels=128, out_channels=128)
          self.sg2 = SG(s=samples[1], in_channels=256, out_channels=out_channels)
        elif module == 'SG_K':
          self.sg1 = SG_K(s=samples[0], in_channels=128, out_channels=128)
          self.sg2 = SG_K(s=samples[1], in_channels=256, out_channels=out_channels)

    def forward(self, x):
        """
        Input:
            x: [B, 3, N]
        """
        xyz = x.permute(0, 2, 1)  # [B, N ,3]

        features = F.relu(self.bn1(self.conv1(x)))        # [B, 64, N]
        features = F.relu(self.bn2(self.conv2(features))) # [B, 64, N]

        xyz1, features1 = self.sg1(features, xyz)         # [B, 128, 512]
        xyz2, features2 = self.sg2(features1, xyz1)          # [B, 256, 256]

        return features2, xyz2

class NR_process(nn.Module):
    def __init__(self, dimk = 64, dim_input =3):
        super(NR_process, self).__init__()

        self.conv1 = nn.Conv1d(dim_input, dimk, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(dimk, dimk, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(dimk)
        self.bn2 = nn.BatchNorm1d(dimk)

        #self.sg1 = SG(s=samples[0], in_channels=128, out_channels=128)
        #self.sg2 = SG(s=samples[1], in_channels=256, out_channels=256)
    
    def forward(self, x):
        """
        Input:
            x: [B, 3, N]
        """
        xyz = x.permute(0, 2, 1)  # [B, N ,3]

        features = F.relu(self.bn1(self.conv1(x)))        # [B, 64, N]
        features = F.relu(self.bn2(self.conv2(features))) # [B, 64, N]

        #xyz1, features1 = self.sg1(features, xyz)         # [B, 128, 512]
        #_, features2 = self.sg2(features1, xyz1)          # [B, 256, 256]

        return features

class OA(nn.Module):
    """
    Offset-Attention Module.
    """
    
    def __init__(self, channels):
        super(OA, self).__init__()

        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)

        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)  # change dim to -2 and change the sum(dim=1, keepdims=True) to dim=2

    def forward(self, x):
        """
        Input:
            x: [B, de, N]
        
        Output:
            x: [B, de, N]
        """
        x_q = self.q_conv(x).permute(0, 2, 1)
        x_k = self.k_conv(x)    
        x_v = self.v_conv(x)

        energy = torch.bmm(x_q, x_k)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))  # here

        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r

        return x

def cal_loss(pred, ground_truth, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    ground_truth = ground_truth.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, ground_truth.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, ground_truth, reduction='mean')

    return loss

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]

    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Ball query.

    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    
    Output:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def knn_point(k, xyz, new_xyz):
    """
    K nearest neighborhood.

    Input:
        k: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    
    Output:
        group_idx: grouped points index, [B, S, k]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, k, dim=-1, largest=False, sorted=False)
    return group_idx

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    
    Output:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def sample_and_ball_group(s, radius, n, coords, features):
    """
    Sampling by FPS and grouping by ball query.

    Input:
        s[int]: number of points to be sampled by FPS
        k[int]: number of points to be grouped into a neighbor by ball query
        n[int]: fix number of points in ball neighbor
        coords[tensor]: input points coordinates data with size of [B, N, 3]
        features[tensor]: input points features data with size of [B, N, D]
    
    Returns:
        new_coords[tensor]: sampled and grouped points coordinates by FPS with size of [B, s, k, 3]
        new_features[tensor]: sampled and grouped points features by FPS with size of [B, s, k, 2D]
    """
    batch_size = coords.shape[0]
    coords = coords.contiguous()

    # FPS sampling
    fps_idx = pointnet2_utils.furthest_point_sample(coords, s).long()  # [B, s]
    new_coords = index_points(coords, fps_idx)                         # [B, s, 3]
    new_features = index_points(features, fps_idx)                     # [B, s, D]

    # ball_query grouping
    idx = query_ball_point(radius, n, coords, new_coords)              # [B, s, n]
    grouped_features = index_points(features, idx)                     # [B, s, n, D]
    
    # Matrix sub
    grouped_features_norm = grouped_features - new_features.view(batch_size, s, 1, -1)  # [B, s, n, D]

    # Concat, my be different in many networks
    aggregated_features = torch.cat([grouped_features_norm, new_features.view(batch_size, s, 1, -1).repeat(1, 1, n, 1)], dim=-1)  # [B, s, n, 2D]

    return new_coords, aggregated_features  # [B, s, 3], [B, s, n, 2D]

def knn_group(k, coords, features):
    """
    Sampling by FPS and grouping by KNN.

    Input:
        s[int]: number of points to be sampled by FPS
        k[int]: number of points to be grouped into a neighbor by KNN
        coords[tensor]: input points coordinates data with size of [B, N, 3]
        features[tensor]: input points features data with size of [B, N, D]
    
    Returns:
        new_coords[tensor]: sampled and grouped points coordinates by FPS with size of [B, s, k, 3]
        new_features[tensor]: sampled and grouped points features by FPS with size of [B, s, k, 2D]
    """
    batch_size, N = coords.shape[:2]#[B,N,3]
    
    coords = coords.contiguous()

    # FPS sampling
    #fps_idx = pointnet2_utils.furthest_point_sample(coords, s).long()  # [B, s]
    #new_coords = index_points(coords, fps_idx)                         # [B, s, 3]
    #new_features = index_points(features, fps_idx)                     # [B, s, D]

    # K-nn grouping
    idx = knn_point(k, coords, coords)                                              # [B, s, k]
    grouped_features = index_points(features, idx)                                      # [B, s, k, D]
    
    # Matrix sub
    grouped_features_norm = grouped_features - features.view(batch_size, N, 1, -1)  # [B, s, k, D]

    # Concat
    aggregated_features = torch.cat([grouped_features_norm, features.view(batch_size, N, 1, -1).repeat(1, 1, k, 1)], dim=-1)  # [B, s, k, 2D]

    return coords, aggregated_features  # [B, s, 3], [B, s, k, 2D]

def sample_and_knn_group(s, k, coords, features):
    """
    Sampling by FPS and grouping by KNN.

    Input:
        s[int]: number of points to be sampled by FPS
        k[int]: number of points to be grouped into a neighbor by KNN
        coords[tensor]: input points coordinates data with size of [B, N, 3]
        features[tensor]: input points features data with size of [B, N, D]
    
    Returns:
        new_coords[tensor]: sampled and grouped points coordinates by FPS with size of [B, s, k, 3]
        new_features[tensor]: sampled and grouped points features by FPS with size of [B, s, k, 2D]
    """
    batch_size = coords.shape[0]
    coords = coords.contiguous()

    # FPS sampling
    fps_idx = pointnet2_utils.furthest_point_sample(coords, s).long()  # [B, s]
    new_coords = index_points(coords, fps_idx)                         # [B, s, 3]
    new_features = index_points(features, fps_idx)                     # [B, s, D]

    # K-nn grouping
    idx = knn_point(k, coords, new_coords)                                              # [B, s, k]
    grouped_features = index_points(features, idx)                                      # [B, s, k, D]
    
    # Matrix sub
    grouped_features_norm = grouped_features - new_features.view(batch_size, s, 1, -1)  # [B, s, k, D]

    # Concat
    aggregated_features = torch.cat([grouped_features_norm, new_features.view(batch_size, s, 1, -1).repeat(1, 1, k, 1)], dim=-1)  # [B, s, k, 2D]

    return new_coords, aggregated_features  # [B, s, 3], [B, s, k, 2D]

def link_mat(j_num, bone_link, double_link):
    a = torch.eye(j_num)
    a[bone_link[:,0], bone_link[:,1]] = 0.5
    a[bone_link[:,1], bone_link[:,0]] = 0.5
    a[double_link[:,0], double_link[:,1]] = 0.1
    a[double_link[:,1], double_link[:,0]] = 0.1
    return a

class Logger():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()

class PCT(nn.Module):
    def __init__(self, samples=[512, 512], neighbor = True):
        super().__init__()
        if neighbor:
          self.neighbor_embedding = NeighborEmbedding(samples)
        else:
          self.neighbor_embedding = NR_process()
        #self.nr = NR_process()
        self.OA_dim = 256 if neighbor else 64
        self.oa1 = OA(self.OA_dim)
        self.oa2 = OA(self.OA_dim)
        self.oa3 = OA(self.OA_dim)
        self.oa4 = OA(self.OA_dim)

        self.linear = nn.Sequential(
            nn.Conv1d(self.OA_dim*5, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        x, xyz = self.neighbor_embedding(x)
        x1 = self.oa1(x)
        x2 = self.oa2(x1)
        x3 = self.oa3(x2)
        x4 = self.oa4(x3)

        x = torch.cat([x, x1, x2, x3, x4], dim=1)

        x = self.linear(x)

        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x_max = torch.max(x, dim=-1)[0]
        x_mean = torch.mean(x, dim=-1)

        return x, x_max, x_mean, xyz

class PCT_nr(nn.Module):
    def __init__(self, input_dim = 3, output_dim = 256, neighbor = False):
        super().__init__()
        self.neighbor = neighbor
        self.OA_dim = 64
        if neighbor:
          self.nr = NeighborEmbedding(module = 'SG_K', out_channels = self.OA_dim)
        else:
          self.nr = NR_process(dim_input = input_dim)
        
        #self.neighbor_embedding = NeighborEmbedding(samples)
        #self.nr = NR_process()
        
        self.oa1 = OA(self.OA_dim)
        self.oa2 = OA(self.OA_dim)
        self.oa3 = OA(self.OA_dim)
        self.oa4 = OA(self.OA_dim)

        self.linear = nn.Sequential(
            nn.Conv1d(self.OA_dim*5, output_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        x = self.nr(x)
        if self.neighbor: 
            x = x[0]
        x1 = self.oa1(x)
        x2 = self.oa2(x1)
        x3 = self.oa3(x2)
        x4 = self.oa4(x3)

        x = torch.cat([x, x1, x2, x3, x4], dim=1)

        x = self.linear(x)

        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x_max = torch.max(x, dim=-1)[0]
        x_mean = torch.mean(x, dim=-1)

        return x, x_max, x_mean

class Classification(nn.Module):
    def __init__(self, num_categories=40):
        super().__init__()

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, num_categories)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        self.dp1 = nn.Dropout(p=0.5)
        self.dp2 = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.linear1(x)))
        x = self.dp1(x)
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class Segmentation(nn.Module):
    def __init__(self, input_dim = 256, part_num = 16):
        super().__init__()

        self.part_num = part_num
        '''
        self.label_conv = nn.Sequential(
            nn.Conv1d(16, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        '''
        self.convs1 = nn.Conv1d(input_dim * 3, input_dim * 2, 1)
        self.convs2 = nn.Conv1d(input_dim * 2, input_dim, 1)
        self.convs3 = nn.Conv1d(input_dim, self.part_num, 1)

        self.bns1 = nn.BatchNorm1d(input_dim * 2)
        self.bns2 = nn.BatchNorm1d(input_dim)

        self.dp1 = nn.Dropout(0.5)
    
    def forward(self, x, x_max, x_mean):
        batch_size, _, N = x.size()

        x_max_feature = x_max.unsqueeze(-1).repeat(1, 1, N)
        x_mean_feature = x_mean.unsqueeze(-1).repeat(1, 1, N)

        #cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        #cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)

        x = torch.cat([x, x_max_feature, x_mean_feature], dim=1)  # 1024 * 3 + 64

        x = F.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = F.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)

        return x

class NormalEstimation(nn.Module):
    def __init__(self):
        super().__init__()

        self.convs1 = nn.Conv1d(1024 * 3, 512, 1)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, 3, 1)

        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

        self.dp1 = nn.Dropout(0.5)
    
    def forward(self, x, x_max, x_mean):
        N = x.size(2)

        x_max_feature = x_max.unsqueeze(-1).repeat(1, 1, N)
        x_mean_feature = x_mean.unsqueeze(-1).repeat(1, 1, N)
        
        x = torch.cat([x_max_feature, x_mean_feature, x], dim=1)

        x = F.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = F.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)

        return x

class Regression(nn.Module):
    def __init__(self, input_dim = 256, out_dim = 16):
        super().__init__()

        '''
        self.label_conv = nn.Sequential(
            nn.Conv1d(16, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        '''
        self.convs1 = nn.Conv1d(input_dim * 3, input_dim * 2, 1)
        self.convs2 = nn.Conv1d(input_dim * 2, input_dim, 1)
        self.convs3 = nn.Conv1d(input_dim, out_dim, 1)

        self.bns1 = nn.BatchNorm1d(input_dim * 2)
        self.bns2 = nn.BatchNorm1d(input_dim)

        self.dp1 = nn.Dropout(0.5)
    
    def forward(self, x, x_max, x_mean):
        batch_size, _, N = x.size()

        x_max_feature = x_max.unsqueeze(-1).repeat(1, 1, N)
        x_mean_feature = x_mean.unsqueeze(-1).repeat(1, 1, N)

        #cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        #cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)

        x = torch.cat([x, x_max_feature, x_mean_feature], dim=1)  # 1024 * 3 + 64

        x = F.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        feat = F.relu(self.bns2(self.convs2(x)))
        x = self.convs3(feat)
        # x = x.mean(dim = -1)
        return x, feat

class PCTSeg(nn.Module):
    def __init__(self, samples = [512, 512], segment_num = 15, return_feature = False, neighbor = True):
        super().__init__()
    
        # self.encoder = PCT(samples=samples)
        self.encoder = PCT_nr(neighbor = neighbor)
        self.seg = Segmentation(segment_num)
        self.return_feature = return_feature

    def forward(self, x, cls_label = None):
        # x, x_max, x_mean, _ = self.encoder(x)
        x, x_max, x_mean = self.encoder(x)
        if self.return_feature:
            x_f = self.seg(x, x_max, x_mean)
            # x_f = F.softmax(x_f, dim = 1)
            return x_f, x
        else:
            x = self.seg(x, x_max, x_mean)
            # x = F.softmax(x, dim = 1)
            return x, None

class PCTSeg_Reg(nn.Module):
    # PRN
    def __init__(self, config, return_feature = False):
        super().__init__()
    
        self.encoder = PCT_nr(neighbor = False)
        self.seg = Segmentation(config.segment_num)
        self.reg = Regression(3)
        self.return_feature = return_feature
        self.bone_seg = False

    def forward(self, xyz):
        x, x_max, x_mean = self.encoder(xyz)
        x_s = self.seg(x, x_max, x_mean)
        x_r = self.reg(x, x_max, x_mean)
        batch_size, _, N = x.size()
        x_max_feature = x_max.unsqueeze(-1).repeat(1, 1, N)
        x_mean_feature = x_mean.unsqueeze(-1).repeat(1, 1, N)
        e_x = torch.cat([x, x_max_feature, x_mean_feature], dim=1) 
        return {'seg':x_s, 'reg_p':x_r, 'feature':x, 'xyz':xyz, 'extend_feature': e_x}
    
    def get_loss(self, ret_dict, data, *args):
      disp_dict = {}
      pcd = data['points'].float()
      pose = data['joints'].float()
      reg_label = data['reg_label'].float()
      class_ = ret_dict['seg'].permute(0,2,1)
      reg_pose = self.soft_reg_pose(ret_dict)['reg']
      regress_ = ret_dict['reg_p'].permute(0,2,1)
      label = data['seg_label']
      label, pose = label.to(regress_.device), pose.to(regress_.device)
      cls_loss = get_cls_loss(class_, label)
      r_loss = self.get_reg_loss(regress_, data)
      n_loss = self.get_norm_loss(reg_pose, pose)
      disp_dict.update({'cls': np.round(cls_loss.cpu().detach().numpy(), 3)})
      disp_dict.update({'reg': np.round(r_loss.cpu().detach().numpy(), 3)})
      loss = cls_loss + r_loss + n_loss
      
      return {'cls_loss':cls_loss, 'reg_loss':r_loss, 'n_loss': n_loss}

    def get_norm_loss(self, regress_, pose):
        return torch.mean(torch.norm(regress_ - pose, dim = -1))

    def get_reg_loss(self, output, data):
      class_label = data['seg_label'].to(output.device)
      pose = data['joints'].float().to(output.device)
      pcd = data['points'].float().to(output.device)
      reg_label = torch.zeros_like(output).to(output.device)
      class_label_all = class_label < 15
      
      for b in range(reg_label.shape[0]):
          class_label_this = class_label_all[b]
          reg_label[b,class_label_this,:] = pose[b, class_label[b, class_label_this], :] - pcd[b, class_label_this, :]
      
      loss = reg_label[class_label_all] - output[class_label_all]
      if loss.shape[0] > 0:
        loss = torch.norm(loss, dim = 1)
        loss = torch.mean(loss)
      else:
          loss = torch.tensor(0.0)
      return loss
    
    def get_optim(self):
        dict_ = {}
        g_optim = optim.Adam([{"params":self.encoder.parameters()}, {"params":self.seg.parameters()}, \
                {"params":self.reg.parameters()}], \
                    lr = 1e-3, betas=(0.9,0.999), weight_decay = 5e-6) #{"params":self.pct_model.parameters()},
        dict_.update({'g_optim':g_optim})
        return dict_

    def soft_reg_pose(self, input_):
      if isinstance(input_,dict):
        ret_dict = input_
        xyz = ret_dict['xyz'].permute(0,2,1)
      else:
        xyz = input_.float().cuda()
        ret_dict = self.forward(xyz.permute(0,2,1))
      class_ = ret_dict['seg'].permute(0,2,1)
      class_ = torch.softmax(class_, dim = 2)
      max_class, _ = torch.max(class_, dim = 2)
      class_mask = (class_[:,:,-1] < 0.8) & (max_class > 0.6)
      regress_ = ret_dict['reg_p'].permute(0,2,1)
      reg_pose = regress_ + xyz
      pose_r = torch.zeros([class_.shape[0], 15, 3]).to(class_.device)
      class_sum_all = []
      for b in range(class_.shape[0]):
        class_this = class_[b, class_mask[b]]#[N,15]
        class_sum_this = torch.sum(class_this, dim = 0).unsqueeze(0)
        class_sum_all.append(class_sum_this)
        class_k = class_this / (class_sum_this + 1e-8)
        pose = torch.mm(class_k.permute(1,0), reg_pose[b, class_mask[b], :])
        pose_r[b] = pose[:15,:]
      class_sum_all = torch.stack(class_sum_all, dim = 0) #[B,15]
      r_dict = {}
      r_dict.update({'reg':pose_r, 'seg':class_, 'reg_p':regress_, 'class_sum':class_sum_all})
      return r_dict
    
    def reg_pose(self, input_):
      
      if isinstance(input_,dict):
        ret_dict = input_
        xyz = ret_dict['xyz'].permute(0,2,1)
      else:
        xyz = input_.float().cuda()
        ret_dict = self.forward(xyz.permute(0,2,1))
      class_ = ret_dict['seg'].permute(0,2,1)
      class_ = torch.softmax(class_, dim = 2)
      max_class, _ = torch.max(class_, dim = 2)
      ma_c = torch.argmax(class_, dim = 2)
      class_mask = (class_[:,:,-1] < 0.8) & (max_class > 0.6)
      regress_ = ret_dict['reg'].permute(0,2,1)
      reg_pose = regress_ + xyz
      pose_r = torch.zeros([class_.shape[0], 15, 3]).to(class_.device)
      
      for b in range(class_.shape[0]):
        class_this = class_[b, class_mask[b]]#[N,15]
        class_sum_this = torch.sum(class_this, dim = 0).unsqueeze(0)
        class_k = class_this / (class_sum_this + 1e-8)
        reg_pose_this = reg_pose[b,class_mask[b]]
        pose = torch.mm(class_k.permute(1,0), reg_pose[b, class_mask[b], :])
        pose_r[b] = pose[:15,:]
      
      r_dict = {}
      r_dict.update({'reg':pose_r, 'seg':ma_c})
      return r_dict
