import open3d as o3d
import torch
import numpy as np
import pickle
import torch.distributed as dist
import random
import smplx
smpl_model = smplx.create('./smplx_models/', model_type = 'smpl',
                                    gender='neutral', 
                                    use_face_contour=False,
                                    ext="npz").cuda()
J_regressor = smpl_model.J_regressor
JOINTS_IDX = [0, 1, 2, 4, 5, 7, 8, 12, 15, 16, 17, 18, 19, 20, 21]
def mean_per_edge_error(pred, gt, mesh_face):
    with torch.no_grad():
        if type(mesh_face) != torch.Tensor:
            v1_indx = mesh_face[:,0].astype(int)
            v2_indx = mesh_face[:,1].astype(int)
            v3_indx = mesh_face[:,2].astype(int)
        else:
            v1_indx = mesh_face[:,0]
            v2_indx = mesh_face[:,1]
            v3_indx = mesh_face[:,2]
        pred_v1, pred_v2, pred_v3 = pred[:,v1_indx], pred[:,v2_indx], pred[:,v3_indx]
        gt_v1, gt_v2, gt_v3 = gt[:,v1_indx], gt[:,v2_indx], gt[:,v3_indx]
        pred_l1, pred_l2, pred_l3 = (pred_v1 - pred_v2).norm(dim = -1), (pred_v3 - pred_v2).norm(dim = -1), (pred_v3 - pred_v1).norm(dim = -1)
        gt_l1, gt_l2, gt_l3 = (gt_v1 - gt_v2).norm(dim = -1), (gt_v3 - gt_v2).norm(dim = -1), (gt_v3 - gt_v1).norm(dim = -1)
        diff1 = (pred_l1 - gt_l1).abs()
        diff2 = (pred_l2 - gt_l2).abs()
        diff3 = (pred_l3 - gt_l3).abs()
        mpee = ((diff1 + diff2 + diff3) / 3).cpu().mean()
        mpere = ((diff1 / gt_l1 + diff2 / gt_l2 + diff3 / gt_l3) / 3).cpu()#.mean()
        return mpee, mpere

def mean_per_vertex_error(pred, gt, has_smpl = None):
    """
    Compute mPVE
    """
    if has_smpl is not None:
        pred = pred[has_smpl == 1]
        gt = gt[has_smpl == 1]

    with torch.no_grad():
        error = torch.sqrt( ((pred - gt) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        return error

def mean_per_joint_position_error(pred, gt, has_3d_joints):
    """ 
    Compute mPJPE
    """
    gt = gt[has_3d_joints == 1]
    # gt = gt[:, :, :-1]
    pred = pred[has_3d_joints == 1]

    with torch.no_grad():
        gt_pelvis = (gt[:, 2,:] + gt[:, 3,:]) / 2
        gt = gt - gt_pelvis[:, None, :]
        pred_pelvis = (pred[:, 2,:] + pred[:, 3,:]) / 2
        pred = pred - pred_pelvis[:, None, :]
        # import pdb; pdb.set_trace()
        error = torch.sqrt( ((pred - gt) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        return error

def compute_similarity_transform(S1, S2):
    """Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat

def compute_similarity_transform_batch(S1, S2):
    """Batched version of compute_similarity_transform."""
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat

def reconstruction_error(S1, S2, reduction='mean'):
    """Do Procrustes alignment and compute reconstruction error."""
    S1_hat = compute_similarity_transform_batch(S1, S2)
    re = np.sqrt( ((S1_hat - S2)** 2).sum(axis=-1)).mean(axis=-1)
    if reduction == 'mean':
        re = re.mean()
    elif reduction == 'sum':
        re = re.sum()
    return re

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_mesh(vertices, faces):
    o_v = o3d.utility.Vector3dVector(vertices)
    o_f = o3d.utility.Vector3iVector(faces)
    mesh = o3d.geometry.TriangleMesh(o_v, o_f)
    mesh.compute_vertex_normals()
    return mesh

def compute_error_accel(joints_gt, joints_pred, fps = 20):
    """
    Computes acceleration error:
        1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    Note that for each frame that is not visible, three entries in the
    acceleration error should be zero'd out.
    Args:
        joints_gt (Nx14x3).
        joints_pred (Nx14x3).
        vis (N).
    Returns:
        error_accel (N-2).
    """
    # (N-2)x14x3
    if len(joints_gt.shape) == 4: #[B,T,N,3]
        accel_gt = joints_gt[:,:-2] - 2 * joints_gt[:,1:-1] + joints_gt[:,2:]
        accel_pred = joints_pred[:,:-2] - 2 * joints_pred[:,1:-1] + joints_pred[:,2:]
    else: #[T,N,3]
        accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
        accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]
    normed = (accel_pred - accel_gt).norm(dim = -1)
    return normed.mean() * fps * fps

