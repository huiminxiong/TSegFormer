import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import sklearn.metrics as metrics


def cal_loss(pred, gold, smoothing=True):
    """
    Calculate cross entropy loss, apply label smoothing if needed.
    pred: b*n, 33; gold: B*N,
    """
    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


class Focal_Loss(nn.Module):
    def __init__(self, gamma=2, size_average=True):
        super(Focal_Loss, self).__init__()
        self.size_average = size_average
        self.gamma = gamma

    def forward(self, preds, labels):
        preds = preds.view(-1, preds.size(-1))
        preds_logsoft = F.log_softmax(preds, dim=1)  # log_softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()

        return loss


def mask_cur_loss(pred, gold, AvgAngle_idx):
    part_preds = pred[AvgAngle_idx]
    part_labels = gold[AvgAngle_idx]
    focal_loss = Focal_Loss()
    loss2 = focal_loss(part_preds, part_labels)

    return loss2


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


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


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, 3]
        idx: sample index data, [B, S0, S1, ..., Sm-1]
    Return:
        new_points:, indexed points data, [B, S0, S1, ..., Sm-1, 3]
    """
    device = points.device
    B = points.shape[0]

    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)  # [B, 1, 1, 1, 1, ...]

    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1  # [1, S0, S1, ..., Sm-1]

    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(
        repeat_shape)  # same shape with idx, to use advance indexing
    new_points = points[batch_indices, idx, :]
    return new_points  # advance indexing will create a copy instead of a view


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, N, 3] (For segmentation, == xyz)
    Return:
        grosup_idx: grouped points index, [B, N, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)  # [B, N, N]
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx  # [B, N, k]


def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Input:
        npoint: number of output points
        radius: 
        nsample: k
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, N, k, 3]
        new_points: sampled points data, [B, N, k, 2 * D]
    """
    B, N, C = xyz.shape

    xyz = xyz.contiguous()

    #######################################
    # For segmentation, do no change the number or order of the points
    assert (npoint == N)
    new_xyz = xyz
    new_points = points
    #######################################

    idx = knn_point(nsample, xyz, new_xyz)  # [B, N, k]
    grouped_xyz = index_points(xyz, idx)  # [B, N, k, 3]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, N, 1, C)
    grouped_points = index_points(points, idx)  # [B, N, k, D]
    grouped_points_norm = grouped_points - new_points.view(B, N, 1, -1)
    new_points = torch.cat([grouped_points_norm, new_points.view(B, N, 1, -1).repeat(1, 1, nsample, 1)],
                           dim=-1)  # [B, N, k, 2D]
    return new_xyz, new_points


def calculate_shape_IoU(pred_np, seg_np, label):
    """
    Args:
        pred_np: [B, N], predicted segmentation for each point in a Batch
        seg_np: [B, N], true segmentation label for each point in a Batch
        label: [B], label(0 or 1) for each point cloud

    Returns:
        shape IoU of each point cloud
    """

    shape_ious = []
    for shape_idx in range(seg_np.shape[0]):
        if label[shape_idx] == 0:
            parts = range(17)
        else:
            parts = [0]
            parts.extend(list(range(17, 33, 1)))
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            if U == 0:
                iou = 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
    return shape_ious


def calculate_metrics(pred_np, seg_np):
    b = pred_np.shape[0]
    all_f1 = []
    all_sensitivity = []
    all_specificity = []
    all_ppv = []
    all_npv = []
    for i in range(b):
        f1 = metrics.f1_score(seg_np[i], pred_np[i], average='macro')
        confusion_matrix = metrics.confusion_matrix(seg_np[i], pred_np[i])  # n_class * n_class(<=17)
        FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  # n_class，
        FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)  # n_class，
        TP = np.diag(confusion_matrix)  # n_class，
        TN = confusion_matrix.sum() - (FP + FN + TP)  # n_class，

        TPR = []
        PPV = []
        for j in range(len(TP)):
            if (TP[j] + FN[j]) == 0:
                TPR.append(1)
            else:
                TPR.append(TP[j] / (TP[j] + FN[j]))
        for j in range(len(TP)):
            if (TP[j] + FP[j]) == 0:
                PPV.append(1)
            else:
                PPV.append(TP[j] / (TP[j] + FP[j]))
        TNR = TN / (TN + FP)
        NPV = TN / (TN + FN)

        all_f1.append(f1)
        all_ppv.append(np.mean(PPV))
        all_npv.append(np.mean(NPV))
        all_sensitivity.append(np.mean(TPR))
        all_specificity.append(np.mean(TNR))
    return all_f1, all_ppv, all_npv, all_sensitivity, all_specificity
