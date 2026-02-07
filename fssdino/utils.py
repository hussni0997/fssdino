import numpy as np 
import torch
import random
from functools import partial
import torch
import torch.nn.functional as F

def fix_randseed(seed):
    r""" Set random seeds for reproducibility """
    if seed is None:
        seed = int(random.random() * 1e5)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)

def align_masks_to_features(features, masks):
    """
    Aligns high-res masks to match flat DINO features spatially.
    
    Args:
        features: [B, N, C] (e.g., [4, 1024, 768])
        masks:    [B, K, H_img, W_img] (e.g., [4, 20, 512, 512])
        
    Returns:
        feats_spatial: [B, C, h, w] (e.g., [4, 768, 32, 32])
        masks_spatial: [B, K, h, w] (e.g., [4, 20, 32, 32])
    """
    B, N, C = features.shape
    # Calculate spatial dim (sqrt(1024) -> 32)
    h = w = int(N ** 0.5)
    
    # 1. Reshape features to spatial grid: [B, C, h, w]
    feats_spatial = features.permute(0, 2, 1).view(B, C, h, w)
    
    # 2. Downsample masks to match features: [B, K, h, w]
    masks_spatial = F.interpolate(
        masks.float(), 
        size=(h, w), 
        mode="nearest" # Critical for class IDs/binary masks
    )
    
    return feats_spatial, masks_spatial

def initialize(X, num_clusters, seed):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param seed: (int) seed for kmeans
    :return: (np.array) initial state
    """
    num_samples = len(X)
    if seed == None:
        indices = np.random.choice(num_samples, num_clusters, replace=False)
    else:
        np.random.seed(seed) ; indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = X[indices]
    return initial_state


def kmeans(
        X,
        num_clusters,
        distance='euclidean',
        cluster_centers=[],
        tol=1e-4,
        tqdm_flag=False,
        iter_limit=0,
        device=torch.device('cpu'),
        gamma_for_soft_dtw=0.001,
        seed=None,
):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param seed: (int) seed for kmeans
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :param tqdm_flag: Allows to turn logs on and off
    :param iter_limit: hard limit for max number of iterations
    :param gamma_for_soft_dtw: approaches to (hard) DTW as gamma -> 0
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    if tqdm_flag:
        print(f'running k-means on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = partial(pairwise_distance, device=device, tqdm_flag=tqdm_flag)
    elif distance == 'cosine':
        pairwise_distance_function = partial(pairwise_cosine, device=device)
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    # initialize
    if type(cluster_centers) == list:  # ToDo: make this less annoyingly weird
        initial_state = initialize(X, num_clusters, seed=seed)
    else:
        if tqdm_flag:
            print('resuming')
        # find data point closest to the initial cluster center
        initial_state = cluster_centers
        dis = pairwise_distance_function(X, initial_state)
        choice_points = torch.argmin(dis, dim=0)
        initial_state = X[choice_points]
        initial_state = initial_state.to(device)
    
    ## early exit
    if num_clusters == 1:
        center = X.mean(dim=0, keepdim=True)
        labels = torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
        return labels.cpu(), center.cpu()

    iteration = 0
    if tqdm_flag:
        tqdm_meter = tqdm(desc='[running kmeans]')
    while True:
        dis = pairwise_distance_function(X, initial_state)

        choice_cluster = torch.argmin(dis, dim=1)

        initial_state_pre = initial_state.clone()

        for index in range(num_clusters):
            # selected = torch.nonzero(choice_cluster == index).squeeze().to(device)
            selected = torch.nonzero(choice_cluster == index, as_tuple=True)[0].to(device)


            selected = torch.index_select(X, 0, selected)

            # https://github.com/subhadarship/kmeans_pytorch/issues/16
            if selected.shape[0] == 0:
                selected = X[torch.randint(len(X), (1,))]

            initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        # increment iteration
        iteration = iteration + 1

        # update tqdm meter
        if tqdm_flag:
            tqdm_meter.set_postfix(
                iteration=f'{iteration}',
                center_shift=f'{center_shift ** 2:0.6f}',
                tol=f'{tol:0.6f}'
            )
            tqdm_meter.update()
        if center_shift ** 2 < tol:
            break
        if iter_limit != 0 and iteration >= iter_limit:
            break

    return choice_cluster.cpu(), initial_state.cpu()

def pairwise_distance(data1, data2, device=torch.device('cpu'), tqdm_flag=True):
    if tqdm_flag:
        print(f'device is :{device}')
    
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    # dis = dis.sum(dim=-1).squeeze()
    dis = dis.sum(dim=-1)
    return dis


def pairwise_cosine(data1, data2, device=torch.device('cpu')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    # cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    cosine_dis = 1 - cosine.sum(dim=-1)
    return cosine_dis


def fg_miou(pred, target, eps=1e-6):
    """
    pred, target: (C, H, W) binary or boolean tensors
                  channel 0 = background (ignored)

    Returns:
        fg_miou: scalar (mean IoU over valid foreground classes)
    """
    pred = pred.bool()
    target = target.bool()

    # Per-class intersection & union
    intersection = (pred & target).sum(dim=(1, 2)).float()
    union = (pred | target).sum(dim=(1, 2)).float()

    # Valid classes: union > 0 (same logic as confusion matrix)
    valid = union > 0

    # Ignore background (index 0)
    valid[0] = False

    # If no valid foreground classes
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device)

    iou_per_class = intersection / (union + eps)

    # Foreground mIoU over valid classes
    fg_miou = iou_per_class[valid].mean()

    return fg_miou


def compute_miou_fb_iou(confusion_matrix: np.ndarray) -> dict[str, object]:
    intersection = np.diag(confusion_matrix).astype(float)
    union = confusion_matrix.sum(0) + confusion_matrix.sum(1) - intersection
    iou_per_class = np.zeros_like(intersection, dtype=float)
    valid = union > 0
    iou_per_class[valid] = intersection[valid] / union[valid]

    fg_iou = iou_per_class[1:]
    mIoU = float(fg_iou.mean() * 100) if fg_iou.size else 0.0
    bg_iou = float(iou_per_class[0]) if iou_per_class.size else 0.0
    FB_IoU = float(((fg_iou.mean() if fg_iou.size else 0.0) + bg_iou) * 50)

    return {"per_class_iou": iou_per_class, "mIoU": mIoU, "FB-IoU": FB_IoU}


def convert_results_for_json(results: dict) -> dict:
    """Recursively convert numpy arrays in results to Python lists."""
    new_results = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            new_results[k] = v.tolist()
        elif isinstance(v, dict):
            new_results[k] = convert_results_for_json(v)
        else:
            new_results[k] = v
    return new_results