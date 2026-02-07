import torch
import torch.nn.functional as F
import numpy as np
import yaml
import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModel
from torchmetrics.classification import MulticlassConfusionMatrix
import itertools
from typing import Dict, List, Optional, Any
import sys, logging
# Existing imports from your fssdino package
from fssdino.utils import fix_randseed, fg_miou, align_masks_to_features
from fssdino.module import FSSModule
from fssdino.dataset.cdfss import CDFSSDataset_multiclass
from fssdino.dataset.coco import FSSCOCODataset_multiclass

def setup_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO),
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        stream=sys.stdout)

def fisher_discriminant(
    features,
    cluster_masks,
    mode="cosine",
    eps=1e-6,
    ignore_background=False,
):
    """
    Compute Fisher discriminant score for clustered feature maps.

    the higher the score, the more seperable the clusters are

    Args:
        features (Tensor): [C, H, W]
        cluster_masks (Tensor): [K, H, W], binary masks (k=0 is background)
        mode (str): "cosine" or "euclidean"
        eps (float): numerical stability
        ignore_background (bool): if True, ignore cluster 0

    Returns:
        Tensor: scalar Fisher discriminant score
    """

    assert mode in ["cosine", "euclidean"], \
        "mode must be 'cosine' or 'euclidean'"

    C, H, W = features.shape
    K = cluster_masks.shape[0]

    # [HW, C]
    flat_features = features.view(C, -1).permute(1, 0)

    intra_scatter = flat_features.new_tensor(0.0)
    cluster_means = []
    cluster_sizes = []

    # ---- which clusters to iterate over ----
    start_k = 1 if ignore_background else 0

    # ---- per-cluster statistics ----
    for k in range(start_k, K):
        mask = cluster_masks[k].view(-1).bool()

        if mask.sum() == 0:
            continue

        feats_k = flat_features[mask]      # [Nk, C]
        mean_k = feats_k.mean(dim=0)       # [C]

        if mode == "euclidean":
            intra_scatter += ((feats_k - mean_k) ** 2).sum()

        elif mode == "cosine":
            intra_scatter += (
                1.0 - F.cosine_similarity(
                    feats_k, mean_k.unsqueeze(0), dim=1
                )
            ).sum()

        cluster_means.append(mean_k)
        cluster_sizes.append(mask.sum())

    # ---- need at least 2 valid clusters ----
    if len(cluster_means) < 2:
        return flat_features.new_tensor(0.0)

    cluster_means = torch.stack(cluster_means)          # [K', C]
    cluster_sizes = torch.stack(cluster_sizes).float()  # [K']

    # ---- global (weighted) mean ----
    global_mean = (
        cluster_means * cluster_sizes[:, None]
    ).sum(dim=0) / cluster_sizes.sum()

    # ---- inter-cluster scatter ----
    if mode == "euclidean":
        inter_scatter = (
            cluster_sizes[:, None] *
            (cluster_means - global_mean) ** 2
        ).sum()

    elif mode == "cosine":
        inter_scatter = (
            cluster_sizes *
            (
                1.0 - F.cosine_similarity(
                    cluster_means,
                    global_mean.unsqueeze(0),
                    dim=1,
                )
            )
        ).sum()

    # ---- Fisher score ----
    fisher_score = inter_scatter / (intra_scatter + eps)

    return fisher_score


def get_dataset(cfg, device):
    """
    Factory function to initialize the correct dataset based on the benchmark key.
    """
    benchmark = cfg.get("benchmark", "coco").lower()
    img_size = cfg.get("img_size", 512)
    nshot = cfg.get("nshot", 1)
    kway = cfg.get("kway", 3)

    if benchmark == "coco":
        # COCO Initialization
        print(f"Initializing COCO Dataset for benchmark: {benchmark}")
        dataset = FSSCOCODataset_multiclass(
            instances_path=cfg["instances_path"],
            img_dir=cfg["img_dir"],
            img_size=img_size,
            kway=kway,
            nshot=nshot,
            remove_small_annotations=True, # Kept as per your original script
            device=device,
            val_fold_idx=cfg.get("val_fold_idx", 0),
            n_folds=cfg.get("n_folds", 4),
        )
        return dataset
    else:
        # CDFSS Initialization (DeepGlobe, ISIC, ChestX, etc.)
        print(f"Initializing CDFSS Dataset for benchmark: {benchmark}")
        # 1. Initialize global settings for CDFSS
        CDFSSDataset_multiclass.initialize(img_size=img_size, datapath=cfg["datapath"])
        
        # 2. Build the dataloader to get the dataset object
        dataloader = CDFSSDataset_multiclass.build_dataloader(
            benchmark=benchmark, 
            shot=nshot, 
            way=kway, 
            device=device
        )
        return dataloader.dataset


# --- REVERSE ENGINEERING METRICS ---

def get_entropy(sim_maps_dict):
    """Measures prediction 'sharpness'. Lower is more confident."""
    ents = []
    for k, v in sim_maps_dict.items():
        if v is None: continue
        # [K, H, W] -> Softmax over clusters
        p = F.softmax(v, dim=0)
        e = -torch.sum(p * torch.log(p + 1e-8), dim=0).mean()
        ents.append(e)
    return torch.stack(ents).mean().item() if ents else 1.0

def get_gram_dist(sp_grams, q_grams, cat_ids):
    """Measures style consistency between support and query."""
    dists = []
    for c in cat_ids:
        if c == 0: continue # Skip background
        if sp_grams.get(c) is not None and q_grams.get(c) is not None:
            dists.append(torch.dist(sp_grams[c], q_grams[c], p=2))
    return torch.stack(dists).mean().item() if dists else 1.0

class OracleOptimizer:
    def __init__(self, num_classes):
        self.data = []  
        self.num_classes = num_classes
        self.metric_directions = {
            'fisher': 1, 'gram': -1, 'entropy': -1,
            'rv_miou': 1, 'sp_self_miou': 1, 'reg_ratio': 1
        }

    def collect_episode(self, episode_layer_stats: List[Dict[str, Any]]):
        self.data.append(episode_layer_stats)

    def _compute_miou_standard(self, confusion_matrix: np.ndarray) -> float:
        """
        Calculates mIoU globally following the provided standard:
        1. Calculate IoU per class.
        2. Exclude background (class 0).
        3. Mean of foreground IoUs * 100.
        """
        intersection = np.diag(confusion_matrix).astype(float)
        union = confusion_matrix.sum(0) + confusion_matrix.sum(1) - intersection
        
        iou_per_class = np.zeros_like(intersection, dtype=float)
        valid = union > 0
        iou_per_class[valid] = intersection[valid] / (union[valid] + 1e-8)

        # Exclude background (index 0) for mIoU
        fg_iou = iou_per_class[1:]
        mIoU = float(fg_iou.mean() * 100) if fg_iou.size else 0.0
        return mIoU

    def calculate_oracle(self):
        """Theoretical Peak: Best layer per episode aggregated globally."""
        oracle_cm = np.zeros((self.num_classes, self.num_classes))
        for episode in self.data:
            # Oracle choice based on best individual mIoU
            best_layer = max(episode, key=lambda x: x['target_miou'])
            oracle_cm += best_layer['target_confmat']
        return self._compute_miou_standard(oracle_cm)

    def calculate_last_layer(self):
        """Naive Baseline: Performance of the final backbone block."""
        last_layer_cm = np.zeros((self.num_classes, self.num_classes))
        for episode in self.data:
            last_layer_cm += episode[-1]['target_confmat']
        return self._compute_miou_standard(last_layer_cm)

    def run_grid_search(self, search_space: Dict[str, List[float]], top_n_configs=10):
        metric_names = [k.replace('w_', '') for k in search_space.keys()]
        weight_keys = list(search_space.keys())
        combinations = [dict(zip(weight_keys, v)) for v in itertools.product(*search_space.values())]
        
        results = []
        for config in tqdm(combinations, desc="Global Grid Search"):
            if all(v == 0 for v in config.values()): continue
            
            combined_cm = np.zeros((self.num_classes, self.num_classes))
            for episode in self.data:
                best_layer_score = -float('inf')
                chosen_cm = None
                
                for layer_stats in episode:
                    score = 0
                    for m_name in metric_names:
                        val = layer_stats.get(m_name, 0)
                        weight = config.get(f'w_{m_name}', 0)
                        direction = self.metric_directions.get(m_name, 1)
                        if m_name == 'fisher': val = np.log1p(val)
                        score += (direction * weight * val)
                    
                    if score > best_layer_score:
                        best_layer_score = score
                        chosen_cm = layer_stats['target_confmat']
                combined_cm += chosen_cm
            
            results.append((config, self._compute_miou_standard(combined_cm)))

        return sorted(results, key=lambda x: x[1], reverse=True)[:top_n_configs]

# --- MAIN EVALUATION LOOP ---

def run_oracle_search(cfg, device):
    fix_randseed(cfg.get('seed', 0))
    fss_dset = get_dataset(cfg, device)
    num_classes = len(fss_dset.categories) + 1
    dinov3 = AutoModel.from_pretrained(cfg['model_name'], output_hidden_states=True).to(device).eval()
    fss_module = FSSModule(categories=[{"id": i} for i in range(len(fss_dset.categories) + 1)],
                           cat_ids=fss_dset.cat_ids, model=dinov3, device=device)
    
    optimizer = OracleOptimizer(num_classes)
    num_reg = dinov3.config.num_register_tokens
    layer_cm_metric = MulticlassConfusionMatrix(num_classes=num_classes).to(device)

    for i in tqdm(range(cfg.get('num_episodes', 100)), desc="Collecting Episodes"):
        out_dict = fss_dset[i]
        all_imgs = torch.cat([out_dict['query_img'], out_dict['support_imgs']], dim=0)
        
        with torch.no_grad():
            outputs = dinov3(all_imgs)
        
        episode_layer_stats = []
        for blk_num, full_feat in enumerate(outputs.hidden_states):
            # Register Logic
            reg_tokens = full_feat[:, 1:1+num_reg, :]
            patch_tokens = full_feat[:, 1+num_reg:, :]
            reg_ratio = (reg_tokens.norm(dim=-1).mean() / (patch_tokens.norm(dim=-1).mean() + 1e-8)).item()

            # Alignment
            q_feat, _ = align_masks_to_features(patch_tokens[:1], out_dict['query_mask'])
            sp_feats, sp_masks = align_masks_to_features(patch_tokens[1:], out_dict['support_masks'])
            # Support & Gram Extraction
            sp_protos = fss_module.cluster_support_image_features_external(sp_feats, sp_masks)
            sp_grams = fss_module.calculate_class_gram_matrices(sp_feats, sp_masks)
        

            # 1. Support Self-IoU (sp_self_miou)
            sp_self_ious = []
            for sf, sm_raw, sm in zip(sp_feats, out_dict['support_masks'], sp_masks):
                s_sim = fss_module.calculate_sim_maps(sp_protos, sf.unsqueeze(0))
                s_sim = fss_module.add_gram_refinement(s_sim, sp_feats, sp_masks, sf.unsqueeze(0))

                s_crp, _ = fss_module.class_region_proposal(s_sim, image_size=cfg['img_size'])
                s_pred = torch.stack([v for _, v in s_crp.items()]).float()
                sp_self_ious.append(fg_miou(s_pred, sm_raw).item())
            
            # 2. Query Prediction & Reverse IoU (rv_miou)
            sim_maps = fss_module.calculate_sim_maps(sp_protos, q_feat)
            sim_maps = fss_module.add_gram_refinement(sim_maps, sp_feats, sp_masks, q_feat)

            crp, _ = fss_module.class_region_proposal(sim_maps, image_size=cfg['img_size'])
            q_pred = torch.stack([v for _, v in crp.items()]).float()

            q_pred_down = F.interpolate(q_pred.unsqueeze(0), size=q_feat.shape[-2:], mode="nearest")
            q_protos_from_pred = fss_module.cluster_support_image_features_external(q_feat, q_pred_down)
            
            rv_ious = []
            for sf, sm_raw, sm in zip(sp_feats, out_dict['support_masks'], sp_masks):
                rv_sim = fss_module.calculate_sim_maps(q_protos_from_pred, sf.unsqueeze(0))
                rv_sim = fss_module.add_gram_refinement(rv_sim, q_feat, q_pred_down, sf.unsqueeze(0))
                rv_crp, _ = fss_module.class_region_proposal(rv_sim, image_size=cfg['img_size'])
                rv_pred = torch.stack([v for _, v in rv_crp.items()]).float()
                rv_ious.append(fg_miou(rv_pred, sm_raw).item())

            # 6. Gram & Fisher
            q_grams = fss_module.calculate_class_gram_matrices(q_feat, q_pred_down)
            f_val = fisher_discriminant(q_feat.squeeze(0), q_pred_down.squeeze(0)).item()
            g_dist = get_gram_dist(sp_grams, q_grams, fss_dset.cat_ids)
            
            # 7. Oracle Target
            current_miou = fg_miou(q_pred.squeeze(0), out_dict['query_mask'].squeeze(0)).item()

            layer_cm_metric.reset()
            layer_cm_metric.update(q_pred.argmax(0).int(), out_dict['query_mask'][0].argmax(0).int())
            current_layer_cm = layer_cm_metric.confmat.detach().cpu().numpy()   

            episode_layer_stats.append({
                'fisher': f_val,
                'gram': g_dist,
                'entropy': get_entropy(sim_maps),
                'rv_miou': np.mean(rv_ious),
                'sp_self_miou': np.mean(sp_self_ious),
                'reg_ratio': reg_ratio,
                'target_miou': current_miou, # Kept for 'picking' the best layer
                'target_confmat': current_layer_cm # Needed for Global mIoU
            })
            
        optimizer.collect_episode(episode_layer_stats)

    # --- 1. Benchmarks ---
    oracle_miou = optimizer.calculate_oracle()
    last_layer_miou = optimizer.calculate_last_layer()

    # --- 2. Individual Metric Power ---
    individual_results = {}
    available_metrics = ['fisher', 'gram', 'entropy', 'rv_miou', 'sp_self_miou', 'reg_ratio']
    for m in available_metrics:
        solo_space = {f'w_{name}': [1.0 if name == m else 0.0] for name in available_metrics}
        res = optimizer.run_grid_search(solo_space, top_n_configs=1)
        individual_results[m] = res[0][1]
        
    # --- 3. Grid Search ---
    search_space = {
        'w_fisher': [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
        'w_gram': [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
        'w_rv_miou': [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
        'w_reg_ratio': [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
        'w_sp_self_miou':[0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
        'w_entropy':[0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
    }
    top_configs = optimizer.run_grid_search(search_space)
    
    # --- 4. Final Final Reporting ---
    print("\n" + "!"*50)
    print(f"{'ORACLE PEAK mIoU':<25}: {oracle_miou:.4f}")
    print("!"*50)
    
    print("\nINDIVIDUAL METRIC SELECTION PERFORMANCE:")
    print("-" * 65)
    print(f"{'Metric':<15} | {'mIoU':<8} | {'vs Last Layer':<15} | {'vs Oracle'}")
    print("-" * 65)
    for metric, score in sorted(individual_results.items(), key=lambda x: x[1], reverse=True):
        diff_oracle = score - oracle_miou
        print(f"{metric:<15} | {score:.4f} | {diff_oracle:.4f}")
    print("\n" + "="*50)
    print("TOP WEIGHTED CONFIGURATIONS")
    print("="*50)
    for config, score in top_configs:
        active_weights = {k: v for k, v in config.items() if v > 0}
        print(f"mIoU: {score:.4f} | Weights: {active_weights}")

    return top_configs

def main(argv: Optional[List[str]] = None) -> None:
    setup_logging()

    if argv is None:
        argv = sys.argv[1:]

    if len(argv) != 1:
        # Fallback if argument not provided, handy for VSCode debugging
        print("Usage: python unified_eval.py <config.yaml>")
        return

    config_path = Path(argv[0])
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load YAML config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    work_dir = Path(cfg.get("work_dir", "./results"))
    work_dir.mkdir(parents=True, exist_ok=True)

    # Save the exact config used in the work_dir
    cfg_save_path = work_dir / "config_used.yaml"
    with open(cfg_save_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    print(f"Config saved to: {cfg_save_path}")

    device = torch.device(cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    top_configs = run_oracle_search(cfg, device)


if __name__ == "__main__":
    main()