import torch
import argparse
import sys
import yaml
import json
import logging
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, List, Dict
from tqdm import tqdm
from transformers import AutoModel
from torchmetrics.classification import MulticlassConfusionMatrix
import numpy as np

# Import both dataset modules
from fssdino.dataset.cdfss import CDFSSDataset_multiclass
from fssdino.dataset.coco import FSSCOCODataset_multiclass
from fssdino.module import FSSModule
from fssdino.utils import fix_randseed, fg_miou, compute_miou_fb_iou, convert_results_for_json, align_masks_to_features
from fssdino.visualizer import Visualizer

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


def setup_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO),
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        stream=sys.stdout)

def merge_sim_maps(maps: List[Dict[int, torch.Tensor | None]], dim: int = 0) -> Dict[int, torch.Tensor | None]:
    out: Dict[int, torch.Tensor | None] = {}
    if not maps:
        return out
    keys = maps[0].keys()
    for k in keys:
        tensors = [m[k] for m in maps if m.get(k) is not None]
        out[k] = torch.cat(tensors, dim=dim) if tensors else None
    return out

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

def run_evaluation(cfg: dict, work_dir: Path, device: torch.device) -> dict:
    # Extract config parameters
    model_name = cfg["model_name"]
    n_clusters = cfg.get("n_clusters", 20)
    num_episodes = cfg.get("num_episodes", 600)
    img_size = cfg.get("img_size", 512)
    vis_th = cfg.get("vis_th", 0)
    seed = cfg.get("seed", 0)
    
    print(f"Benchmark: {cfg.get('benchmark')}")
    print(f"Config: shot={cfg.get('nshot')}, way={cfg.get('kway')}")

    fix_randseed(seed)
    visualizer = Visualizer(work_dir=str(work_dir))

    # --- Dataset Setup ---
    fss_dset = get_dataset(cfg, device)

    # --- Model Setup ---
    num_classes = len(fss_dset.categories) + 1
    categories = [{"id": i} for i in range(len(fss_dset.categories) + 1)]
    cat_ids = fss_dset.cat_ids
    
    dinov3 = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    dinov3.eval()
    
    fss_module = FSSModule(
        categories=categories,
        cat_ids=cat_ids,
        model=dinov3,
        device=device,
    )

    pred_metrics = MulticlassConfusionMatrix(num_classes=num_classes).to(device)

    # --- Evaluation Loop ---
    loop_range = range(num_episodes)
    for i in tqdm(loop_range, desc="episodes"):
        out_dict = fss_dset[i]
        
        # Prepare Inputs
        ori_hw = out_dict['query_ori_image_hw']
        imgs_to_encode = [out_dict['query_img'], out_dict['support_imgs']]
        all_imgs = torch.cat(imgs_to_encode, dim=0)

        with torch.no_grad():
            outputs = fss_module.model(all_imgs)


        feat = outputs.hidden_states[-1]
        feat = feat[:, 1 + fss_module.model.config.num_register_tokens:, :] # remove register & cls tokens
        q_feat_raw = feat[:1]
        q_mask_raw = out_dict['query_mask']
        sp_feats_raw = feat[1:1 + len(out_dict['support_imgs'])]
        sp_masks_raw = out_dict['support_masks']

        sp_feats, sp_masks = align_masks_to_features(sp_feats_raw, sp_masks_raw)
        q_feat, q_mask = align_masks_to_features(q_feat_raw, q_mask_raw)

        sp_protos = fss_module.cluster_support_image_features_external(
            sp_feats, sp_masks, n_clusters=n_clusters
        )
        sim_maps = fss_module.calculate_sim_maps(sp_protos, q_feat)
        sim_maps = fss_module.add_gram_refinement(sim_maps, sp_feats, sp_masks, q_feat)

        # Final Prediction
        crp, pixel_scores = fss_module.class_region_proposal(sim_maps, image_size=img_size)    
        crp_array = torch.stack([v for _, v in crp.items()]).unsqueeze(0).float()

        # Padded prediction
        pred_lbl_vis = crp_array[0].argmax(0).int()
        gt_lbl_vis = q_mask_raw[0].argmax(0).int()

        # Interpolate to original size for metric calculation
        crp_array = F.interpolate(crp_array, size=ori_hw, mode="nearest-exact")
        query_mask_array = F.interpolate(q_mask_raw.float(), size=ori_hw, mode="nearest-exact")
        pred_lbl = crp_array[0].argmax(0).int()
        gt_lbl = query_mask_array[0].argmax(0).int()
        _ = pred_metrics.forward(pred_lbl, gt_lbl)

        # Visualization
        if vis_th > 0:
            episode_miou = fg_miou(crp_array.squeeze(0), query_mask_array.squeeze(0)).item()
            if episode_miou >= vis_th:

                visualizer.plot_segmentation(
                    query_img=out_dict['query_img'][0],
                    gt_mask=gt_lbl_vis,
                    pred_mask=pred_lbl_vis,
                    support_imgs=out_dict['support_imgs'],
                    support_masks=out_dict['support_masks'],
                    episode_idx=i,
                    miou=episode_miou,
                )
                # visualizer.plot_tsne(
                #     sp_protos=sp_protos,
                #     q_feat=q_feat, 
                #     q_mask=q_mask, 
                #     cat_ids=cat_ids,
                #     episode_idx=i
                # )

    cm_crp = pred_metrics.confmat.detach().cpu().numpy()
    results = {
        "confmat_crp": cm_crp,
        "miou_crp": compute_miou_fb_iou(cm_crp),
    }
    return results

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
    
    # Run Evaluation
    results = run_evaluation(cfg, work_dir, device)

    summary_lines = [
        "=== Results Summary ===",
        f"mIoU (CRP): {results['miou_crp']['mIoU']:.2f}",
        f"FB-IoU (CRP): {results['miou_crp']['FB-IoU']:.2f}",
    ]
    for line in summary_lines:
        print(line)

    # Save results
    results_path = work_dir / "results_summary.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"Results summary saved to: {results_path}")
    
    metrics_path = work_dir / "full_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        # Convert numpy types to native python types for JSON serialization
        json.dump(convert_results_for_json(results), f, indent=4)
    print(f"Full metrics saved to: {metrics_path}")

if __name__ == "__main__":
    main()