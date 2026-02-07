import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from PIL import Image
import cv2

class Visualizer:
    def __init__(self, work_dir):
        self.work_dir = work_dir
        self.cmap = plt.get_cmap('tab10')

    def _tensor_to_image(self, img_tensor):
        """
        Convert tensor [3,H,W] to numpy image [H,W,3].
        Assumes normalized image.
        """
        img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()

        # simple min-max normalization for display
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        return img

    def _mask_to_numpy(self, m):
        if isinstance(m, torch.Tensor):
            if m.ndim == 3:
                m = m.argmax(0)
            m = m.cpu().numpy()
        return np.ma.masked_where(m == 0, m)

    def plot_segmentation(
        self,
        query_img,
        gt_mask,
        pred_mask,
        support_imgs,
        support_masks,
        episode_idx,
        miou,
    ):
        # --- convert query ---
        query_img_np = self._tensor_to_image(query_img)

        gt_masked = self._mask_to_numpy(gt_mask)
        pred_masked = self._mask_to_numpy(pred_mask)

        # --- support images ---
        sup_imgs_np = [
            self._tensor_to_image(img) for img in support_imgs
        ]

        num_support = len(sup_imgs_np)
        num_columns = max(num_support, 3)
        fig = plt.figure(figsize=(num_columns * 4, 8))
        gs = fig.add_gridspec(2, num_columns)

        # --- supports ---
        for i, (s_img, s_mask_raw) in enumerate(zip(sup_imgs_np, support_masks)):
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(s_img)

            s_masked = self._mask_to_numpy(s_mask_raw)
            ax.imshow(s_masked, alpha=0.5, cmap='tab10', vmin=0, vmax=10)
            ax.set_title(f"Support {i+1}")
            ax.axis('off')

        # --- query image ---
        ax_q = fig.add_subplot(gs[1, 0])
        ax_q.imshow(query_img_np)
        ax_q.set_title("Query Image")
        ax_q.axis('off')

        # --- GT ---
        ax_gt = fig.add_subplot(gs[1, 1])
        ax_gt.imshow(query_img_np)
        ax_gt.imshow(gt_masked, alpha=0.6, cmap='tab10', vmin=0, vmax=10)
        ax_gt.set_title("Ground Truth")
        ax_gt.axis('off')

        # --- prediction ---
        ax_pred = fig.add_subplot(gs[1, 2])
        ax_pred.imshow(query_img_np)
        ax_pred.imshow(pred_masked, alpha=0.6, cmap='tab10', vmin=0, vmax=10)
        ax_pred.set_title(f"Pred (mIoU: {miou:.2f})")
        ax_pred.axis('off')

        plt.tight_layout()
        save_path = f"{self.work_dir}/vis_ep{episode_idx}_miou_{miou:.2f}.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        return save_path

    def plot_tsne(self, sp_protos, q_feat, q_mask, cat_ids, episode_idx):
        """
        Visualizes t-SNE of Support Prototypes vs Query Features (Class-wise).
        
        Args:
            sp_protos: Dict {cat_id: [N_clusters, C]}
            q_feat: Tensor [1, C, 32, 32] (Low res features)
            q_mask: Tensor [1, K, 32, 32] (Low res masks)
            cat_ids: List of category IDs corresponding to q_mask channels
        """
        features_list = []
        labels_list = []
        markers_list = [] # 0 for Query, 1 for Support

        # Pre-process Query Features: [1, C, H, W] -> [C, H*W] -> [H*W, C]
        B, C, H, W = q_feat.shape
        q_feat_flat = q_feat.squeeze(0).view(C, -1).permute(1, 0) # [1024, C]
        q_mask_flat = q_mask.squeeze(0).view(len(cat_ids), -1)    # [K, 1024]

        # --- Data Collection ---
        valid_classes = []
        
        for idx, cat_id in enumerate(cat_ids):
            # 1. Collect Support Prototypes (Stars)
            protos = sp_protos.get(cat_id)
            if protos is not None and protos.numel() > 0:
                features_list.append(protos.cpu().numpy())
                labels_list.extend([idx] * protos.shape[0])
                markers_list.extend(['support'] * protos.shape[0])
                if idx not in valid_classes: valid_classes.append(idx)

            # 2. Collect Query Features (Dots)
            # Use mask to select only pixels belonging to this class
            mask_bool = q_mask_flat[idx] == 1
            if mask_bool.sum() > 0:
                class_q_feats = q_feat_flat[mask_bool]
                
                # Downsample if too many pixels (e.g., max 100 per class) to keep plot readable
                if class_q_feats.shape[0] > 100:
                    indices = torch.randperm(class_q_feats.shape[0])[:100]
                    class_q_feats = class_q_feats[indices]
                
                features_list.append(class_q_feats.cpu().numpy())
                labels_list.extend([idx] * class_q_feats.shape[0])
                markers_list.extend(['query'] * class_q_feats.shape[0])
                if idx not in valid_classes: valid_classes.append(idx)

        if not features_list: return # Nothing to plot

        X = np.concatenate(features_list, axis=0)
        
        # --- t-SNE Execution ---
        # Perplexity must be < number of samples. Default 30, but dynamic is safer.
        n_samples = X.shape[0]
        perp = min(30, n_samples - 1) if n_samples > 1 else 1
        
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
        X_emb = tsne.fit_transform(X)

        # --- Plotting ---
        plt.figure(figsize=(10, 10))
        
        # Draw scatter points
        for i in range(len(labels_list)):
            lbl = labels_list[i]
            marker_type = markers_list[i]
            
            # Style logic
            marker = '*' if marker_type == 'support' else 'o'
            size = 200 if marker_type == 'support' else 30
            alpha = 0.9 if marker_type == 'support' else 0.4
            edgecolor = 'black' if marker_type == 'support' else None
            
            plt.scatter(
                X_emb[i, 0], X_emb[i, 1], 
                c=self.colors[lbl % len(self.colors)], 
                marker=marker, 
                s=size, 
                alpha=alpha,
                edgecolors=edgecolor,
                label=f"Class {cat_ids[lbl]} ({marker_type})" if f"Class {cat_ids[lbl]} ({marker_type})" not in plt.gca().get_legend_handles_labels()[1] else ""
            )

        plt.title(f"t-SNE: Support Protos (*) vs Query Pixels (o) - Ep {episode_idx}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        save_path = f"{self.work_dir}/vis_ep{episode_idx}_tsne.png"
        plt.savefig(save_path)
        plt.close()
