import torch
from .utils import kmeans
import torch.nn.functional as F

class FSSModule:
    def __init__(self, categories, cat_ids, model, device):
        super().__init__()
        self.categories = categories
        self.cat_ids = cat_ids
        self.model = model
        self.device = device
        self.model.to(device)
    
    @torch.no_grad()
    def extract_class_features(self, features, class_masks):
        """
        Expects spatially aligned inputs.
        features:    [B, C, H, W]
        class_masks: [B, K, H, W]
        """
        # Flatten spatial dimensions for easy masking: [B, C, H*W]
        B, C, H, W = features.shape
        features_flat = features.flatten(2)       # [B, C, 1024]
        features_flat = features_flat.transpose(1, 2) # [B, 1024, C]
        
        # Flatten masks: [B, K, H*W] -> [B, K, 1024]
        masks_flat = class_masks.flatten(2)

        class_feats = {}

        for ind, cat_id in enumerate(self.cat_ids):
            # Boolean mask for current category: [B, 1024]
            current_mask = masks_flat[:, ind, :] == 1
            
            # Extract valid features
            extracted = features_flat[current_mask]
            
            class_feats[cat_id] = extracted if extracted.shape[0] > 0 else torch.tensor([], device=features.device)

        return class_feats

    @torch.no_grad()
    def cluster_support_image_features_external(self, features, class_masks, n_clusters=10, no_cluster=False):
        # 1. Extract Features (Inputs assumed aligned)
        class_feats = self.extract_class_features(features, class_masks)
        
        # 2. Cluster
        clustered_feats = {cat_id: torch.Tensor([]) for cat_id in self.cat_ids}
        
        for k, class_feat in class_feats.items():
            if class_feat.numel() == 0 or len(class_feat) < n_clusters:
                clustered_feats[k] = class_feat
                continue
            
            cluster_ids_x, cluster_centers = kmeans(
                X=class_feat, 
                num_clusters=n_clusters, 
                distance='cosine', 
                tqdm_flag=False,
                device=self.device
            )
            clustered_feats[k] = cluster_centers.to(self.device)
            
        return clustered_feats

    @torch.no_grad()
    def calculate_class_gram_matrices(self, features, class_masks, normalize=True):
        """
        Computes a C x C Gram matrix for each class based on the provided masks.
        features:    [B, C, H, W]
        class_masks: [B, K, H, W]
        """
        # 1. Reuse your existing extraction logic to get [N, C] per class
        class_feats = self.extract_class_features(features, class_masks)
        
        gram_matrices = {}
        
        for cat_id, feats in class_feats.items():
            # feats shape: [N, C] where N is number of pixels matching the class
            if feats.numel() == 0:
                gram_matrices[cat_id] = None
                continue
                
            # Optional: Normalize features to focus on correlations rather than magnitude
            if normalize:
                feats = F.normalize(feats, p=2, dim=1)
                
            # Gram Matrix calculation: (C x N) @ (N x C) -> (C x C)
            # We transpose feats to get [C, N]
            f = feats.t() 
            gram = torch.mm(f, feats)
            
            # Normalize by the number of pixels (N) to make the signature 
            # independent of the size of the object
            n_pixels = feats.shape[0]
            gram_matrices[cat_id] = gram / n_pixels

        return gram_matrices
    
    def get_gram_similarity_map(self, sp_gram, q_feat):
        """
        sp_gram: [C, C] (e.g., [768, 768])
        q_feat:  [1, C, H, W] (e.g., [1, 768, 32, 32])
        returns: [1, 1, H, W] activation map
        """
        C, H, W = q_feat.shape[1:]
        
        # 1. Project Query Features using Support Gram logic
        # [C, C] @ [C, HW] -> [C, HW]
        q_flat = q_feat.view(C, -1) 
        projected = torch.mm(sp_gram, q_flat) # [768, 1024]
        
        # 2. Compute "Energy" or "Correlation"
        # Dot product of original features with projected features
        # This highlights pixels that follow the support's channel-correlation "style"
        gram_map = torch.sum(q_flat * projected, dim=0) # [1024]
        
        # 3. Reshape and Normalize
        gram_map = gram_map.view(1, 1, H, W)
        
        # Normalize to [0, 1] so it matches the scale of your cosine sim_maps
        gram_map = (gram_map - gram_map.min()) / (gram_map.max() - gram_map.min() + 1e-8)
        
        return gram_map

    def add_gram_refinement(self, sim_maps, sp_feats, sp_masks, q_feat):
        sp_grams = self.calculate_class_gram_matrices(sp_feats, sp_masks)
        for cat_id in self.cat_ids:
            if sp_grams.get(cat_id) is None:
                continue
            g_map = self.get_gram_similarity_map(sp_grams[cat_id], q_feat)
            if cat_id in sim_maps:
                sim_maps[cat_id] = torch.cat([sim_maps[cat_id], g_map.squeeze(0)], dim=0)
        return sim_maps
    

    @torch.no_grad()
    def calculate_sim_maps(self, sp_protos, q_feat):
        """
        Calculates cosine similarity between support prototypes and query features.
        Returns Tensors directly [N_clusters, h, w] for efficiency.
        """
        # 1. Handle Input Shapes
        assert q_feat.shape[0] == 1, "Batch size of q_feat must be 1"
        q_feat = q_feat.squeeze(0) # [C, h, w]
        C, h, w = q_feat.shape

        # 2. Prepare Query Features (Flatten & Normalize)
        # [C, h, w] -> [C, h*w]
        q_feat_flat = q_feat.view(C, -1)
        q_feat_norm = F.normalize(q_feat_flat, p=2, dim=0)

        sims = {}

        for cat_id, prototypes in sp_protos.items():
            # Handle empty prototypes
            if prototypes.numel() == 0:
                sims[cat_id] = None
                continue

            # 3. Vectorized Cosine Similarity
            # Normalize prototypes: [N_clusters, C]
            prototypes_norm = F.normalize(prototypes, p=2, dim=1)
            
            # Matrix Multiply: [N_clusters, C] @ [C, h*w] -> [N_clusters, h*w]
            sim_matrix = prototypes_norm @ q_feat_norm
            
            # 4. Reshape to spatial grid [N_clusters, h, w]
            sims[cat_id] = sim_matrix.view(-1, h, w)

        return sims

    @torch.no_grad()
    def class_region_proposal(self, sim_maps, image_size,zero_bg=False):
        """
        Generates class proposals.
        UPSCALES FIRST (High compute, High accuracy) before aggregating scores.
        """
        device = self.device
        class_score_maps = []
        
        for cat_id in self.cat_ids:
            # sim_tensor shape: [N_clusters, 32, 32]
            sim_tensor = sim_maps.get(cat_id)

            if sim_tensor is None:
                neg_inf = torch.full((image_size, image_size), float('-inf'), device=device)
                class_score_maps.append(neg_inf)
                continue

            # --- Step 1: Upscale ALL clusters to High Res ---
            # We treat N_clusters as the Batch dimension for F.interpolate
            # [N, h, w] -> [N, 1, h, w]
            sim_tensor = sim_tensor.unsqueeze(1)
            
            sim_high_res = F.interpolate(
                sim_tensor, 
                size=(image_size, image_size), 
                mode="bilinear", 
                align_corners=False
            ).squeeze(1) 
            # Result: [N_clusters, 512, 512]

            # --- Step 2: Aggregate at High Resolution ---
            # Now compute mean/max on the detailed 512x512 maps
            mean_map = sim_high_res.mean(dim=0)
            max_map, _ = sim_high_res.max(dim=0)
            
            score_map = max_map * mean_map
            class_score_maps.append(score_map)

        # Stack into [Num_Classes, H_img, W_img]
        class_scores_tensor = torch.stack(class_score_maps)

        # Final Pixel-wise Competition
        pixel_scores, _ = torch.max(class_scores_tensor, dim=0)
        pixel_labels = torch.argmax(class_scores_tensor, dim=0)

        # Generate Binary Masks (CRP)
        crp = {}
        for class_idx, cat_id in enumerate(self.cat_ids):
            if sim_maps.get(cat_id) is None:
                crp[cat_id] = torch.zeros((image_size, image_size), device=device, dtype=torch.int)
            else:
                crp[cat_id] = (pixel_labels == class_idx).int()
        
        if zero_bg and 0 in crp:
            crp[0] = torch.zeros_like(crp[0])

        return crp, pixel_scores
