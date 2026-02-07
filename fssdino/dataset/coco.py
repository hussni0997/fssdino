import os

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import numpy as np
import cv2
import pycocotools.mask as mask_util
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
import random
from tqdm import tqdm

import concurrent.futures
from torchvision.transforms.functional import to_tensor


class FSSCOCODataset_multiclass(Dataset):
    def __init__(
        self,
        instances_path,
        img_dir,
        img_size=1024,
        kway=1,
        nshot=1,
        remove_small_annotations=True,
        device=None,
        val_fold_idx=None,
        n_folds=None,
    ):
        
        self.instances_path = instances_path
        with open(self.instances_path, "r") as f:
            instances = json.load(f)
        self.annotations = {x["id"]: x for x in instances["annotations"]}
        self.categories = {x["id"]: x for x in instances["categories"]} # have no bg cat
        
        self.cat_id_order = [v['id'] for k,v in self.categories.items()]
        
        self.remove_small_annotations = remove_small_annotations
        # useful dicts
        self._load_annotation_dicts()
        
        # load image ids and info
        img2cat_keys = set(self.img2cat.keys())
        self.images = {x["id"]: x for x in instances["images"] 
                       if x["id"] in img2cat_keys}
        self.image_ids = list(self.images.keys())
        self.img_dir = img_dir
        self.nshot = nshot
        self.kway=kway
        self.cat_ids = [i for i in range(len(self.categories.keys())+1)]
        self.num_categories = len(self.categories.keys())
        self.img_size=img_size

        self.transform = A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.Normalize(),
            A.PadIfNeeded(img_size,
                          img_size,
                          position='top_left',
                          border_mode=cv2.BORDER_CONSTANT,
                          value=0),
            ToTensorV2(transpose_mask=True),],
        )

        
        self.transform_img = A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(img_size,
                          img_size,
                          position='top_left',
                          border_mode=cv2.BORDER_CONSTANT,
                          value=0),
        ])
        
        self.device = torch.device('cpu') if device == None else device
        self.val_fold_idx=val_fold_idx
        self.n_folds=n_folds
        if val_fold_idx is not None:
            self.prepare_benchmark()
        
    def prepare_benchmark(self):
        n_categories = len(self.categories)
        idxs_val = [
            self.val_fold_idx + i * self.n_folds
            for i in range(n_categories // self.n_folds)
        ]
        idxs_train = [i for i in range(n_categories) if i not in idxs_val]
        
        idxs = idxs_val
        
        self.categories = {k: v for i, (k, v) in enumerate(self.categories.items()) if i in idxs}
        self._load_annotation_dicts()

        with open(self.instances_path, "r") as f:
            instances = json.load(f)
        img2cat_keys = set(self.img2cat.keys())
        self.images = {x["id"]: x for x in instances["images"] 
                       if x["id"] in img2cat_keys}
        self.image_ids = list(self.images.keys())
        self.cat_ids = [i for i in range(len(self.categories.keys())+1)]
        self.num_categories = len(self.categories.keys())
        self.cat_id_order = [v['id'] for k,v in self.categories.items()]
    
    def transform_image(self,image_file):
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed=self.transform_img(image=image)
        return transformed['image']
    
    def revert_transform_numpy(self, np_array, ori_hw,):
        '''resize and remove pad'''
        np_array = cv2.resize(np_array, (max(ori_hw), max(ori_hw)))
        np_array = np_array[:ori_hw[0],:ori_hw[1]]
        return np_array

    
    def _load_annotation_dicts(self) -> tuple[dict, dict, dict, dict, dict]:
        """Load useful annotation dicts.

        Returns:
            (dict, dict, dict, dict, dict): Returns four dictionaries:
                0. img_annotations: A dictionary mapping image ids to lists of annotations.
                1. img2cat: A dictionary mapping image ids to sets of category ids.
                2. img2cat_annotations: A dictionary mapping image ids to dictionaries 
                   mapping category ids to annotations.
                3. cat2img: A dictionary mapping category ids to sets of image ids.
                4. cat2img_annotations: A dictionary mapping category ids to dictionaries 
                   mapping image ids to annotations.
        """
        img_annotations = {}
        img2cat_annotations = {}
        cat2img_annotations = {}
        img2cat = {}
        cat2img = {}

        category_ids = set(self.categories.keys())

        for ann in self.annotations.values():
            # if self._remove_small_annotations(ann):
            if self.remove_small_annotations:
                if ann["area"] < 2 * 32 * 32:
                    continue
            
            if "iscrowd" in ann and ann["iscrowd"] == 1:
                continue

            if ann["category_id"] not in category_ids:
                continue

            if ann["image_id"] not in img_annotations:
                img_annotations[ann["image_id"]] = []
            img_annotations[ann["image_id"]].append(ann)

            if ann["image_id"] not in img2cat_annotations:
                img2cat_annotations[ann["image_id"]] = {}
                img2cat[ann["image_id"]] = set()

            if (ann["category_id"] not in img2cat_annotations[ann["image_id"]]):
                img2cat_annotations[ann["image_id"]][ann["category_id"]] = []
                img2cat[ann["image_id"]].add(ann["category_id"])

            img2cat_annotations[ann["image_id"]][ann["category_id"]].append(ann)

            if ann["category_id"] not in cat2img_annotations:
                cat2img_annotations[ann["category_id"]] = {}
                cat2img[ann["category_id"]] = set()

            if ann["image_id"] not in cat2img_annotations[ann["category_id"]]:
                cat2img_annotations[ann["category_id"]][ann["image_id"]] = []
                cat2img[ann["category_id"]].add(ann["image_id"])

            cat2img_annotations[ann["category_id"]][ann["image_id"]].append(ann)
            
        self.img_annotations = img_annotations
        self.img2cat = img2cat
        self.img2cat_annotations = img2cat_annotations
        self.cat2img = cat2img
        self.cat2img_annotations = cat2img_annotations
    
    def ann_to_rle(self, ann, h, w):
        segm = ann
        if isinstance(segm, list):
            rles = mask_util.frPyObjects(segm, h, w)
            rle = mask_util.merge(rles)
        elif isinstance(segm["counts"], list):
            rle = mask_util.frPyObjects(segm, h, w)
        else:
            rle = ann
        return rle
            
    def convert_mask(self, mask, h, w):
        rle = self.ann_to_rle(mask, h, w)
        matrix = mask_util.decode(rle)
        # if matrix is made by all zeros
        if np.all(matrix == 0):
            if isinstance(mask, list):
                first_polygon = mask[0]
                fp_x, fp_y = int(first_polygon[0]), int(first_polygon[1])
                # check if fp_x and fp_y are within the image
                fp_x = min(fp_x, w - 1)
                fp_y = min(fp_y, h - 1)
                # check if fp_x and fp_y are negative
                fp_x = max(fp_x, 0)
                fp_y = max(fp_y, 0) 
                matrix[fp_y, fp_x] = 1
            else:
                matrix[0, 0] = 1
        return matrix
    
    def get_images(self, image_ids, cat_ids):
        """Optimized image and mask loading with parallel processing."""

        transform_fn = self.transform
        
        def load_single_image(img_id):
            image_data = self.images[img_id]
            image_file = f"{self.img_dir}/{image_data['file_name']}"
            image = cv2.imread(image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            h, w = image_data["height"], image_data["width"]
            ground_truth = np.zeros((self.num_categories+1, h, w), dtype=np.uint8)

            for ann in self.img_annotations.get(img_id, []):
                if ann["category_id"] in cat_ids:
                    cat_idx = self.cat_id_order.index(ann["category_id"]) + 1 # +1 to account for bg
                    ann_mask = self.convert_mask(ann["segmentation"], h, w)
                    ground_truth[cat_idx][ann_mask == 1] += 1
            
            ground_truth = np.where(ground_truth>1,1,ground_truth)
            zero_mask = np.where(np.sum(ground_truth,0)==0,1,0)
            ground_truth[0]=zero_mask

            transformed = transform_fn(image=image, mask=ground_truth.transpose(1, 2, 0))
            return transformed["image"], transformed["mask"], image_file, (h, w)

        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(load_single_image, image_ids))

        images, masks, image_files, image_sizes = zip(*results)
        return torch.stack(images), torch.stack(masks), image_files, image_sizes
    
    def __getitem__(self,idx):
        if self.kway == 1:
            cat_ids = [-1, random.choice(list(self.categories.keys()))]
            image_ids = random.sample(
                list(self.cat2img[cat_ids[1]]), self.nshot + 1
            )
        else:
            # sample kway categories
            cat_ids = random.sample(list(self.categories.keys()), self.kway)
            # Choose a random image from the first category
            query_image_id = random.choice(list(self.cat2img[cat_ids[0]]))
            # sample nshot images from each category
            image_ids = [query_image_id]
            for cat_id in cat_ids:
                cat_image_ids = list(self.cat2img[cat_id])
                cat_image_ids = random.sample(cat_image_ids, self.nshot)
                image_ids += cat_image_ids
            cat_ids = [-1] + sorted(cat_ids)
        
        images , ground_truths, image_files, image_ori_sizes = self.get_images(image_ids, cat_ids)
        
        query_image = images[:1].type(torch.float).to(self.device).contiguous()
        query_mask = ground_truths[:1].type(torch.float).to(self.device).contiguous()
        query_file = image_files[:1][0]
        query_ori_image_hw = image_ori_sizes[:1][0]
        support_images = images[1:].type(torch.float).to(self.device).contiguous()
        support_masks = ground_truths[1:].type(torch.float).to(self.device).contiguous()
        support_files = image_files[1:]
        
        out_dict ={
            'query_img':query_image, # torch.Size([1, 3, 1024, 1024])
            'query_mask':query_mask, # torch.Size([1, 1024, 1024])            
            'query_name': query_file, # str
            'query_ori_image_hw':query_ori_image_hw,
            'support_imgs':support_images, # torch.Size([N, 3, 1024, 1024])
            'support_masks':support_masks, # torch.Size([N, 1024, 1024])
            # 'support_files':support_files,
            'support_names':support_files,
            
        }
        
        return out_dict
