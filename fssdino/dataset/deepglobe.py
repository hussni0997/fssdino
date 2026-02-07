# taken from https://github.com/Vision-Kek/ABCDFSS/tree/main/data
r""" FSS-1000 few-shot semantic segmentation dataset """
import os
import glob

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
import random
import cv2
from functools import lru_cache

class DatasetDeepglobe_multiclass(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, way, img_transform=None, num_val=600, device=None):
        self.split = split
        self.benchmark = 'deepglobe'
        self.shot = shot
        self.way = way
        self.num_val = num_val

        self.base_path = os.path.join(datapath)
        self.to_annpath = lambda p: p.replace('jpg', 'png').replace('origin', 'groundtruth')

        self.categories = ['1','2','3','4','5','6']
        self.class_names = ['urban','agriculture', 'rangeland', 'forest', 'water', 'barren']
        self.class_ids = range(0, 6)
        self.cat_ids = [i for i in range(len(self.categories)+1)] # have bg
        
        self.img_metadata_classwise, self.num_images = self.build_img_metadata_classwise()

        self.transform = transform
        self.img_transform = img_transform
        if device is None:
            device = torch.device('cpu')
        self.device = device

    def __len__(self):
        # if it is the target domain, then also test on entire dataset
        return self.num_images if self.split !='val' else self.num_val

    def __getitem__(self, idx):
        
        query_name, support_names, selected_class_ids = self.sample_episode(idx)
        query_img, query_mask, query_name, query_img_shape, support_imgs, support_masks, support_names = self.load_frame(query_name, support_names, selected_class_ids)
        
        query_img = self.transform(query_img)
        query_mask = F.interpolate(query_mask.float(), query_img.size()[-2:], mode='nearest').squeeze()
        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])

        support_masks_tmp = []
        for smask in support_masks:
            smask = F.interpolate(smask.unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            support_masks_tmp.append(smask)
        support_masks = torch.stack(support_masks_tmp)
        
        batch = {'query_img': query_img.unsqueeze(0).type(torch.float).to(self.device).contiguous(),
                 'query_mask': query_mask.unsqueeze(0).type(torch.float).to(self.device).contiguous(),
                 'query_name': query_name, 
                 'query_ori_image_hw':query_img_shape,
                 'support_imgs': support_imgs.type(torch.float).to(self.device).contiguous(), 
                 'support_masks': support_masks.type(torch.float).to(self.device).contiguous(),
                 'support_names': support_names, 
                 'class_id': torch.tensor(selected_class_ids).type(torch.float).to(self.device).contiguous()
                } 
        

        return batch


    def load_frame(self, query_name, support_names, selected_class_ids):
        selected_class_ids.sort()
        selected_class_samples = [self.categories[i] for i in selected_class_ids]
        
        def find_full_path(file_list, base_name):
            for full_path in file_list:
                if os.path.splitext(os.path.basename(full_path))[0] == base_name:
                    return full_path
            return None

        def build_mask(img_shape, filename):
            """
            filename is the file selected to get all masks that is relevant in selected_class_samples for
            
            1. loop over class_ids
            2. get class_metadata which is list of images that have that class
            3. if class_sample is selected
            4. get full_path of image of the the class that matches filename
            5. if it exists convert full_path to mask_path
            6. read mask
            
            """
            mask_layers = [torch.zeros(img_shape)]
            for c_id in self.class_ids:
                class_sample = self.categories[c_id]
                class_metadata = self.img_metadata_classwise[class_sample]
                
                if class_sample in selected_class_samples:
                    
                    class_img_path = find_full_path(class_metadata, filename)
                    # images with the same base name but on different class directories
                    # are the same image
                    if class_img_path is not None:
                        ann_path = os.path.join(self.base_path, class_sample, 'test', 'groundtruth')
                        mask_path = os.path.join(ann_path, filename) + '.png'
                        mask = self.read_mask(mask_path)
                        mask = mask[:img_shape[0], :img_shape[1]]
                        mask_layers.append(mask)
                        
                    else:
                        mask_layers.append(torch.zeros(img_shape))
                else:
                    mask_layers.append(torch.zeros(img_shape))

            mask_stack = torch.stack(mask_layers)
            covered = mask_stack[1:].any(dim=0)
            mask_stack[0] = ~covered
            return mask_stack
        
        query_img = Image.open(query_name).convert('RGB')
        query_img_shape = (query_img.size[1], query_img.size[0])
        query_filename = os.path.splitext(os.path.basename(query_name))[0]
        query_mask = build_mask(query_img_shape, query_filename).unsqueeze(0)
        
        support_imgs = [Image.open(name).convert('RGB') for name in support_names]
        
        support_mask_list = []
        for support_name, support_img in zip(support_names, support_imgs):
            support_img_shape = (support_img.size[1], support_img.size[0])
            support_filename = os.path.splitext(os.path.basename(support_name))[0]
            support_mask = build_mask(support_img_shape, support_filename)
            support_mask_list.append(support_mask)
        

        return query_img, query_mask, query_name, query_img_shape, support_imgs, support_mask_list, support_names

    @lru_cache(maxsize=1000)
    def read_mask(self, img_name):
        mask = torch.tensor(np.array(Image.open(img_name).convert('L')))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    def sample_episode(self, idx):
        
        selected_class_ids = [idx % len(self.class_ids)]
        while len(selected_class_ids) !=self.way:
            select_class_id = random.sample(list(self.class_ids), 1)[0]
            if select_class_id not in selected_class_ids:
                selected_class_ids.append(select_class_id)
                
        # selected_class_ids = random.sample(list(self.class_ids), self.way) # dont sort
        
        class_samples = [self.categories[i] for i in selected_class_ids]
        query_class = class_samples[0]
        
        query_name = np.random.choice(self.img_metadata_classwise[query_class], 1, replace=False)[0]
        
        support_names =[]
        for class_sample in class_samples:
            class_shot=0 
            while True: 
                support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
                
                if query_name != support_name: 
                    support_names.append(support_name)
                    class_shot+=1
                if class_shot==self.shot:
                    break
        
        return query_name, support_names, selected_class_ids
    
    def transform_image(self,image_file):
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed=self.img_transform(image=image)
        return transformed['image']
    
    def revert_transform_numpy(self, np_array, ori_hw,):
        '''resize and remove pad'''
        np_array = cv2.resize(np_array, [ori_hw[1], ori_hw[0]])
        return np_array


    def build_img_metadata_classwise(self):
        num_images = 0
        img_metadata_classwise = {cat: [] for cat in self.categories}

        # Get all image paths in one go
        all_img_paths = glob.glob(os.path.join(self.base_path, '*', 'test', 'origin', '*.jpg'))

        for img_path in all_img_paths:
            category = img_path.split(os.sep)[-4]  # Extract category name from path
            if category in img_metadata_classwise:  # Ensure category is valid
                img_metadata_classwise[category].append(img_path)
                num_images += 1

        return img_metadata_classwise, num_images