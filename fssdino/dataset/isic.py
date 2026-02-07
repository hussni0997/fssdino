# taken from https://github.com/Vision-Kek/ABCDFSS/tree/main/data
r""" ISIC few-shot semantic segmentation dataset """
import os
import glob
import cv2
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
import random

    
class DatasetISIC_multiclass(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, way, 
                 img_transform=None, num_val=600, device=None):
        self.split = split
        self.benchmark = 'isic'
        self.shot = shot
        self.way = way
        self.num_val = num_val

        self.base_path = os.path.join(datapath)
        self.categories = ['1', '2', '3']
        self.class_names = ['nevus', 'melanoma', 'other']
        
        self.class_ids = range(0, 3)
        self.cat_ids = [i for i in range(len(self.categories)+1)] # have bg
        
        self.img_metadata_classwise,self.num_images = self.build_img_metadata_classwise()

        self.transform = transform
        self.img_transform = img_transform
        if device is None:
            device = torch.device('cpu')
        self.device = device

    def __len__(self):
        return self.num_images if self.split != 'val' else self.num_val

    def __getitem__(self, idx):
        query_name, support_names, selected_class_ids = self.sample_episode(idx)
        query_img, query_mask, query_name, query_img_shape, support_imgs, support_masks, support_names = \
        self.load_frame(query_name, support_names, selected_class_ids)
        
        query_img = self.transform(query_img)
        query_mask = F.interpolate(query_mask.float(), query_img.size()[-2:], mode='nearest').squeeze()
        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])

        support_masks_tmp = []
        for smask in support_masks:
            smask = F.interpolate(smask.unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            support_masks_tmp.append(smask.squeeze(0))
        support_masks = torch.stack(support_masks_tmp)

        batch = {'query_img': query_img.unsqueeze(0).type(torch.float).to(self.device).contiguous(),
                 'query_mask': query_mask.unsqueeze(0).type(torch.float).to(self.device).contiguous(),
                 'query_name': query_name, # REMOVE
                 'query_ori_image_hw':query_img_shape,
                 'support_imgs': support_imgs.type(torch.float).to(self.device).contiguous(), # REMOVE
                 'support_masks': support_masks.type(torch.float).to(self.device).contiguous(), # REMOVE
                 'support_names': support_names, # REMOVE
                 'class_id': torch.tensor(selected_class_ids).type(torch.float).to(self.device).contiguous()
                } 

        return batch

    def load_frame(self, query_name, support_names, selected_class_ids):
        # selected_class_ids.sort()
        selected_class_samples = [self.categories[i] for i in selected_class_ids]
        
        q_class_sample = selected_class_samples[0]
        
        def build_mask(img_shape, filename, class_id):
            mask_layers = [torch.zeros(img_shape)]
            for c_id in self.class_ids:
                class_sample = self.categories[c_id]
                # if class_sample in selected_class_samples:
                if class_id == class_sample:
                    mask_path = filename
                    mask = self.read_mask(mask_path)
                    mask = mask[:img_shape[0], :img_shape[1]]
                    mask_layers.append(mask)
                else:
                    mask_layers.append(torch.zeros(img_shape))
            mask_stack = torch.stack(mask_layers)
            covered = mask_stack[1:].any(dim=0)
            mask_stack[0] = ~covered
            return mask_stack
                        
        
        query_img = Image.open(query_name).convert('RGB')
        support_imgs = [Image.open(name).convert('RGB') for name in support_names]
        query_id = query_name.split('/')[-1].split('.')[0]
        support_ids = [name.split('/')[-1].split('.')[0] for name in support_names]
        query_img_shape = (query_img.size[1], query_img.size[0])
        
        ann_path = os.path.join(self.base_path, 'ISIC2018_Task1_Training_GroundTruth')
        
        # even though this is multiclass dataset but each image only have single class
        # so we dont have to check other available classes for an image, only take
        
        query_gt_name = os.path.join(ann_path, query_id) + '_segmentation.png'
        support_gt_names= [os.path.join(ann_path, sid) + '_segmentation.png' for name, sid in zip(support_names, support_ids)]
        
        
        # build mask
        query_mask = build_mask(query_img_shape, query_gt_name, q_class_sample).unsqueeze(0)
        
        # selected_class_samples = support_class_sample
        support_class_samples = [support_class.split('/')[-2] for support_class in support_names]
        support_mask_list = []
        for support_name, support_img, support_class_sample  in zip(support_gt_names, support_imgs, support_class_samples):
            support_img_shape = (support_img.size[1], support_img.size[0])
            support_filename = support_name
            support_mask = build_mask(support_img_shape, support_filename, support_class_sample)
            support_mask_list.append(support_mask)

        return query_img, query_mask, query_name, query_img_shape, support_imgs, support_mask_list, support_names


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
        # np_array = cv2.resize(np_array, ori_hw)
        np_array = cv2.resize(np_array, [ori_hw[1], ori_hw[0]])
        # np_array = np_array[:ori_hw[0],:ori_hw[1]]
        return np_array

    
    def build_img_metadata(self):
        img_metadata = []
        for cat in self.categories:
            os.path.join(self.base_path, cat)
            img_paths = sorted([path for path in glob.glob('%s/*' % os.path.join(self.base_path, 'ISIC2018_Task1-2_Training_Input', cat))])
            for img_path in img_paths:
                if os.path.basename(img_path).split('.')[1] == 'jpg':
                    img_metadata.append(img_path)
        return img_metadata

    def build_img_metadata_classwise(self):
        num_images=0
        img_metadata_classwise = {}
        for cat in self.categories:
            img_metadata_classwise[cat] = []

        for cat in self.categories:
            img_paths = sorted([path for path in glob.glob('%s/*' % os.path.join(self.base_path, 'ISIC2018_Task1-2_Training_Input', cat))])
            for img_path in img_paths:
                if os.path.basename(img_path).split('.')[1] == 'jpg':
                    img_metadata_classwise[cat] += [img_path]
                    num_images += 1
        return img_metadata_classwise, num_images