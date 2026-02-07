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


class DatasetSUIM_multiclass(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, way, img_transform=None, num_val=600, device=None):
        self.split = split
        self.benchmark = 'suim'
        self.shot = shot
        self.way = way
        self.num_val = num_val

        self.base_path = os.path.join(datapath)
        self.img_path = os.path.join(self.base_path, 'images')
        self.ann_path = os.path.join(self.base_path, 'masks')

        self.categories = ['FV','HD','PF','RI','RO','SR','WR']
        self.class_names = ['FV','HD','PF','RI','RO','SR','WR']
        self.class_ids = range(len(self.categories)) # no bg
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
                 'query_name': query_name,
                 'query_ori_image_hw':query_img_shape,
                 'support_imgs': support_imgs.type(torch.float).to(self.device).contiguous(), 
                 'support_masks': support_masks.type(torch.float).to(self.device).contiguous(), 
                 'support_names': support_names, 
                 'class_id': torch.tensor(selected_class_ids).type(torch.float).to(self.device).contiguous()
                } 


        return batch


    def load_frame(self, query_mask_path, support_mask_paths, selected_class_ids):
        selected_class_ids.sort()
        selected_class_samples = [self.categories[i] for i in selected_class_ids]

        def maskpath_to_imgpath(maskpath):
            filename = os.path.splitext(os.path.basename(maskpath))[0]
            return os.path.join(self.img_path, filename + '.jpg')

        def find_full_path(file_list, base_name):
            for full_path in file_list:
                if os.path.splitext(os.path.basename(full_path))[0] == base_name:
                    return full_path
            return None

        def build_mask(img_shape, filename):
            mask_layers = [torch.zeros(img_shape)]
            for c_id in self.class_ids:
                class_sample = self.categories[c_id]
                class_metadata = self.img_metadata_classwise[class_sample]

                if class_sample in selected_class_samples:
                    mask_path = find_full_path(class_metadata, filename)
                    if mask_path is not None:
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

        # Load query image
        query_name = maskpath_to_imgpath(query_mask_path)
        query_img = Image.open(query_name).convert('RGB')
        query_img_shape = (query_img.size[1], query_img.size[0])
        query_filename = os.path.splitext(os.path.basename(query_name))[0]
        query_mask = build_mask(query_img_shape, query_filename).unsqueeze(0)

        # Load support images and masks
        support_names = [maskpath_to_imgpath(path) for path in support_mask_paths]
        support_imgs = [Image.open(name).convert('RGB') for name in support_names]

        support_mask_list = []
        for support_name, support_img in zip(support_names, support_imgs):
            support_img_shape = (support_img.size[1], support_img.size[0])
            support_filename = os.path.splitext(os.path.basename(support_name))[0]
            support_mask = build_mask(support_img_shape, support_filename)
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
        # np_array = cv2.resize(np_array, ori_hw)
        np_array = cv2.resize(np_array, [ori_hw[1], ori_hw[0]])
        # np_array = np_array[:ori_hw[0],:ori_hw[1]]
        return np_array

    def build_img_metadata_classwise(self):
        num_images = 0
        img_metadata_classwise = {cat: [] for cat in self.categories}

        # Get all mask paths at once to avoid multiple glob calls
        all_mask_paths = glob.glob(os.path.join(self.ann_path, '*', '*'))

        for mask_path in all_mask_paths:
            category = mask_path.split(os.sep)[-2]  # Extract category name from path
            if category not in img_metadata_classwise:
                continue  # Skip if category is not recognized

            # Fast check if the mask is empty
            with Image.open(mask_path) as mask:
                if mask.getextrema()[1] > 0:  # If max pixel value > 0, mask is not empty
                    img_metadata_classwise[category].append(mask_path)
                    num_images += 1

        return img_metadata_classwise, num_images
