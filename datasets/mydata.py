import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np

class Mydata():
    """
    UAVid dataset from ISPRS 
    UAVid: A Semantic Segmentation Dataset for UAV Imagery
    https://arxiv.org/abs/1810.10438

    Download dataset from here: https://uavid.nl/ 

    """

    UavidClass = namedtuple('UavidClass', ['name', 'id', 'train_id', 'category', 'category_id', 'has_instances', 'ignore_in_eval', 'color'])

    classes = [
        UavidClass('clutter',       0, 255, 'void', 0, False, True, (0, 0, 0)), 
        UavidClass('building',      1, 0,   'void', 1, False, True, (128, 0, 0)),
        UavidClass('road',          2, 1,   'void', 2, False, True, (128, 64, 128)),
        UavidClass('tree',          3, 2,   'void', 3, False, True, (0, 128, 0)),
        UavidClass('vegetation',    4, 3,   'void', 4, False, True, (128, 128, 0)),
        UavidClass('moving car',    5, 4,   'void', 5, False, True, (64, 0, 128)),
        UavidClass('static car',    6, 5,   'void', 5, False, True, (192, 0, 192)),
        UavidClass('human',         7, 6,   'void', 6, False, True, (64, 64, 0)) 
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    # <class 'list'> [(128, 0, 0), (128, 64, 128), (0, 128, 0), (128, 128, 0), (64, 0, 128), (192, 0, 192), (64, 64, 0)] 
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])
    
    # [[128   0   0][128  64 128][  0 128   0][128 128   0][ 64   0 128][192   0 192][ 64  64   0][  0   0   0]]

    def __init__(self, root, split='train', mode='fine', target_type='semantic', transform=None):
        print("I am here")
        self.root = os.path.expanduser(root)
        # self.mode = 'gtFine'
        self.target_type = target_type
        self.images_dir = os.path.join(self.root, split)
        print(f'images_dir {self.images_dir}')

        self.targets_dir = os.path.join(self.root, split)
        print(f'targets_dir {self.targets_dir}')
        self.transform = transform

        self.split = split
        self.images = []
        self.targets = []

        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train" split="test" or split="val"')

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the specified "split" and "mode" are inside the "root" directory')
        
        for uav_data in os.listdir(self.images_dir):
            if uav_data == 'images':
                img_dir = os.path.join(self.images_dir, uav_data)
                # print(img_dir, 'img_dir')
                for file_name in os.listdir(img_dir):
                    self.images.append(os.path.join(img_dir, file_name))
                
            if uav_data == 'masks':
                target_dir = os.path.join(self.targets_dir, uav_data)
                
                for file_name in os.listdir(target_dir):
                    self.targets.append(os.path.join(target_dir, file_name))
        
        print(f'images -> {self.images} \n masks -> {self.targets}')


    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])
        if self.transform:
            image, target = self.transform(image, target)
        target = self.encode_target(target)
        return image, target

    def __len__(self):
        return len(self.images)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        elif target_type == 'polygon':
            return '{}_polygons.json'.format(mode)
        elif target_type == 'depth':
            return '{}_disparity.png'.format(mode)        



m = Mydata("D:\FKIE\git_workspace\DeepLabV3Plus-Pytorch\datasets\data")
print(m)