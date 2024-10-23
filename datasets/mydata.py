import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np

class Mydata(data.Dataset):
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

        print(self.images_dir)
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
        
        # print(f'images -> {self.images} \n masks -> {self.targets}')


    @classmethod
    def encode_target(cls, target):
        # print(f'encode_target {cls, target}')
        target_array = np.array(target)
        return cls.id_to_train_id[target_array]
    
    @classmethod
    def encode_target(cls, target):
        '''
        In UAVid, the target image is an RGB image, meaning that each pixel needs to be treated as an RGB tuple.
        Whereas in Cityscapes target image is grayscale, each pixel represents a class. 

        pixel_rgb = [128, 0, 0]  # This pixel is red
        pixel_grayscale = 1  # This pixel represents "road"

        This function converts the RGB segmentation mask (target image) into a class ID format and maps
        these class IDs to trainable IDs. 
        The encoding happens by matching each pixel’s RGB value with the predefined color for each class, 
        and then converting it to the corresponding training ID.
        
        1. Converts the input target (PIL image) into a NumPy array.
        2. Creates an empty array to store the class IDs for each pixel.
        3. Iterates through each pixel in the target array, extracts the RGB value, and matches it to
           a predefined class color.
        4. Once the RGB value is matched to a class color, the corresponding class ID is assigned.
        5. Class IDs are mapped to train IDs using `cls.id_to_train_id`.
        '''
        target_array = np.array(target)

        # Initialize an array for storing the mapped train IDs
        train_id_array = np.zeros((target_array.shape[0], target_array.shape[1]), dtype=np.int32)
        # print(train_id_array.shape)
        # Iterate over each pixel in the target image
        for i in range(target_array.shape[0]):
            for j in range(target_array.shape[1]):
                pixel_color = target_array[i, j]  # Get the RGB color of the pixel

                # Check if the pixel matches a known color
                match = np.where(np.all(cls.train_id_to_color == pixel_color, axis=1))[0]

                if len(match) > 0:

                    train_id_array[i, j] = cls.id_to_train_id[match[0]]
                else:
                    # Handle unknown colors
                    train_id_array[i, j] = 255  # 255 can represent 'void' or clutter

        return train_id_array


    @classmethod
    def decode_target(cls, target):
        # target[target == 255] = 19
        if target == 255:
            target = 7
        #target = target.astype('uint8') + 1
        # print(cls.train_id_to_color, 'cls.train_id_to_color')
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
        print(f'image {image}')

        target = Image.open(self.targets[index])
        print(f'target {np.array(target).shape}')

        if self.transform:
            image, target = self.transform(image, target)
        target = self.encode_target(target)
        print(f' new target ===> {target.shape}')
        return image, target

    def __len__(self):
        '''
        Returns total number of images/samples in the dataset.
        '''
        return len(self.images)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    # def _get_target_suffix(self, mode, target_type):
    #     if target_type == 'instance':
    #         return '{}_instanceIds.png'.format(mode)
    #     elif target_type == 'semantic':
    #         return '{}_labelIds.png'.format(mode)
    #     elif target_type == 'color':
    #         return '{}_color.png'.format(mode)
    #     elif target_type == 'polygon':
    #         return '{}_polygons.json'.format(mode)
    #     elif target_type == 'depth':
    #         return '{}_disparity.png'.format(mode)        



m = Mydata("D:\FKIE\git_workspace\DeepLabV3Plus-Pytorch\datasets\data")
print(m)
# encode_targets = m.encode_target([0,2,3])
# print(encode_targets)
# decode_targets = m.decode_target(255)
# print(decode_targets)

m.__getitem__(1)