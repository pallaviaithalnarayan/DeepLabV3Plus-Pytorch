import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
# import matplotlib.pyplot as plt

class Custompark(data.Dataset):
    
    """
    CustomDataset: load from citypark environment generated using unreal with airsim computer vision mode
    """

    DatasetClass = namedtuple('DatasetClass', ['name', 'id', 'train_id', 'category', 'category_id', 'has_instances', 'ignore_in_eval', 'color'])

    custom_classes = [
        DatasetClass('clutter', 0, 255, 'void', 0, False, True, (0, 0, 0)),     
        DatasetClass('building', 1, 0, 'void', 1, False, True, (196, 30, 8)),    
        DatasetClass('road', 2, 1, 'void', 2, False, True, (102, 16, 239)),   
        DatasetClass('tree', 3, 2, 'void', 3, False, True, (11, 236, 9)),    
        DatasetClass('vegetation', 4, 3, 'void', 4, False, True, (153, 108, 6)),   
        DatasetClass('moving car', 5, 4, 'void', 5, False, True, (0,0,0)),
        DatasetClass('vehicle', 6, 5, 'void', 5, False, True, (242, 107, 146)),  
        DatasetClass('human', 7, 6, 'void', 6, False, True, (255, 255, 255))  
    ]

    train_id_to_color = [c.color for c in custom_classes if (c.train_id != -1 and c.train_id != 255)]
    # <class 'list'> [(128, 0, 0), (128, 64, 128), (0, 128, 0), (128, 128, 0), (64, 0, 128), (192, 0, 192), (64, 64, 0)] 
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in custom_classes])
    print(train_id_to_color, id_to_train_id)
    # [[128   0   0][128  64 128][  0 128   0][128 128   0][ 64   0 128][192   0 192][ 64  64   0][  0   0   0]]

    def __init__(self, root, split='train', mode='fine', target_type='semantic', transform=None):
        # print("I am here")
        self.root = os.path.expanduser(root)
        # self.mode = 'gtFine'
        self.target_type = target_type
        self.images_dir = os.path.join(self.root, split)
        # print(f'images_dir {self.images_dir}')

        self.targets_dir = os.path.join(self.root, split)
        # print(f'targets_dir {self.targets_dir}')
        self.transform = transform

        self.split = split
        self.images = []
        self.targets = []

        split_dir = os.path.join(self.root, split)
        img_dir = os.path.join(split_dir, 'Images')
        mask_dir = os.path.join(split_dir, 'Labels')
        self.images = sorted([os.path.join(img_dir, img_file) for img_file in os.listdir(img_dir)])
        if split != 'test':
            self.targets = sorted([os.path.join(mask_dir, mask_file) for mask_file in os.listdir(mask_dir)])
                    
        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train" split="test" or split="val"')

        # print(self.images_dir)
        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the specified "split" and "mode" are inside the "root" directory')
        
        if split == 'test' and not self.targets:
            print("Test set detected. Only images will be loaded.")

        elif len(self.images) != len(self.targets):
            raise RuntimeError("The number of images and masks do not match. Please check the dataset structure.")
    
    @classmethod
    def encode_target(cls, target):
        '''
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
        target[target == 255] = len(cls.train_id_to_color) - 1
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
        # print(f'image {image}')
        
        if self.split == 'test':
            if self.transform:
                image = self.transform(image)
            return image
        
        target = Image.open(self.targets[index])
        # print(f'target {np.array(target).shape}')

        if self.transform:
            image, target = self.transform(image, target)
        target = self.encode_target(target)
        # print(f' new target ===> {target.shape}')
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



# if __name__ == "__main__":
#     dataset = CustomDataset(root="D:/FKIE/git_workspace/DeepLabV3Plus-Pytorch/datasets/data", split="train")
    
#     # print("Color to Train ID Mapping:", dataset.color_to_train_id)
#     # print("Train ID to Color Mapping:", dataset.id_to_train_id)

#     sample_target_path = dataset.targets[1]
#     sample_target = Image.open(sample_target_path).convert('RGB')
#     sample_target_array = np.array(sample_target)

#     encoded_target = dataset.encode_target(sample_target_array)
#     print("Encoded Target:\n", encoded_target)

#     decoded_target = dataset.decode_target(encoded_target)
#     print("Decoded Target:\n", decoded_target)

#     # Plotting original, encoded, and decoded images
#     fig, ax = plt.subplots(1, 3, figsize=(15, 5))
#     ax[0].imshow(sample_target_array)
#     ax[0].set_title("Original Target")
#     ax[0].axis("off")

#     ax[1].imshow(encoded_target, cmap='tab20')
#     ax[1].set_title("Encoded Target")
#     ax[1].axis("off")

#     ax[2].imshow(decoded_target)
#     ax[2].set_title("Decoded Target")
#     ax[2].axis("off")

#     plt.show()