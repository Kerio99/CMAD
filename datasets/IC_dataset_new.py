# --------------------------------------------------------
# Images Speak in Images: A Generalist Painter for In-Context Visual Learning (https://arxiv.org/abs/2212.02499)
# Github source: https://github.com/baaivision/Painter
# Copyright (c) 2022 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Xinlong Wang, Wen Wang
# Based on MAE, BEiT, detectron2, Mask2Former, bts, mmcv, mmdetetection, mmpose, MIRNet, MPRNet, and Uformer codebases
# --------------------------------------------------------'

import os.path
import json
from typing import Any, Callable, List, Optional, Tuple
import copy
import cv2
from .build import DATASET_REGISTRY
import logging

from PIL import Image
import numpy as np

from torchvision import utils as vutils

import torch
from torchvision.datasets.vision import VisionDataset, StandardTransform


from torch import nn
from torchvision import transforms
from prototype_utils import get_pt_from_imgPath


logging.getLogger('PIL').setLevel(logging.WARNING)

def tile_image(img, stride_ratio=0.8):
    height, width, _ = img.shape
    shorter_edge = torch.nn.functional.gaussian_blurmin(height, width)
    tile_size = shorter_edge
    stride = int(tile_size * stride_ratio)

    tile_image = []
    for y in range(0, height - tile_size + 1, stride):
        for x in range(0, width - tile_size + 1, stride):
            # Extract a tile from the image
            tile = img[y:y + tile_size, x:x + tile_size]
            tile_image.append(tile)

    return tile_image


@DATASET_REGISTRY.register()
class IC_dataset(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        normal_json_path_list: list,
        outlier_json_path_list: list,
        transform: Optional[Callable] = None,
        shot = None,
        modality = None,
        split = None, # train or test
        layer_list = None
    ) -> None:
        super().__init__(root)

        self.normal_samples = []
        self.outlier_samples = []
        self.image = []
        self.total_n = 0
        self.total_o = 0
        self.shot = shot
        
        self.modality = modality
        self.split = split
        self.layer_list = layer_list
        self.mask_transform = transforms.Compose([
                                        transforms.Resize(240, interpolation=transforms.InterpolationMode.BICUBIC),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),           # 转换为Tensor
                                        transforms.Resize((224, 224))  # 调整大小
                                        ])

        if len(normal_json_path_list) == 1:
            cur_normal = json.load(open(normal_json_path_list[0]))
            self.normal_samples.extend(cur_normal)
        else:
            for idx, json_path in enumerate(normal_json_path_list):
                cur_normal = json.load(open(json_path))
                self.normal_samples.extend(cur_normal)

        if len(outlier_json_path_list) == 1:
            cur_outlier = json.load(open(outlier_json_path_list[0]))
            self.outlier_samples.extend(cur_outlier)
        else:
            for idx, json_path in enumerate(outlier_json_path_list):
                cur_outlier = json.load(open(json_path))
                self.outlier_samples.extend(cur_outlier)

        self.transform = transform
        self.total_n, self.total_o = len(self.normal_samples), len(self.outlier_samples)
        self.image = self.normal_samples + self.outlier_samples
        # for train, 加速
        self.label_list = [0] * self.total_n + [1] * self.total_o

    def _load_image(self, path: str) -> Image.Image:
        if 'npy' in path[-3:]:
            img = np.load(path)
            return img
        if self.modality in ['rgb', 'depth', 'medical', 'thermal']:
            return Image.open(path).convert('RGB')
        else:
            raise ValueError(f"Unsupported modality: {self.modality}")

    def _combine_images(self, image, image2):
        h, w = image.shape[1], image.shape[2]
        dst = torch.cat([image, image2], dim=1)
        return dst

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return_item = {}
        sample = self.image[index]
        image = self._load_image(sample['image_path'])
        global_pt, patch_pt = get_pt_from_imgPath([sample['image_path']], self.layer_list)
        return_item['token'] = global_pt.squeeze(0)[0]
        return_item['Fp_list'] = patch_pt.squeeze(0)
        label = sample['target']
        sample_type = sample['type']

        # decide mode for interpolation
        cur_transforms = self.transform
        image = cur_transforms(image)
        
        path_list = [sample['image_path']]

        image_type = sample_type
        
        if "train" in sample['image_path']:
            mask = torch.zeros(image.shape[1:], dtype=image.dtype).unsqueeze(0)
        elif "test" in sample['image_path']:
            if "good" in sample['image_path']:
                mask = torch.zeros(image.shape[1:], dtype=image.dtype).unsqueeze(0)
            else:
                suffix = sample['image_path'].split('.')[-1]
                mask_path = sample['image_path'].replace("test", "ground_truth").replace(f".{suffix}", "_mask.png")
                if os.path.exists(mask_path):
                    mask = Image.open(mask_path).convert('L')
                    mask = self.mask_transform(mask)
                    mask = torch.where(mask < 0.5, torch.zeros_like(mask), torch.ones_like(mask))
                    mask = mask
                else:
                    mask = torch.ones(image.shape[1:], dtype=image.dtype).unsqueeze(0)
                    
        return_item['image_type'] = image_type
        return_item['label'] = label
        return_item['path_list'] = path_list
        return_item['mask'] = mask
        
        return return_item
        

    def __len__(self) -> int:
        return len(self.image)
