# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

"""Train/Evaluation workflow."""
import os
import random
import json

import torchvision
import open_clip
from open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss
import open_clip.utils.checkpoint as cu
import open_clip.utils.distributed as du
import open_clip.utils.logging as logging
import open_clip.utils.misc as misc
import numpy as np
import torch
from datasets import loader
from torchvision import transforms
from open_clip.utils.meters import EpochTimer, TrainMeter, ValMeter
from sklearn.metrics import average_precision_score, roc_auc_score
import torch.distributed as dist
import matplotlib.pyplot as plt
from open_clip.model import get_cast_dtype
from open_clip.utils.env import checkpoint_pathmgr as pathmgr

from constant import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from datetime import datetime

from prototype_utils import FocalLoss, BinaryDiceLoss, compute_loss

logger = logging.get_logger(__name__)

def _convert_to_rgb(image):
    return image.convert('RGB')


def train(cfg):
    """
    Train a model on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in open_clip/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    if cfg.NUM_GPUS:
        device = torch.cuda.current_device()

    # Build the model and print model statistics.
    cf = './open_clip/model_configs/ViT-B-16-plus-240.json'
    with open(cf, 'r') as f:
        model_cfg = json.load(f)
    embed_dim = model_cfg["embed_dim"]
    vision_cfg = model_cfg["vision_cfg"]
    text_cfg = model_cfg["text_cfg"]
    cast_dtype = get_cast_dtype('fp32')
    quick_gelu = False


    model = open_clip.model.CMAD(cfg, embed_dim, vision_cfg, text_cfg, quick_gelu, cast_dtype=cast_dtype)

    if torch.cuda.is_available():
        assert (
            cfg.NUM_GPUS <= torch.cuda.device_count()
        ), "Cannot use more GPU devices than available"
    else:
        assert (
            cfg.NUM_GPUS == 0
        ), "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs."

    if cfg.NUM_GPUS:
        # Transfer the model to the current GPU device
        model = model.cuda(device=device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[device], output_device=device
        )
  
    transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD),
        ]
    )
    
    optimizer = torch.optim.Adam(list(model.prototype_global.parameters()) + list(model.prototype_patch.parameters()), lr=1e-3, betas=(0.5, 0.999))

    # Create the train loader.
    train_loader = loader.construct_loader(cfg, "train", transform)

    tokenizer = None

    start_iter = 0
    print("Start iter: {}".format(start_iter+1))
    
    with open(os.path.join(cfg.output_dir, f"logs.txt"), 'w') as f:
        f.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    loss_focal_fun = FocalLoss()
    loss_dice_fun = BinaryDiceLoss()
    
    
    max_iter = 1500 
    all_global_loss = 0.0
    all_patch_loss = 0.0
    cur_iter = 0
    for _ in range(0, 400):
        for _, data_dict in enumerate(train_loader):
            inputs = data_dict
            types = data_dict['image_type']
            labels = data_dict['label']
            masks = data_dict['mask']
            
            if cfg.NUM_GPUS:
                labels = labels.cuda()

            prototype_global, prototype_patch, embedding_global, embedding_patch_list, _ = model(tokenizer, inputs, types, None, cfg) # 2, 1024 | b c | b m l c
            
            # Compute the loss.
            global_loss, patch_loss = compute_loss(prototype_global, prototype_patch, embedding_global, embedding_patch_list, labels, masks, loss_focal_fun, loss_dice_fun)
            
            # check Nan Loss.
            misc.check_nan_losses(patch_loss + global_loss)
            # Perform the backward pass.
            optimizer.zero_grad()
            (patch_loss + global_loss).backward()
            # Update the parameters.
            optimizer.step()
            
            all_patch_loss = all_patch_loss + patch_loss.item()
            all_global_loss = all_global_loss + global_loss.item()
    
            print_every_iters = 100
            if (cur_iter + 1) % print_every_iters == 0:
                all_global_loss_avg = all_global_loss / print_every_iters
                all_patch_loss_avg = all_patch_loss / print_every_iters
                all_global_loss = 0.0
                all_patch_loss = 0.0
                print(f"Iter: {cur_iter+1}, train_loss: ", (all_global_loss_avg + all_patch_loss_avg))
                
                path = os.path.join(cfg.output_dir, "checkpoints", f"checkpoint_{cur_iter+1}.pyth")
                torch.save(model.state_dict(), path)

                with open(os.path.join(cfg.output_dir, f"logs.txt"), 'a') as f:
                    f.write(f"Iter: {cur_iter+1}, Global loss: {all_global_loss_avg}, Patch loss: {all_patch_loss_avg}\n")
                
            cur_iter += 1
                        
        if (cur_iter+1) >= max_iter:    
            break
         
    with open(os.path.join(cfg.output_dir, f"logs.txt"), 'a') as f:
        f.write(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

