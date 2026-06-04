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
from PIL import Image

from constant import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

from constant import category_list
import csv

from prototype_utils import get_few_shot_samples, get_closest_prototype, infer_global, infer_patch
from einops import rearrange

logger = logging.get_logger(__name__)

def _convert_to_rgb(image):
    return image.convert('RGB')

@torch.no_grad()
def eval_epoch(val_loader, model, cfg, tokenizer, few_shot_samples, mode=None):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            open_clip/config/defaults.py
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()

    total_label = torch.Tensor([]).cuda()
    total_pred = torch.Tensor([]).cuda()
    
    for _, data_dict in enumerate(val_loader):
        inputs = data_dict
        types = data_dict['image_type']
        labels = data_dict['label']
        
        if cfg.NUM_GPUS:
            labels = labels.cuda()

        prototype_global, prototype_patch, token, Fp_list, patch_res = model(tokenizer, inputs, types, few_shot_samples, cfg)

        prototype_global_list, prototype_global_weight_list, prototype_patch_list, prototype_patch_weight_list = get_closest_prototype(token, Fp_list, cfg.prototype_list_path, cfg.test_dataset_name)
        
        preds_global = infer_global(prototype_global_list, prototype_global_weight_list, token)
        
        preds_map = infer_patch(prototype_patch_list, prototype_patch_weight_list, Fp_list)
               
        preds_map = rearrange(preds_map, 'b h w -> b (h w)')
        preds_topk, _ = torch.topk(preds_map, 50, dim=-1)
        preds_patch = preds_topk.mean(dim=-1)
        
        preds = 0.8*preds_global + 0.2*preds_patch

        # enable patch_res fusion for few-shot
        if int(cfg.shot) != 0:
            preds = 0.5 * preds + 0.5 * patch_res

        total_pred = torch.cat((total_pred, preds), 0)
        total_label = torch.cat((total_label, labels), 0)


    total_pred = total_pred.cpu().numpy()  #.squeeze()
    total_label = total_label.cpu().numpy()
    # print("Predict " + mode + " set: ")
    total_roc, total_pr = aucPerformance(total_pred, total_label)

    return total_roc, total_pr

def aucPerformance(mse, labels, prt=True):
    roc_auc = roc_auc_score(labels, mse)
    ap = average_precision_score(labels, mse)
    # if prt:
    #     print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    return roc_auc, ap


def test(cfg, load=None, mode = None):
    """
    Perform testing on the pretrained model.
    Args:
        cfg (CfgNode): configs. Details can be found in open_clip/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    device = torch.cuda.current_device()
    
    transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD),
        ]
    )

    cf = './open_clip/model_configs/ViT-B-16-plus-240.json'
    with open(cf, 'r') as f:
        model_cfg = json.load(f)
    embed_dim = model_cfg["embed_dim"]
    vision_cfg = model_cfg["vision_cfg"]
    text_cfg = model_cfg["text_cfg"]
    cast_dtype = get_cast_dtype('fp32')
    quick_gelu = False

    model = open_clip.model.CMAD(cfg, embed_dim, vision_cfg, text_cfg, quick_gelu, cast_dtype=cast_dtype, imagebind=None)
    model = model.cuda(device=device)
    
    tokenizer = None

    _category_list = category_list[cfg.test_dataset_name]
    
    csv_file = os.path.join(cfg.output_result_file, cfg.test_dataset_name, f"{cfg.shot}shot_result.csv")
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    roc_scores = []
    pr_scores = []
    
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Category', 'AUC-ROC', 'AUC-PR'])
        
        print(f"\n***Test On {cfg.test_dataset_name} | {cfg.shot} shot***")
        
        data_processed_json_root = getattr(cfg, "data_processed_json_root", "./data_processed_json")

        for category in _category_list:
            
            cfg.val_normal_json_path = [rf"{data_processed_json_root}/{cfg.test_dataset_name}/{category}_val_normal.json"]
            cfg.val_outlier_json_path = [rf"{data_processed_json_root}/{cfg.test_dataset_name}/{category}_val_outlier.json"]
            cfg.category = category

            if load == None:
                load = loader.construct_loader(cfg, "test", transform)
                mode = "test"
            
            test_dataset_name = cfg.test_dataset_name
            test_category = category

            shot_num = int(cfg.shot)
            if shot_num <= 0:
                few_shot_samples = None
            else:
                # few-shot inference: load support samples/features for current dataset & category
                few_shot_samples = get_few_shot_samples(
                    test_dataset_name,
                    test_category,
                    shot_num,
                    cfg.layer_list,
                )

            # Create meters.
            total_roc, total_pr = eval_epoch(load, model, cfg, tokenizer, few_shot_samples, mode)
        
            roc_scores.append(total_roc)
            pr_scores.append(total_pr)

            # Write the results to the CSV file
            writer.writerow([category, total_roc, total_pr])
            file.flush()
            
            print(f"Category: {category}, I-ROC: {total_roc}, I-PR: {total_pr}")
            
            # important !!!
            load = None
            
        avg_roc = sum(roc_scores) / len(roc_scores)
        avg_pr = sum(pr_scores) / len(pr_scores)
        # Write the average values in the last row
        writer.writerow(['Average', avg_roc, avg_pr])
        print(f"Average I-ROC: {avg_roc}, Average I-PR: {avg_pr}")
    
    return total_roc
