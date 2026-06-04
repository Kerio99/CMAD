import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import cv2

from constant import rgb_modal_list, depth_modal_list


def parse_imgPath(imgPath):
    path_parts = os.path.normpath(imgPath).split(os.sep)
    image_name = os.path.splitext(path_parts[-1])[0]
    anomaly_type = path_parts[-2]
    train_or_test = path_parts[-3]
    category = path_parts[-4]
    dataset_name = path_parts[-5]
    modality = path_parts[-6]
    return modality, dataset_name, category, train_or_test, anomaly_type, image_name


def get_pt_from_imgPath(imgPath_list, layer_list,
                        pt_root=""
                        ):
    
    global_list_batch = []
    patch_list_batch = []
    
    for imgPath in imgPath_list:
        
        modality, dataset_name, category, train_or_test, anomaly_type, image_name = parse_imgPath(imgPath)
        
        global_list_single = []
        patch_list_single = []
        
        for layer in layer_list:
            
            pt_dir = os.path.join(
                pt_root,
                f"mvtec_format_layer{layer}",
                modality,
                dataset_name,
                category,
                train_or_test,
                anomaly_type,
            )
            
            pt_path = os.path.join(pt_dir, f"{image_name}.pt")
            pt = torch.load(pt_path, map_location='cpu')
            
            global_list_single.append(pt['global'])
            patch_list_single.append(pt['patch'])
            
        global_pt_single = torch.stack(global_list_single)
        patch_pt_single = torch.stack(patch_list_single)
        
        global_list_batch.append(global_pt_single)
        patch_list_batch.append(patch_pt_single)
        
    global_list_batch = torch.stack(global_list_batch)
    patch_list_batch = torch.stack(patch_list_batch)

    return global_list_batch, patch_list_batch


def get_few_shot_samples(test_dataset_name, test_category, shot, layer_list, 
                         few_shot_samples_root=""
                         ):
    
    few_shot_samples_global = list()
    few_shot_samples_patch = list()
    
    for layer in layer_list:
        
        few_shot_path = os.path.join(few_shot_samples_root, f"layer{layer}", test_dataset_name, str(shot), f"{test_category}.pt")
        pt = torch.load(few_shot_path, map_location='cpu')
        
        few_shot_samples_global.append(pt['global'])
        few_shot_samples_patch.append(pt['patch'])
        
    few_shot_samples_global = torch.stack(few_shot_samples_global)  # m s c
    few_shot_samples_patch = torch.stack(few_shot_samples_patch)  # m s c h w
    
    few_shot_samples = {'global': few_shot_samples_global[0], 'patch': few_shot_samples_patch}

    return few_shot_samples


def softmax_with_temperature(values, temperature=1.0):
    values = torch.tensor(values)
    softmax_values = torch.softmax(values / temperature, dim=0)
    return softmax_values.tolist()


def get_closest_prototype(token, Fp_list, prototype_list_path, test_dataset_name):
    prototype_list = torch.load(prototype_list_path)

    token = token.mean(dim=0)
    token = token / token.norm(dim=-1, keepdim=True)
    Fp_list = Fp_list.mean(dim=0)
    Fp_list = Fp_list / Fp_list.norm(dim=-1, keepdim=True)

    Fp_list = rearrange(Fp_list, 'm l c -> (m l) c')

    global_closest_distance = -10
    patch_closest_distance = -10

    if test_dataset_name in ["mvtec", "visa", "btad", "mpdd"]:
        usable_dataset_list = depth_modal_list
    elif test_dataset_name in ["mvtec3d_depth", "eyecandies_normals"]:
        usable_dataset_list = rgb_modal_list
    elif test_dataset_name in ["brainad", "retina_oct2017", "wtb", "leaves"]:
        usable_dataset_list = depth_modal_list + rgb_modal_list
    else:
        usable_dataset_list = depth_modal_list + rgb_modal_list

    prototype_global_list = []
    prototype_global_weight_list = []
    prototype_patch_list = []
    prototype_patch_weight_list = []
    for dataset_name in prototype_list.keys():
        if dataset_name in usable_dataset_list:
            prototype_global = prototype_list[dataset_name]['global'].to(token.device)
            prototype_global = prototype_global / prototype_global.norm(dim=-1, keepdim=True)

            global_dis = torch.max((token @ prototype_global.T))
            if global_dis > global_closest_distance:
                global_closest_distance = global_dis

            prototype_global_list.append(prototype_global)
            prototype_global_weight_list.append(global_dis.item())

            prototype_patch = prototype_list[dataset_name]['patch']
            prototype_patch = prototype_patch / prototype_patch.norm(dim=-1, keepdim=True)
            prototype_patch = prototype_patch.to(Fp_list.device)

            patch_dis = torch.max(torch.mean((Fp_list @ prototype_patch.T), dim=0))
            if patch_dis > patch_closest_distance:
                patch_closest_distance = patch_dis

            prototype_patch_list.append(prototype_patch)
            prototype_patch_weight_list.append(patch_dis.item())

    prototype_global_weight_list = softmax_with_temperature(prototype_global_weight_list, temperature=0.04)
    prototype_patch_weight_list = prototype_global_weight_list

    return prototype_global_list, prototype_global_weight_list, prototype_patch_list, prototype_patch_weight_list


class FocalLoss(nn.Module):
    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha
        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), self.gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        return loss


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        N = targets.size()[0]
        smooth = 1
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)
        intersection = input_flat * targets_flat
        n_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        loss = 1 - n_dice_eff.sum() / N
        return loss


class Prototype_global(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.empty(2, 1024)
        nn.init.normal_(self.features, std=0.02)
        self.features = nn.Parameter(self.features)

    def forward(self, cls_id=None):
        return self.features


class Prototype_patch(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.empty(2, 1280)
        nn.init.normal_(self.features, std=0.02)
        self.features = nn.Parameter(self.features)

    def forward(self, cls_id=None):
        return self.features


def get_similarity_map(sm, shape):
    side = int(sm.shape[1] ** 0.5)
    sm = sm.reshape(sm.shape[0], side, side, -1).permute(0, 3, 1, 2)
    sm = torch.nn.functional.interpolate(sm, shape, mode='bilinear')
    sm = sm.permute(0, 2, 3, 1)
    return sm


def compute_similarity(image_features, text_features, t=2):
    b, n_t, n_i, c = image_features.shape[0], text_features.shape[0], image_features.shape[1], image_features.shape[2]
    feats = image_features.reshape(b, n_i, 1, c) * text_features.reshape(1, 1, n_t, c)
    similarity = feats.sum(-1)
    return (similarity / 0.07).softmax(-1)


def compute_loss(prototype_global, prototype_patch, token, Fp_list, label, mask, loss_focal, loss_dice):
    image_features = token / token.norm(dim=-1, keepdim=True)

    text_features_global = torch.stack(torch.chunk(prototype_global, dim=0, chunks=2), dim=1)
    text_features_global = text_features_global / text_features_global.norm(dim=-1, keepdim=True)

    text_probs = image_features.unsqueeze(1) @ text_features_global.permute(0, 2, 1)
    text_probs = text_probs[:, 0, ...] / 0.07
    global_loss = F.cross_entropy(text_probs.squeeze(), label.long().cuda())

    Fp_list = rearrange(Fp_list, 'b m l c -> m b l c')
    patch_features = Fp_list

    similarity_map_list = []
    text_features_patch = torch.stack(torch.chunk(prototype_patch, dim=0, chunks=2), dim=1)
    text_features_patch = text_features_patch / text_features_patch.norm(dim=-1, keepdim=True)

    for patch_feature in patch_features:
        patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
        similarity = compute_similarity(patch_feature, text_features_patch[0])
        similarity_map = get_similarity_map(similarity, 224).permute(0, 3, 1, 2)
        similarity_map_list.append(similarity_map)

    gt = mask.squeeze(1).to(prototype_global.device)
    patch_loss = 0
    for i in range(len(similarity_map_list)):
        patch_loss += loss_focal(similarity_map_list[i], gt)
        patch_loss += loss_dice(similarity_map_list[i][:, 1, :, :], gt)
        patch_loss += loss_dice(similarity_map_list[i][:, 0, :, :], 1 - gt)

    return global_loss, patch_loss


def infer_global(prototype_global_list, prototype_global_weight_list, token):
    text_probs_weighted = 0

    for prototype_global, prototype_global_weight in zip(prototype_global_list, prototype_global_weight_list):
        text_features = prototype_global
        image_features = token

        text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_probs = image_features @ text_features.permute(0, 2, 1)
        text_probs = (text_probs / 0.07).softmax(-1)

        text_probs = text_probs[0, :, 1]
        text_probs = text_probs * prototype_global_weight
        text_probs_weighted += text_probs

    return text_probs_weighted


def infer_patch(prototype_patch_list, prototype_patch_weight_list, Fp_list):
    preds_map_weighted = 0

    Fp_list = rearrange(Fp_list, 'b m l c -> m b l c')
    patch_features = Fp_list

    for prototype_patch, prototype_patch_weight in zip(prototype_patch_list, prototype_patch_weight_list):
        similarity_map_list = []
        text_features_patch = torch.stack(torch.chunk(prototype_patch, dim=0, chunks=2), dim=1)
        text_features_patch = text_features_patch / text_features_patch.norm(dim=-1, keepdim=True)

        for patch_feature in patch_features:
            patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
            similarity = compute_similarity(patch_feature, text_features_patch[0])
            similarity_map = get_similarity_map(similarity, 224).permute(0, 3, 1, 2)
            similarity_map_list.append(similarity_map)

        preds_map = torch.stack(similarity_map_list)
        preds_map = preds_map.mean(dim=0)[:, 1, :, :]

        preds_map = preds_map * prototype_patch_weight
        preds_map_weighted += preds_map

    return preds_map_weighted
