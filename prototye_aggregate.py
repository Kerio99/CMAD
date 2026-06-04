import os
import torch

import open_clip


_imagebind = None

model = open_clip.model.CMAD(None, None, None, None, None, cast_dtype=None, imagebind=_imagebind)
# model = model.cuda(device="cuda:0")


prompt_list = {}

for ckpt_base_path in ["./logs/train"]:

    for ckpt_dataset_path in sorted(os.listdir(ckpt_base_path)):
        
        if ckpt_dataset_path.split('_')[0] == 'trained':
        
            with torch.no_grad():
            
                dataset_name = ckpt_dataset_path.replace('trained_on_', '')
                
                ckpt_path = os.path.join(ckpt_base_path, ckpt_dataset_path, "checkpoints", "checkpoint_1500.pyth")

                model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
                
                prompt_global = model.prototype_global.features.detach().clone()                
                prompt_patch = model.prototype_patch.features.detach().clone()
                    
                prompt_list[dataset_name] = {
                    'global': prompt_global,
                    'patch': prompt_patch
                }

torch.save(prompt_list, 'prototype_list.pt')
