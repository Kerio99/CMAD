# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

"""Configs (trimmed to currently used fields)."""
from fvcore.common.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()

# -----------------------------------------------------------------------------
# Training options
# -----------------------------------------------------------------------------
_C.TRAIN = CfgNode()
_C.TRAIN.ENABLE = True
_C.TRAIN.DATASET = "IC_dataset"
_C.TRAIN.BATCH_SIZE = 16

# -----------------------------------------------------------------------------
# Testing options
# -----------------------------------------------------------------------------
_C.TEST = CfgNode()
_C.TEST.ENABLE = True
_C.TEST.BATCH_SIZE = 16
_C.TEST.CHECKPOINT_FILE_PATH = ""

# -----------------------------------------------------------------------------
# Solver options (only fields used by assert_and_infer_cfg)
# -----------------------------------------------------------------------------
_C.SOLVER = CfgNode()
_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.WARMUP_START_LR = 1e-8
_C.SOLVER.COSINE_END_LR = 1e-6
_C.SOLVER.BASE_LR_SCALE_NUM_SHARDS = False

# -----------------------------------------------------------------------------
# Misc options
# -----------------------------------------------------------------------------
_C.NUM_GPUS = 1
_C.NUM_SHARDS = 1
_C.SHARD_ID = 0
_C.OUTPUT_DIR = "./tmp"
_C.RNG_SEED = 10
_C.DIST_BACKEND = "nccl"
_C.local_rank = 0

# Project-specific runtime options
_C.normal_json_path = './datasets/AD_json/hyperkvasir_normal.json'
_C.outlier_json_path = './datasets/AD_json/hyperkvasir_outlier.json'
_C.val_normal_json_path = './datasets/AD_json/elpv_normal.json'
_C.val_outlier_json_path = './datasets/AD_json/elpv_outlier.json'

_C.model = 'ViT-B-16'
_C.pretrained = None
_C.shot = 0
_C.image_size = 224
_C.few_shot_dir = "./visa"

# test-time / command-line populated fields
_C.modality = ""
_C.layer_list = []
_C.test_dataset_name = ""
_C.output_result_file = ""
_C.prototype_list_path = "./prototype_list.pt"
_C.steps_per_epoch = 100

# kept for loader compatibility
_C.AUG = CfgNode()
_C.AUG.NUM_SAMPLE = 1

# -----------------------------------------------------------------------------
# Common train/test data loader options
# -----------------------------------------------------------------------------
_C.DATA_LOADER = CfgNode()
_C.DATA_LOADER.data_path = "./data"
_C.DATA_LOADER.NUM_WORKERS = 16
_C.DATA_LOADER.PIN_MEMORY = True


def assert_and_infer_cfg(cfg):
    # TRAIN assertions.
    assert cfg.NUM_GPUS == 0 or cfg.TRAIN.BATCH_SIZE % cfg.NUM_GPUS == 0

    # TEST assertions.
    assert cfg.NUM_GPUS == 0 or cfg.TEST.BATCH_SIZE % cfg.NUM_GPUS == 0

    # Execute LR scaling by num_shards.
    if cfg.SOLVER.BASE_LR_SCALE_NUM_SHARDS:
        cfg.SOLVER.BASE_LR *= cfg.NUM_SHARDS
        cfg.SOLVER.WARMUP_START_LR *= cfg.NUM_SHARDS
        cfg.SOLVER.COSINE_END_LR *= cfg.NUM_SHARDS

    # General assertions.
    assert cfg.SHARD_ID < cfg.NUM_SHARDS
    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()
