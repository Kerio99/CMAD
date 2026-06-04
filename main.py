# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

"""Wrapper to train/test models."""

import argparse
import sys
import torch
from engine_IC import train
from open_clip.utils.misc import launch_job
import open_clip.utils.checkpoint as cu
from open_clip.config.defaults import assert_and_infer_cfg, get_cfg



def parse_args():
    """
    Parse the following arguments for a default parser.
    Args:
        shard_id (int): shard id for the current machine. Starts from 0 to
            num_shards - 1. If single machine is used, then set shard id to 0.
        num_shards (int): number of shards using by the job.
        init_method (str): initialization method to launch the job with multiple
            devices. Options includes TCP or shared file-system for
            initialization. details can be find in
            https://pytorch.org/docs/stable/distributed.html#tcp-initialization
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
    """
    parser = argparse.ArgumentParser(
        description="Provide training and testing pipeline."
    )
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:8888",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See mvit/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--model",
        help="model_name",
        default="ViT-B-16",
        type=str,
    )
    parser.add_argument(
        "--pretrained",
        help="whether use pretarined model",
        default=None,
        type=str
    )
    parser.add_argument('--normal_json_path', default='', nargs='+', type=str, help='json path')
    parser.add_argument('--outlier_json_path', default='', nargs='+', type=str, help='json path')
    parser.add_argument("--shot", type=int, default=0, help="size for visual prompts")
    parser.add_argument("--image_size", type=int, default=224, help="image size")
    parser.add_argument("--output_dir", default="", type=str, help="output path for training")
    parser.add_argument("--modality", default="", type=str, help="modality")
    parser.add_argument('--layer_list', nargs='+', type=int, help='List of integers representing layers')

    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()

    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
        cfg.NUM_SHARDS = args.num_shards
        cfg.SHARD_ID = args.shard_id
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
        
    if hasattr(args, "output_dir"):
        cfg.output_dir = args.output_dir
        
    if hasattr(args, "normal_json_path"):
        cfg.normal_json_path = args.normal_json_path
        
    if hasattr(args, "outlier_json_path"):
        cfg.outlier_json_path = args.outlier_json_path

    if hasattr(args, "local_rank"):
        cfg.local_rank = args.local_rank

    if hasattr(args, "model"):
        cfg.model = args.model

    if hasattr(args, "pretrained"):
        cfg.pretrained = args.pretrained

    if hasattr(args, "shot"):
        cfg.shot = args.shot

    if hasattr(args, "image_size"):
        cfg.image_size = args.image_size
        
    if hasattr(args, "modality"):
        cfg.modality = args.modality  
        
    if hasattr(args, "layer_list"):
        cfg.layer_list = args.layer_list

    return cfg


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)

    # Perform training.
    if cfg.TRAIN.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=train)


if __name__ == "__main__":
    main()
