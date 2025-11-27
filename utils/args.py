import json
import time
import os
import argparse
import re

import torch

from utils.io import randstr


def get_arguments(mode: str = 'train'):
    if mode == 'train':
        return get_train_arguments()
    else:
        return get_inference_arguments()


def save_arguments(args, path):
    with open(path + '/training_arguments.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)


def get_train_arguments():
    parser = argparse.ArgumentParser()
    # Initialize arguments
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--cupy', action='store_true', default=False)
    # Training arguments
    parser.add_argument('--batchSz', type=int, default=2)
    parser.add_argument('--nEpochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--save_all_freq', type=int, default=50)
    # Optimizer arguments
    parser.add_argument('--opt', type=str, default='adam',
                        choices=('sgd', 'adam', 'rmsprop', 'adamw', 'adagrad', 'adadelta', 'adamax', 'asgd', 'lbfgs'))
    parser.add_argument('--no_scheduler', action='store_true', default=False)
    parser.add_argument('--weight_decay', type=float, default=1e-10)
    # Model arguments
    parser.add_argument('--model', type=str, default='VNet')
    parser.add_argument('--backend', type=str, default='torchnn',
                        choices=('torchnn', 'spconv'))
    parser.add_argument('--classes', type=int, default=1)
    parser.add_argument('--inModalities', type=int, default=1)
    parser.add_argument('--inChannels', type=int, default=1)
    parser.add_argument('--interleaver_r', type=int, default=2)
    parser.add_argument('--shapescribed_n', type=int, default=4)
    parser.add_argument('--use_dyt', action='store_true', default=False)
    parser.add_argument('--model_depth', type=int, default=3)
    # Loss arguments
    parser.add_argument('--loss', type=str, default='dice')
    parser.add_argument('--loss_weights', type=str, default='')
    parser.add_argument('--dice_alpha', type=float, default=0.1)
    # Deprecated, use --dice_alpha instead
    parser.add_argument('--alpha', type=float, default=None)
    # Utility arguments
    parser.add_argument('--terminal_show_freq', default=50)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--resume_ref', type=str, default=None)
    parser.add_argument('--tag', type=str, default='')
    # Data arguments
    parser.add_argument('--root', type=str, default='./data')
    parser.add_argument('--out_dir', type=str, default='out')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default="tiny2")
    parser.add_argument('--test_fraction', type=float, required=True)
    parser.add_argument('--z_size', type=int, required=True)
    # Inference post-processing arguments
    parser.add_argument('--cache_size', type=int, default=0, 
                        help='Number of predictions to cache for temporal smoothing (0 to disable)')
    parser.add_argument('--max_pool_size', type=int, default=-1, 
                        help='Size of max pooling dilation kernel (-1 to disable, 0 for default=11)')

    args = parser.parse_args()

    return generate_args(args)


def get_inference_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='.')
    parser.add_argument('--out_dir', type=str, default='out')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--infer_tag', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default=None)
    parser.add_argument('--n_frames', type=int, default=None)
    parser.add_argument('--ckpt_suffix', type=str, default=None)
    # Overwrite the original settings
    parser.add_argument('--z_size', type=int, default=None)
    parser.add_argument('--loss', type=str, default='dice')
    parser.add_argument('--loss_weights', type=str, default='')
    parser.add_argument('--timing', action='store_true', default=False)
    # Inference post-processing arguments
    parser.add_argument('--cache_size', type=int, default=None, 
                        help='Number of predictions to cache for temporal smoothing (overrides training setting)')
    parser.add_argument('--max_pool_size', type=int, default=None, 
                        help='Size of max pooling dilation kernel (overrides training setting)')

    args = parser.parse_args()

    # Read training_arguments json file to get the training arguments
    # In older versions, saved to txt file
    args_file = os.path.join(
        args.root, args.out_dir, args.exp_name, 'training_arguments.json')
    if not os.path.exists(args_file):
        args_file = os.path.join(
            args.root, args.out_dir, args.exp_name, 'training_arguments.txt')
    with open(args_file, 'r') as f:
        infer_args = argparse.Namespace(**json.load(f))

    # Overwrite the training arguments with the inference arguments
    infer_args.root = args.root
    infer_args.out_dir = args.out_dir
    infer_args.save_dir = args.save_dir
    infer_args.exp_name = args.exp_name
    if args.dataset_name is not None:
        infer_args.dataset_name = args.dataset_name
    infer_args.n_frames = args.n_frames
    infer_args.loss = args.loss
    infer_args.loss_weights = args.loss_weights
    if args.infer_tag is not None:
        infer_args.tag = '{}-{}'.format(infer_args.tag, args.infer_tag)
    infer_args.timing = args.timing
    infer_args.save_all_freq = 0  # Disable saving during inference

    if args.z_size is not None:
        infer_args.z_size = args.z_size
    
    # Override post-processing arguments if provided
    if args.cache_size is not None:
        infer_args.cache_size = args.cache_size
    if args.max_pool_size is not None:
        infer_args.max_pool_size = args.max_pool_size

    # For compatibility with older versions
    if "use_dyt" not in infer_args:
        infer_args.use_dyt = False
    if "model_depth" not in infer_args:
        infer_args.model_depth = 3
    if "cache_size" not in infer_args:
        infer_args.cache_size = 0
    if "max_pool_size" not in infer_args:
        infer_args.max_pool_size = -1

        # infer_args.ckpt
    if args.ckpt_suffix is not None:
        ckpt_name = '{}_{}.pth'.format(infer_args.exp_name, args.ckpt_suffix)
        ckpt_dir = os.path.join(
            args.root, args.out_dir, args.exp_name, ckpt_name)
        infer_args.ckpt_suffix = args.ckpt_suffix
    else:
        # Find the `BEST.pth` checkpoint; if not, use `last_epoch.pth`
        ckpt_dir = os.path.join(args.root, args.out_dir, args.exp_name,
                                '{}_BEST.pth'.format(args.exp_name))
        print(args.root, args.exp_name)
        print("Checkpoint dir: ", ckpt_dir)
        if not os.path.exists(ckpt_dir):
            print("BEST checkpoint not found, using last_epoch.pth")
            ckpt_dir = os.path.join(
                args.root, args.out_dir, args.exp_name, '{}_last_epoch.pth'.format(args.exp_name))
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError("{} not found".format(ckpt_dir))
    infer_args.ckpt = ckpt_dir

    return generate_args(infer_args)


def generate_args(args: argparse.Namespace, infer: bool = False):
    name = '{}_{}_{}_{}_{}_{}'.format(args.model, args.dataset_name, args.loss,
                                        time.strftime("%Y%m%d-%H%M%S"),
                                        args.tag or "", randstr())
    if args.save_dir is not None:
        save_dir = args.save_dir
    else:
        save_dir = args.out_dir
    if infer:
        save_dir = os.path.join(args.root, save_dir, name, 'infer_{}'.format(
            time.strftime("%Y%m%d-%H%M%S")))
    else:
        save_dir = os.path.join(args.root, save_dir, name)
    args.save = save_dir
    args.dataset_path = os.path.join(args.root, "datasets", args.dataset_name)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Parse loss
    if ',' in args.loss:
        args.loss = args.loss.split(',')
    if ',' in args.loss_weights:
        args.loss_weights = [float(x) for x in args.loss_weights.split(',')]
    else:
        args.loss_weights = None

    # Parse deprecated arguments
    args = parse_deprecated_args(args)
    
    # Parse legacy cache and max pool settings from save_dir if arguments not explicitly set
    args = parse_legacy_postprocessing_args(args)
    
    return args


def parse_legacy_postprocessing_args(args):
    """Parse legacy cache and max pool settings from save_dir string using smart pattern matching"""
    if not (hasattr(args, 'save_dir') and args.save_dir is not None):
        # Ensure attributes exist even if save_dir is None
        if not hasattr(args, 'cache_size'):
            args.cache_size = 0
        if not hasattr(args, 'max_pool_size'):
            args.max_pool_size = -1
        return args
    
    save_dir = args.save_dir
    
    # Parse cache settings from save_dir if not explicitly set (cache_size == 0 means default/not set)
    if not hasattr(args, 'cache_size') or args.cache_size == 0:
        # Look for cache patterns: cache1, cache2, cache3, cache5, etc.
        cache_match = re.search(r'cache(\d+)', save_dir)
        if cache_match:
            args.cache_size = int(cache_match.group(1))
    
    # Parse max pool settings from save_dir if not explicitly set (max_pool_size == -1 means default/not set)
    if not hasattr(args, 'max_pool_size') or args.max_pool_size == -1:
        # Check for explicit disable first
        if 'nomp' in save_dir:
            args.max_pool_size = -1  # Explicitly disabled
        else:
            # Look for mp patterns: -mp3-, _mp5_, etc. (surrounded by non-word chars)
            mp_match = re.search(r'[-_\b]mp(\d+)[-_\b]', save_dir)
            if mp_match:
                args.max_pool_size = int(mp_match.group(1))
                print(f"Warning: Using legacy max_pool_size={args.max_pool_size} parsed from save_dir. Consider setting --max_pool_size explicitly.")
            # elif hasattr(args, 'cache_size') and args.cache_size > 0:
            #     # If no mp pattern found but cache is being used, default to 11
            #     args.max_pool_size = 11
    
    # Ensure attributes exist
    if not hasattr(args, 'cache_size'):
        args.cache_size = 0
    if not hasattr(args, 'max_pool_size'):
        args.max_pool_size = -1
        
    return args


def parse_deprecated_args(args):
    if args.alpha is not None:
        args.dice_alpha = args.alpha
    return args
