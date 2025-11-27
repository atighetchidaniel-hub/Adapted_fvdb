import setup_paths  # noqa: F401

import os
import sys
import subprocess
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from generate_exp_list import find_folders_with_pth


def run_eval(root_dir, dataset_name, exp_name, exp_path, save_path, cache_size=None, max_pool_size=None, n_frames=None, ckpt_suffix=None):
    """
    Run evaluation on a specific checkpoint and dataset.

    Args:
        root_dir (str): Root directory for the project.
        dataset_name (str): Name of the dataset to evaluate on.
        exp_name (str): Name of the experiment (folder) containing the model.
        exp_path (str): Path to the experiment directory.
        save_path (str): Path to save the output.
        cache_size (int, optional): Number of predictions to cache.
        max_pool_size (int, optional): Size of max pooling dilation kernel.
        n_frames (int, optional): Number of frames to evaluate.
        ckpt_suffix (str, optional): Suffix of the checkpoint to evaluate.
        
    Returns:
        tuple: (success: bool, exp_name: str, dataset_name: str, ckpt_suffix: str, error_msg: str)
    """
    command = [
        sys.executable, "infer.py",
        "--root", root_dir,
        "--save_dir", save_path,
        "--out_dir", exp_path,
        "--exp_name", exp_name,
        "--dataset", dataset_name,
        "--infer_tag", "eval-{}".format(dataset_name),
    ]

    if n_frames is not None:
        command += ["--n_frames", str(n_frames)]
    if cache_size is not None:
        command += ["--cache_size", str(cache_size)]
    if max_pool_size is not None:
        command += ["--max_pool_size", str(max_pool_size)]
    if ckpt_suffix is not None:
        command += ["--ckpt_suffix", str(ckpt_suffix)]

    if not os.path.exists(os.path.join(root_dir, "datasets", dataset_name)):
        error_msg = "Dataset {} not found in {}.".format(
            dataset_name, os.path.join(root_dir, "datasets"))
        return False, exp_name, dataset_name, ckpt_suffix, error_msg

    # Execute the command
    try:
        result = subprocess.run(command, capture_output=True, check=True, text=True)
        return True, exp_name, dataset_name, ckpt_suffix, ""
    except subprocess.CalledProcessError as e:
        error_msg = f"Error during evaluation: {e}\nStdout: {e.stdout}\nStderr: {e.stderr}"
        return False, exp_name, dataset_name, ckpt_suffix, error_msg
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        return False, exp_name, dataset_name, ckpt_suffix, error_msg


def get_datasets(fov, size, z_size, datasets=None):
    prefix = "r{}v{}z{}".format(fov, size, z_size)
    default_datasets = [
        'viking_d30',
        'robotlab_d30',
        'bigcity_d30',
        'industrial_d30',
        'sponza_d30',
    ]
    if not datasets:
        datasets = default_datasets
    elif isinstance(datasets, str):
        datasets = datasets.split(",")
    return ["{}-{}".format(prefix, dataset) for dataset in datasets], prefix


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation on a specific checkpoint and dataset.")
    parser.add_argument("--root", type=str, required=True,
                        help="Root directory for the project.")
    parser.add_argument("--fov", type=int, default=60,
                        help="Field of view for the camera.")
    parser.add_argument("--size", type=int, default=256,
                        help="Size of the dataset.")
    parser.add_argument("--z_size", type=int, default=256,
                        help="Depth of the dataset.")
    parser.add_argument("--datasets", type=str, default=None,
                        help="List of datasets to evaluate on.")
    parser.add_argument("--exp_path", type=str, required=True,
                        help="Path to the experiment directory.")
    parser.add_argument("--save_path", type=str,
                        required=True, help="Path to save the output.")
    parser.add_argument("--keyword", type=str, default="",
                        help="Keyword to match in folder names.")
    parser.add_argument("--n_frames", type=int, default=None,
                        help="Number of frames to evaluate.")
    # Inference post-processing arguments
    parser.add_argument('--cache_size', type=int, default=None, 
                        help='Number of predictions to cache for temporal smoothing (overrides training setting)')
    parser.add_argument('--max_pool_size', type=int, default=None, 
                        help='Size of max pooling dilation kernel (overrides training setting)')
    parser.add_argument('--use_all_ckpt', action='store_true', 
                        help='Use all .pth files in the experiment folder for evaluation')
    parser.add_argument('--ckpt_suffix', type=str, default=None,
                        help='Suffix of the checkpoint to evaluate (if not using all ckpts)')
    parser.add_argument('--max_concurrent', type=int, default=1,
                        help='Maximum number of concurrent evaluations to run (default: 1 for sequential execution)')

    args = parser.parse_args()

    datasets, prefix = get_datasets(
        args.fov, args.size, args.z_size, args.datasets)

    exps = find_folders_with_pth(
        os.path.join(args.root, args.exp_path), args.keyword)
    if not exps:
        print("No folders found with .pth files.")
        return
    exps = [os.path.basename(exp) for exp in exps if prefix in exp]
    num_eval = len(exps) * len(datasets)

    print("Found {} exps, {} datasets. Total: {}".format(
        len(exps), len(datasets), num_eval))

    # Prepare all evaluation tasks
    eval_tasks = []
    
    for exp in exps:
        if not args.use_all_ckpt:
            for dataset in datasets:
                eval_tasks.append({
                    'root_dir': args.root,
                    'dataset_name': dataset,
                    'exp_name': exp,
                    'exp_path': args.exp_path,
                    'save_path': args.save_path,
                    'n_frames': args.n_frames,
                    'cache_size': args.cache_size,
                    'max_pool_size': args.max_pool_size,
                    'ckpt_suffix': args.ckpt_suffix,
                })
        else:
            exp_folder = os.path.join(args.root, args.exp_path, exp)
            if not os.path.exists(exp_folder):
                print(f"Experiment folder {exp_folder} not found. Skipping.")
                continue
            pth_files = [f for f in os.listdir(exp_folder) if f.endswith('.pth')]
            if not pth_files:
                print(f"No .pth files found in {exp_folder}. Skipping.")
                continue
            for dataset in datasets:
                for pth_file in pth_files:
                    # exp_name + _ + ckpt_suffix + .pth
                    ckpt_suffix = pth_file[len(exp)+1:-len('.pth')]  # remove .pth
                    eval_tasks.append({
                        'root_dir': args.root,
                        'dataset_name': dataset,
                        'exp_name': exp,
                        'exp_path': args.exp_path,
                        'save_path': args.save_path,
                        'n_frames': args.n_frames,
                        'cache_size': args.cache_size,
                        'max_pool_size': args.max_pool_size,
                        'ckpt_suffix': ckpt_suffix,
                    })

    total_tasks = len(eval_tasks)
    print(f"Total evaluation tasks: {total_tasks}")
    print(f"Running with max concurrent workers: {args.max_concurrent}")

    # Execute evaluations concurrently or sequentially based on max_concurrent
    if args.max_concurrent == 1:
        # Sequential execution (original behavior)
        run_evaluations_sequential(eval_tasks)
    else:
        # Concurrent execution
        run_evaluations_concurrent(eval_tasks, args.max_concurrent)
        
    print("Evaluation completed.")


def run_evaluations_sequential(eval_tasks):
    """Run evaluations sequentially with progress tracking."""
    failed_tasks = []
    
    with tqdm(total=len(eval_tasks), desc="Running evaluations") as pbar:
        for i, task in enumerate(eval_tasks):
            pbar.set_description(
                f"[{i+1}/{len(eval_tasks)}] Running {task['exp_name']} on {task['dataset_name']}"
                + (f" (ckpt: {task['ckpt_suffix']})" if task['ckpt_suffix'] else "")
            )
            
            success, exp_name, dataset_name, ckpt_suffix, error_msg = run_eval(**task)
            
            if not success:
                failed_tasks.append((exp_name, dataset_name, ckpt_suffix, error_msg))
                print(f"\nFAILED: {exp_name} on {dataset_name}" 
                      + (f" (ckpt: {ckpt_suffix})" if ckpt_suffix else ""))
                print(f"Error: {error_msg}")
            
            pbar.update(1)
    
    if failed_tasks:
        print(f"\n{len(failed_tasks)} evaluations failed:")
        for exp_name, dataset_name, ckpt_suffix, error_msg in failed_tasks:
            print(f"  - {exp_name} on {dataset_name}" 
                  + (f" (ckpt: {ckpt_suffix})" if ckpt_suffix else ""))


def run_evaluations_concurrent(eval_tasks, max_concurrent):
    """Run evaluations concurrently with progress tracking."""
    completed_tasks = 0
    failed_tasks = []
    lock = threading.Lock()
    
    def update_progress(future):
        nonlocal completed_tasks
        success, exp_name, dataset_name, ckpt_suffix, error_msg = future.result()
        
        with lock:
            if not success:
                failed_tasks.append((exp_name, dataset_name, ckpt_suffix, error_msg))
                print(f"\nFAILED: {exp_name} on {dataset_name}" 
                      + (f" (ckpt: {ckpt_suffix})" if ckpt_suffix else ""))
                print(f"Error: {error_msg}")
            
            completed_tasks += 1
    
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        with tqdm(total=len(eval_tasks), desc="Running evaluations") as pbar:
            # Submit all tasks
            futures = []
            for task in eval_tasks:
                future = executor.submit(run_eval, **task)
                future.add_done_callback(lambda f: pbar.update(1))
                future.add_done_callback(update_progress)
                futures.append(future)
            
            # Wait for all tasks to complete
            for future in as_completed(futures):
                pass  # Progress is updated in callbacks
    
    if failed_tasks:
        print(f"\n{len(failed_tasks)} evaluations failed:")
        for exp_name, dataset_name, ckpt_suffix, error_msg in failed_tasks:
            print(f"  - {exp_name} on {dataset_name}" 
                  + (f" (ckpt: {ckpt_suffix})" if ckpt_suffix else ""))


if __name__ == "__main__":
    main()
