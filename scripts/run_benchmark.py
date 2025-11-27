import setup_paths  # noqa: F401

import os
import sys
import subprocess
import argparse

from generate_exp_list import find_folders_with_pth


def run_eval(root_dir, dataset_name, exp_name, exp_path, save_path, n_frames=None, keyword=""):
    """
    Run evaluation on a specific checkpoint and dataset.

    Args:
        root_dir (str): Root directory for the project.
        dataset_name (str): Name of the dataset to evaluate on.
        exp_name (str): Name of the experiment (folder) containing the model.
        out_path (str): Path to save the output.
        save_path (str): Path to the model checkpoint.
    """
    infer_tag = "benchmark-{}".format(dataset_name)
    command = [
        sys.executable, "model/infer.py",
        "--root", root_dir,
        "--save_dir", save_path,
        "--out_dir", exp_path,
        "--exp_name", exp_name,
        "--dataset", dataset_name,
        "--timing"
    ]

    if n_frames is not None:
        command += ["--n_frames", str(n_frames)]

    if keyword:
        infer_tag += "-{}".format(keyword)
    command += ["--infer_tag", infer_tag]

    if not os.path.exists(os.path.join(root_dir, "datasets", dataset_name)):
        print("Dataset {} not found in {}.".format(
            dataset_name, os.path.join(root_dir, "datasets")))
        return False

    # Execute the command
    try:
        subprocess.run(command, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during evaluation: {e}")
        print(f"Command output: {e.output}")
        return False
    return True


def get_datasets(fov, size, z_size, datasets=None):
    prefix = "r{}v{}z{}".format(fov, size, z_size)
    default_datasets = [
        'viking',
        'robotlab',
        'bigcity',
        'industrial',
        'sponza',
    ]
    if not datasets:
        datasets = default_datasets
    elif isinstance(datasets, str):
        datasets = datasets.split(",")
    return ["{}-{}".format(prefix, dataset) for dataset in datasets], prefix


def main():
    parser = argparse.ArgumentParser(
        description="Run timing benchmark on a specific checkpoint and dataset.")
    parser.add_argument("--root", type=str, required=True,
                        help="Root directory for the project.")
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

    args = parser.parse_args()

    keywords = ['30']
    second_keywors = ['DUkFW7to','pNCQ35Ol','QaugfUAX']

    for keyword in keywords:
                
        datasets, prefix = get_datasets(
            int(keyword), args.size, args.z_size, args.datasets)

        matched_exps = find_folders_with_pth(
            os.path.join(args.root, args.exp_path), keyword)
        if len(matched_exps) == 0:
            print(f"No folders found with .pth files for keyword: {keyword}")
            continue
        # Find the first folder that contains the second keyword
        for second_keyword in second_keywors:
            for dataset in datasets:
                finished = False
                for exp in matched_exps:
                    if finished:
                        break
                    if second_keyword in os.path.basename(exp):
                        
                            print(f"Running evaluation for {dataset} with {keyword} and {second_keyword} on {exp}")
                            finished = True
                            # Run evaluation for the matched experiment and dataset
                            run_eval(
                                args.root,
                                dataset_name=dataset,
                                exp_name=os.path.basename(exp),
                                exp_path=args.exp_path,
                                save_path=args.save_path,
                                keyword=second_keyword,
                                n_frames=60,
                            )
                            break
                    
   
    # exps = find_folders_with_pth(
    #     os.path.join(args.root, args.exp_path), args.keyword)
    # if not exps:
    #     print("No folders found with .pth files.")
    #     return
    # exps = [os.path.basename(exp) for exp in exps if prefix in exp]
    # num_eval = len(exps) * len(datasets)

    # print("Found {} exps, {} datasets. Total: {}".format(
    #     len(exps), len(datasets), num_eval))

    # # Execute evaluation for each experiment and dataset
    # num_exps = len(exps)
    # num_datasets = len(datasets)
    # num_eval = num_exps * num_datasets

    # i = 0
    # with tqdm(total=num_eval, desc="Starting evaluations") as pbar:
    #     for exp in exps:
    #         for j, dataset in enumerate(datasets):
    #             # update the text shown next to the bar
    #             pbar.set_description(
    #                 f"[{i}/{num_eval-1} – {j}/{num_datasets-1}] "
    #                 f"Running {exp} on {dataset}"
    #             )
    #             run_eval(
    #                 args.root,
    #                 dataset_name=dataset,
    #                 exp_name=exp,
    #                 exp_path=args.exp_path,
    #                 save_path=args.save_path,
    #                 n_frames=args.n_frames,
    #             )
    #             i += 1
    #             pbar.update(1)
    print("Evaluation completed.")


if __name__ == "__main__":
    main()
