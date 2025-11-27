#!/usr/bin/env python
import os
import glob
import argparse


def find_folders_with_pth(dir, keyword) -> list[str]:
    """
    Find all folders in the current directory that match the given keyword
    and contain at least one *.pth file.

    Args:
        keyword (str): The keyword to match in folder names.

    Returns:
        list: A list of folder paths that match the criteria.
    """
    matching_folders = []

    # Iterate through all folders in the current directory
    for folder in os.listdir(dir):
        folder_path = os.path.join(dir, folder)
        if os.path.isdir(folder_path) and (not keyword or keyword in folder):
            # Check if the folder contains at least one *.pth file
            pth_files = glob.glob(os.path.join(folder_path, "*.pth"))
            if pth_files:
                exp_name = os.path.basename(folder_path)
                matching_folders.append(exp_name)

    return matching_folders


def main():
    # Get path from arguments
    parser = argparse.ArgumentParser(
        description="Find folders with .pth files.")
    parser.add_argument(
        "--dir", type=str, default=".", help="Directory to search for folders")
    parser.add_argument(
        "--keyword", type=str, default="", help="Keyword to match in folder names")

    args = parser.parse_args()
    dir = args.dir
    keyword = args.keyword
    # Find folders with .pth files
    matching_folders = find_folders_with_pth(dir, keyword)
    if not matching_folders:
        print("No folders found with .pth files.")
    else:
        print("Matching folders:")
        # Replace any `,` with `\,` in the folder names
        matching_folders = [folder.replace(",", "\\,")
                            for folder in matching_folders]
        # Join the folder names with commas
        folder_string = ",".join(matching_folders)
        # Print the result
        print(folder_string)


if __name__ == "__main__":
    main()
