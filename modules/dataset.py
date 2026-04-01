import numpy as np

try:
    import cupy as cp
except Exception:
    cp = None

import torch
import torch.utils.data as data
from glob import glob
import os
import os.path
import gzip
import torch.utils.dlpack


test_split = []
train_split = []


def train_test_split(file_lists, test_fraction):
    """
    Accepts a list of file lists (one list per directory). Zips the lists,
    shuffles the list of tuples, splits according to test_fraction, and returns
    (train_split, test_split) as two lists of tuples.
    """
    # Each tuple contains one file from each file_list.
    paired_files = list(zip(*file_lists))

    # Randomize the order.
    np.random.shuffle(paired_files)

    # Determine the number of items to be used for the test set.
    test_count = int(np.ceil(len(paired_files) * test_fraction))

    # Split the data into test and train sets.
    test_split = paired_files[:test_count]
    train_split = paired_files[test_count:]

    # Shuffle the splits separately.
    np.random.shuffle(train_split)
    np.random.shuffle(test_split)

    return train_split, test_split


def make_dataset(paths, test_fraction, seed, mode, exts=None):
    """
    Accepts:
      - paths: a list of directory paths (at least two) where each directory
        contains files matching "*.bin.gz".
      - test_fraction: the fraction of the overall dataset to use for testing.
      - seed: a random seed for reproducibility.
      - mode: "train" to return training data, any other string returns test data.

    Returns a tuple of lists containing filenames from each path, corresponding
    to either the train or the test split.
    """
    if len(paths) < 2:
        raise ValueError("At least two paths must be provided.")

    # For each directory, get the sorted list of matching files.
    file_lists = [full_dataset(
        paths[i], ext=exts[i] if exts else "bin.gz") for i in range(len(paths))]

    # Seed the RNG for reproducibility.
    np.random.seed(seed)

    # Compute the splits.
    train_split, test_split = train_test_split(file_lists, test_fraction)

    # Choose the split according to the mode. Here, if train_split is empty, we use test_split.
    if mode == "train":
        keys = train_split if len(train_split) > 0 else test_split
    else:
        keys = test_split

    # Unpack the tuples back into separate lists (one per original path) and return.
    return tuple(map(list, zip(*keys)))


def full_dataset(path, ext="bin.gz"):
    if not os.path.exists(path):
        print("Path does not exist: {}".format(path))
        return []
    files = glob(os.path.join(path, "*.{}".format(ext)))
    # Filter out files that are not in the format "0000_gv.bin.gz" or "0000_pvv.bin.gz"
    files = [f for f in files if os.path.basename(f).split("_")[0].isdigit()]
    files = sorted(files, key=lambda x: int(os.path.basename(x).split("_")[0]))
    return files


def load_volume(path, amp, z_size, cupy=False):
    if not os.path.exists(path):
        return None
    with gzip.open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.int32)
    data = data.byteswap()
    data = data.view(np.uint8)
    size = np.int32(np.sqrt(data.nbytes*8/z_size))
    unpacked_data = np.unpackbits(data).reshape((size, size, z_size))
    # unpacked_data = cucim.skimage.measure.block_reduce(unpacked_data,4,cp.max)
    if amp:
        datatype = np.float16
    else:
        datatype = np.float32
    # Expand the dimensions to match the expected input shape.
    if cupy:
        if cp is None:
            raise ImportError("load_volume(..., cupy=True) requires cupy, but cupy is not installed.")
        data = cp.expand_dims(unpacked_data.astype(datatype), 0)
        return torch.utils.dlpack.from_dlpack(cp.from_dlpack(data))
    else:
        return np.expand_dims(unpacked_data.astype(datatype), 0)


def load_array(path, amp, cupy=False):
    if amp:
        datatype = np.float16
    else:
        datatype = np.float32
    if cupy:
        if cp is None:
            raise ImportError("load_array(..., cupy=True) requires cupy, but cupy is not installed.")
        data = cp.load(path).astype(datatype)
        return torch.utils.dlpack.from_dlpack(cp.from_dlpack(data))
    else:
        data = np.load(path).astype(datatype)
        return data


class PVSVoxelDataset(data.Dataset):
    def __init__(self, root='.', gv='gv', pvv='pvv', mode="train", test_fraction=0.10, seed=0, cupy=False, amp=False, z_size=256):
        if gv is None:
            raise (RuntimeError("GV must be set"))
        if pvv is None and mode != "infer":
            raise (RuntimeError("both GV and PVV must be set if mode is not 'infer'"))
        self.z_size = z_size
        self.mode = mode
        self.root = root
        self.gv_path = os.path.join(self.root, gv)
        self.pvv_path = os.path.join(self.root, pvv)
        self.cupy = cupy
        self.amp = amp
        if mode == "infer":
            self.gvs = full_dataset(self.gv_path)
            self.pvvs = full_dataset(self.pvv_path)
        else:
            self.gvs, self.pvvs = make_dataset(
                [self.gv_path, self.pvv_path], test_fraction, seed, self.mode)
        if (len(self.gvs) == 0):
            raise (RuntimeError("Found 0 files: " + os.path.join(root) + "\n"))
        if (mode != "infer" and len(self.gvs) != len(self.pvvs)):
            raise (RuntimeError("GVs and PVVs pair miscount!"))

    def __getitem__(self, index):
        gv = load_volume(self.gvs[index], self.amp, self.z_size, self.cupy)
        pvv = load_volume(self.pvvs[index], self.amp, self.z_size, self.cupy) if index < len(self.pvvs) else gv

        # Filter out pvv that are not in gv
        if self.cupy:
            pvv = torch.where(gv > 0, pvv, torch.zeros_like(pvv, dtype=pvv.dtype))
        else:
            pvv = np.where(gv > 0, pvv, np.zeros_like(pvv, dtype=pvv.dtype))

        return {
            "input": gv,
            "target": pvv,
            "extras": {
                "gv": "input", # point to the original gv
                "pvv": "target", # point to the original pvv
            },
        }

    def __len__(self):
        return len(self.gvs)
