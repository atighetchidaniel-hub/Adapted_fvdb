import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from utils.tensor import FvdbTensor


def _feature_values(values: torch.Tensor) -> np.ndarray:
    if values.ndim > 1:
        values = values[:, 0]
    return values.detach().cpu().numpy()


def plot_hist(data: np.ndarray | torch.Tensor | FvdbTensor):
    if isinstance(data, torch.Tensor):
        data = torch.sigmoid(data.reshape(-1)).detach().cpu().numpy()
    elif isinstance(data, FvdbTensor):
        data = torch.sigmoid(data.data.jdata.reshape(-1)).detach().cpu().numpy()

    data = np.random.choice(data, min(10000, len(data)))

    plt.figure(figsize=(8, 5))
    sns.histplot(data, bins=30, kde=True, edgecolor="black", alpha=0.7)

    print(f"Mean: {np.mean(data)}")
    print(f"Median: {np.median(data)}")
    print(f"Std: {np.std(data)}")
    print(f"Max: {np.max(data)}")
    print(f"Min: {np.min(data)}")
    print(f"Has NaN: {np.isnan(data).any()}")
    plt.show()


def scatter_3d(ax, data, title, vmin, vmax):
    """
    Creates a 3D scatter plot on the given axis.

    Supported sparse inputs:
      - FvdbTensor
      - tuple (coords, values)
    """
    if isinstance(data, torch.Tensor):
        dense_data = data.detach().cpu()
        while dense_data.dim() > 3:
            assert 1 in dense_data.shape, "Only 1D batch dimension is supported"
            dense_data = dense_data.squeeze(0)
        dense_np = dense_data.numpy()
        nonzero_idx = np.nonzero(dense_np)
        if len(nonzero_idx[0]) == 0:
            return
        coords = np.stack(nonzero_idx, axis=1)
        values = dense_np[nonzero_idx]
        sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=values, cmap="viridis", vmin=vmin, vmax=vmax)
    elif isinstance(data, FvdbTensor):
        coords = data.grid.ijk.jdata.detach().cpu().numpy()
        values = _feature_values(data.data.jdata)
        if coords.shape[0] == 0:
            return
        sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=values, cmap="viridis", vmin=vmin, vmax=vmax)
    elif isinstance(data, tuple) and len(data) == 2:
        coords, values = data
        if not isinstance(coords, np.ndarray):
            coords = np.array(coords)
        if not isinstance(values, np.ndarray):
            values = np.array(values)
        if coords.shape[1] >= 4:
            spatial = coords[:, 1:4]
        else:
            spatial = coords[:, :3]
        sc = ax.scatter(spatial[:, 0], spatial[:, 1], spatial[:, 2], c=values, cmap="viridis", vmin=vmin, vmax=vmax)
    else:
        raise TypeError(
            "Data must be a torch.Tensor, FvdbTensor, or a tuple (coords, values)"
        )

    ax.set_title(title)
    ax.set_xlabel("H axis")
    ax.set_ylabel("W axis")
    ax.set_zlabel("Z axis")
    return sc


def plot_3d_tensors(dense_items, global_vmin, global_vmax, suptitle="Dense Results") -> plt.Figure:
    num_items = len(dense_items)
    cols = 3
    rows = math.ceil(num_items / cols)
    fig = plt.figure(figsize=(5 * cols, 5 * rows))
    fig.tight_layout()
    axes = []
    for idx, (tensor, title) in enumerate(dense_items, start=1):
        ax = fig.add_subplot(rows, cols, idx, projection="3d")
        scatter_3d(ax, tensor, title, global_vmin, global_vmax)
        axes.append(ax)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=global_vmin, vmax=global_vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.05)
    fig.suptitle(suptitle)
    return fig
