import math
import numpy as np
import torch
import spconv.pytorch as spconv
import seaborn as sns
import matplotlib.pyplot as plt


def plot_hist(data: np.ndarray | torch.Tensor):
    if isinstance(data, torch.Tensor):
        data = torch.sigmoid(data.reshape(-1)).detach().cpu().numpy()
    elif isinstance(data, spconv.SparseConvTensor):
        data = torch.sigmoid(data.features.reshape(-1)).detach().cpu().numpy()

    # randomly sample 10000 data points
    data = np.random.choice(data, 10000)

    # Plot 2D tensor as boxplot
    plt.figure(figsize=(8, 5))

    # Histogram with KDE (Kernel Density Estimate)
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

    For dense data (torch.Tensor), it assumes the tensor represents a 3D grid 
    (optionally with extra batch/channel dimensions) and only nonzero voxels are plotted.

    For sparse data, the function accepts either:
      - A spconv SparseConvTensor, or
      - A tuple (coords, values), where coords is a numpy array with spatial coordinates 
        (using columns 1:4 if available) and values is an array of feature values.
    """
    # Dense case: torch.Tensor (dense volume)
    if isinstance(data, torch.Tensor):
        # Remove extra dimensions until we have a 3D array.
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
        sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                        c=values, cmap='viridis', vmin=vmin, vmax=vmax)
    # Sparse case: tuple (coords, values)
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
        sc = ax.scatter(spatial[:, 0], spatial[:, 1], spatial[:, 2],
                        c=values, cmap='viridis', vmin=vmin, vmax=vmax)
    else:
        raise TypeError(
            "Data must be a torch.Tensor, spconv SparseConvTensor, or a tuple (coords, values)")

    ax.set_title(title)
    ax.set_xlabel('H axis')
    ax.set_ylabel('W axis')
    ax.set_zlabel('Z axis')
    return sc


def plot_3d_tensors(dense_items, global_vmin, global_vmax, suptitle="Dense Results") -> plt.Figure:
    """
    Plots a list of dense tensor items.

    Args:
        dense_items (list of tuples): Each tuple is (dense_tensor, title). 
           The dense_tensor can have shape (B, C, H, W, Z) or (C, H, W, Z).
        global_vmin (float): Minimum value for color normalization.
        global_vmax (float): Maximum value for color normalization.
        suptitle (str): Overall title for the figure.
    Returns:
        fig: The matplotlib figure.
    """
    num_items = len(dense_items)
    cols = 3
    rows = math.ceil(num_items / cols)
    fig = plt.figure(figsize=(5 * cols, 5 * rows))
    fig.tight_layout()
    axes = []
    for idx, (tensor, title) in enumerate(dense_items, start=1):
        ax = fig.add_subplot(rows, cols, idx, projection='3d')
        scatter_3d(ax, tensor, title, global_vmin, global_vmax)
        axes.append(ax)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(
        vmin=global_vmin, vmax=global_vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.05)
    fig.suptitle(suptitle)
    return fig
