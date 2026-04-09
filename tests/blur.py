import setup_paths  # noqa: F401
import torch
from matplotlib import pyplot as plt

from utils.vis import plot_3d_tensors
from utils.tensor import blur_tensor, to_sparse


def test():
    B, C, H, W, Z = 1, 1, 5, 5, 5
    tensor = torch.zeros((B, C, H, W, Z))
    tensor[0, 0, 2, 2, 2] = 1.0
    tensor[0, 0, 0, 0, 0] = 1.0

    # Blur adds tiny non-zero support around active voxels; this is useful for
    # visual sanity checks of sparse neighborhood augmentation.
    output = blur_tensor(tensor)

    print(f"Original tensor sum: {tensor.sum().item()}")
    print(f"Output tensor sum: {output.sum().item()}")
    print(f"Difference: {output.sum().item() - tensor.sum().item()}")

    affected_cells = ((output > 0) & (output < 1)).sum().item()
    print(f"Number of cells affected: {affected_cells}")

    fig = plot_3d_tensors([
        (tensor, "Original Tensor"),
        (output, "Tensor with Added Epsilon"),
        (to_sparse(output, "fvdb"), "fvdb Tensor"),
    ], 0, 1)
    fig.show()
    plt.show()


if __name__ == "__main__":
    test()
