import setup_paths  # noqa: F401
import torch
import spconv.pytorch as spconv
from matplotlib import pyplot as plt

from utils.vis import plot_3d_tensors
from utils.tensor import blur_tensor, dense_to_spconv


def test():
    """
    Create a sample binary tensor, apply the add_surrounding_small_value function,
    and visualize the original and updated tensors slice-by-slice along the Z dimension.
    """
    # Create a small tensor with shape (1, 1, 5, 5, 5)
    B, C, H, W, Z = 1, 1, 5, 5, 5
    tensor = torch.zeros((B, C, H, W, Z))

    # Place some 1's: one in the center and one in a corner.
    tensor[0, 0, 2, 2, 2] = 1.0
    tensor[0, 0, 0, 0, 0] = 1.0

    # Process the tensor to add epsilon to the surrounding zero cells.
    output = blur_tensor(tensor)

    # Print some values to verify the behavior
    print(f"Original tensor sum: {tensor.sum().item()}")
    print(f"Output tensor sum: {output.sum().item()}")
    print(
        f"Difference (should be epsilon * number of affected cells): {output.sum().item() - tensor.sum().item()}")

    # Count how many cells received epsilon
    affected_cells = ((output > 0) & (output < 1)).sum().item()
    print(f"Number of cells affected: {affected_cells}")

    fig = plot_3d_tensors([
        (tensor, "Original Tensor"),
        (output, "Tensor with Added Epsilon"),
        (dense_to_spconv(output), "Sparse Tensor"),
    ], 0, 1)
    fig.show()
    plt.show()


if __name__ == "__main__":
    test()
