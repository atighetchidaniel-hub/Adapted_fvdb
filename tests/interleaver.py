import setup_paths  # noqa: F401
import torch
import spconv.pytorch as spconv
import matplotlib.pyplot as plt
from modules.interleaver import Interleaver, Deinterleaver
from utils.vis import plot_3d_tensors
from utils.tensor import dense_to_spconv


def test():
    torch.cuda.manual_seed(0)
    # --------------------------
    # Create the starting dense tensor.
    # --------------------------
    B, C, H, W, Z = 1, 1, 4, 4, 4  # small grid for visualization
    total_elements = H * W * Z
    # Create a sequential dense tensor with values 0, 1, 2, ..., total_elements-1.
    x_full = torch.arange(
        total_elements, dtype=torch.float32).view(B, C, H, W, Z).to('cuda')
    # Create a random dropout mask (50% chance to keep each voxel).
    mask = (torch.rand(H, W, Z) < 0.5).float().to('cuda')
    # Expand mask to match (B, C, H, W, Z) and apply it.
    x_dense = x_full * mask.unsqueeze(0).unsqueeze(0)

    # Convert the dense tensor to a sparse tensor using spconv.
    x_sparse = dense_to_spconv(x_dense)
    coords_sparse = x_sparse.indices  # shape (N, 4): [batch, h, w, z]
    feats_sparse = x_sparse.features   # shape (N, C)

    # Define global vmin and vmax for shared color mapping.
    global_vmin = x_full.min().item()
    global_vmax = x_full.max().item()

    r = 2  # Down/upscaling factor (must divide H,W,Z)
    interleaver = Interleaver(r).to('cuda')
    deinterleaver = Deinterleaver(r, True).to('cuda')

    # --------------------------
    # Dense Implementation Test
    # --------------------------
    y_dense = interleaver(x_dense)
    x_recon_dense = deinterleaver(y_dense)

    assert torch.allclose(
        x_dense, x_recon_dense), "Interleaved dense tensors do not match"

    print("Dense Implementation:")
    print("  Original dense input shape:", x_dense.shape)
    print("  Dense interleaved shape:", y_dense.shape)
    print("  Dense reconstructed shape:", x_recon_dense.shape)
    print("  Max difference (dense):", torch.abs(
        x_dense - x_recon_dense).max().item())

    # Prepare list of dense items to plot.
    dense_items = []
    dense_items.append((x_dense, "Dense Original"))
    dense_items.append((x_recon_dense, "Dense Reconstructed"))
    # For the interleaved dense tensor, plot each channel separately.
    num_interleaved_channels_dense = y_dense.shape[1]
    for i in range(num_interleaved_channels_dense):
        # y_dense[0, i, :, :, :] has shape (H//r, W//r, Z//r)
        dense_items.append(
            (y_dense[0, i, :, :, :], f"Dense Interleaved Ch {i}"))

    fig_dense = plot_3d_tensors(
        dense_items, global_vmin, global_vmax, "Dense Implementation")
    fig_dense.show()

    # --------------------------
    # Sparse Implementation Test
    # --------------------------
    x_sparse = spconv.SparseConvTensor(feats_sparse, coords_sparse, [H, W, Z], B)
    y_sparse = interleaver(x_sparse)
    x_recon_sparse = deinterleaver(y_sparse)

    # For spconv, use .dense() to convert to dense tensor for comparison
    assert torch.allclose(y_sparse.dense(), y_dense), "Interleaved sparse and dense tensors do not match"
    assert torch.allclose(x_recon_sparse.dense(), x_recon_dense), "Deinterleaved sparse and dense tensors do not match"

    print("Sparse Implementation:")
    print("  Original sparse input shape:", x_sparse.F.shape)
    print("  Sparse interleaved shape:", y_sparse.F.shape)
    print("  Sparse reconstructed shape:", x_recon_sparse.F.shape)
    print("  Max difference (sparse):", torch.abs(
        x_sparse.F - x_recon_sparse.F).max().item())

    # Convert the sparse tensors to dense tensors for visualization.
    x_dense = x_sparse.dense(x_full.shape)[0]
    y_dense = y_sparse.dense(torch.Size((B, C * r**3, H//r, W//r, Z//r)))[0]
    x_recon_dense = x_recon_sparse.dense(x_full.shape)[0]

    # Prepare list of dense items to plot.
    dense_items = []
    dense_items.append((x_dense, "Sparse Original"))
    dense_items.append((x_recon_dense, "Sparse Reconstructed"))
    # For the interleaved sparse tensor, plot each channel separately.
    num_interleaved_channels_sparse = y_dense.shape[1]
    for i in range(num_interleaved_channels_sparse):
        # y_dense[0, i, :, :, :] has shape (H//r, W//r, Z//r)
        dense_items.append(
            (y_dense[0, i, :, :, :], f"Sparse Interleaved Ch {i}"))

    fig_sparse = plot_3d_tensors(
        dense_items, global_vmin, global_vmax, "Sparse Implementation")
    fig_sparse.show()

    plt.show()

    print("Interleaver test passed.")
    return True


if __name__ == '__main__':
    test()
