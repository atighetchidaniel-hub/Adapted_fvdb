import setup_paths  # noqa: F401
import torch
import matplotlib.pyplot as plt

from modules.interleaver import Interleaver, Deinterleaver
from utils.vis import plot_3d_tensors
from utils.tensor import to_sparse, to_dense


def test():
    torch.cuda.manual_seed(0)
    B, C, H, W, Z = 1, 1, 4, 4, 4
    total_elements = H * W * Z
    x_full = torch.arange(total_elements, dtype=torch.float32).view(B, C, H, W, Z).to("cuda")
    mask = (torch.rand(H, W, Z, device="cuda") < 0.5).float()
    x_dense = x_full * mask.unsqueeze(0).unsqueeze(0)
    x_sparse = to_sparse(x_dense, "fvdb")

    global_vmin = x_full.min().item()
    global_vmax = x_full.max().item()

    r = 2
    interleaver = Interleaver(r).to("cuda")
    deinterleaver = Deinterleaver(r, True).to("cuda")

    # First verify that dense interleaving is perfectly invertible.
    y_dense = interleaver(x_dense)
    x_recon_dense = deinterleaver(y_dense)
    assert torch.allclose(x_dense, x_recon_dense), "Interleaved dense tensors do not match"

    # Then verify that the sparse fvdb path matches the dense reference path.
    y_sparse = interleaver(x_sparse)
    x_recon_sparse = deinterleaver(y_sparse)
    y_sparse_dense = to_dense(y_sparse, y_dense.shape)
    x_recon_sparse_dense = to_dense(x_recon_sparse, x_dense.shape)

    assert torch.allclose(y_sparse_dense, y_dense), "Interleaved sparse and dense tensors do not match"
    assert torch.allclose(x_recon_sparse_dense, x_recon_dense), "Deinterleaved sparse and dense tensors do not match"

    dense_items = [
        (x_dense, "Dense Original"),
        (x_recon_dense, "Dense Reconstructed"),
    ]
    for i in range(y_dense.shape[1]):
        dense_items.append((y_dense[0, i, :, :, :], f"Dense Interleaved Ch {i}"))

    fig_dense = plot_3d_tensors(dense_items, global_vmin, global_vmax, "Dense Implementation")
    fig_dense.show()

    sparse_items = [
        (x_dense, "Sparse Original"),
        (x_recon_sparse_dense, "Sparse Reconstructed"),
    ]
    for i in range(y_sparse_dense.shape[1]):
        sparse_items.append((y_sparse_dense[0, i, :, :, :], f"Sparse Interleaved Ch {i}"))

    fig_sparse = plot_3d_tensors(sparse_items, global_vmin, global_vmax, "fvdb Sparse Implementation")
    fig_sparse.show()

    plt.show()
    print("Interleaver test passed.")
    return True


if __name__ == "__main__":
    test()
