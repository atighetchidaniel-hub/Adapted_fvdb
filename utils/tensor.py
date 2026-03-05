# utils/tensor.py  (TEST-READY: spconv optional + fvdb optional + fvdb to_sparse added)
import torch
import torch.nn.functional as F
from typing import Literal, Any

# ---- optional spconv import (so fvdb-only env can import this file) ----
try:
    import spconv.pytorch as spconv
except Exception:
    spconv = None

# ---- optional fvdb import (so spconv-only env can import this file) ----
try:
    import fvdb.nn as fvnn
    from fvdb.nn import VDBTensor
except Exception:
    fvnn = None
    VDBTensor = None


def blur_tensor(input_tensor, epsilon=1e-10, noise_rate=0.3, sparse=False, backend: Literal["spconv", "fvdb"] = "spconv"):
    """
    For each voxel in a binary tensor of shape (B, C, H, W, Z) that is 1,
    add a small value (epsilon) to its 3x3x3 surrounding neighborhood (excluding itself).
    Only zero-valued voxels receive the added epsilon, and each voxel receives epsilon
    at most once, regardless of how many active neighbors it has.
    """
    kernel = torch.ones((1, 1, 3, 3, 3), device=input_tensor.device)
    kernel[0, 0, 1, 1, 1] = 0.0

    B, C, H, W, Z = input_tensor.shape
    x = input_tensor.view(B * C, 1, H, W, Z)

    result = F.conv3d(x, kernel, padding=1)

    if 1 >= noise_rate > 0:
        random_mask = torch.rand_like(result) < noise_rate
        result[random_mask] += epsilon

    noise_mask = (result > 0).float()
    noise_mask = noise_mask.view(B, C, H, W, Z)

    output = input_tensor + (1 - input_tensor) * noise_mask * epsilon

    if sparse:
        output = to_sparse(output, backend=backend)

    return output


def calculate_sparsity(sparse_tensor, original_size=256):
    """
    Compute effective sparsity ratio.
    Works for spconv tensors if spconv is installed.
    """
    if spconv is not None and isinstance(sparse_tensor, spconv.SparseConvTensor):
        stride = 1
        current_size = original_size // stride
        total_voxels = current_size ** 3
        active_voxels = len(sparse_tensor.indices)
        return 1 - (active_voxels / total_voxels)
    else:
        raise TypeError(f"Unsupported tensor type for calculate_sparsity: {type(sparse_tensor)}")


def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxDxHxW label image to NxCxDxHxW, where each label gets converted to its corresponding one-hot vector.
    """
    if input.dim() == 5:
        return input
    assert input.dim() == 4

    input = input.unsqueeze(1)
    shape = list(input.size())
    shape[1] = C

    if ignore_index is not None:
        mask = input.expand(shape) == ignore_index
        input = input.clone()
        input[input == ignore_index] = 0
        result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)
        result[mask] = ignore_index
        return result
    else:
        return torch.zeros(shape).to(input.device).scatter_(1, input, 1)


def flatten(tensor):
    """(N, C, D, H, W) -> (C, N * D * H * W)"""
    C = tensor.size(1)
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    transposed = tensor.permute(axis_order)
    return transposed.contiguous().view(C, -1)


def dense_to_spconv(dense_tensor: torch.Tensor, coords: torch.Tensor | None = None) -> Any:
    """
    Converts dense [B, C, H, W, Z] to spconv.SparseConvTensor.
    This function requires spconv to be installed.
    """
    if spconv is None:
        raise ImportError("spconv is not installed but dense_to_spconv was called.")

    B, C, H, W, Z = dense_tensor.shape

    if coords is None:
        coords = dense_to_coords(dense_tensor)

    features_flat = dense_tensor.permute(0, 2, 3, 4, 1).reshape(-1, C)
    mask = torch.any(features_flat != 0, dim=1)

    valid_indices = coords[mask].int()
    valid_features = features_flat[mask]

    sparse_tensor = spconv.SparseConvTensor(
        valid_features, valid_indices, [H, W, Z], B
    )
    return sparse_tensor


def dense_to_coords(dense_tensor: torch.Tensor) -> torch.Tensor:
    B, C, H, W, Z = dense_tensor.shape
    device = dense_tensor.device

    b_indices = torch.arange(B, device=device)
    h_indices = torch.arange(H, device=device)
    w_indices = torch.arange(W, device=device)
    z_indices = torch.arange(Z, device=device)

    b_grid, h_grid, w_grid, z_grid = torch.meshgrid(
        b_indices, h_indices, w_indices, z_indices, indexing='ij'
    )
    coords = torch.stack([b_grid, h_grid, w_grid, z_grid], dim=-1).reshape(-1, 4)
    return coords


def dense_to_oacnns_input(dense_input):
    """
    Convert dense [B, C, H, W, Z] into dict format {"grid_coord","feat","offset"}.
    """
    B, C, H, W, Z = dense_input.shape
    device = dense_input.device

    h_coords = torch.arange(H, device=device)
    w_coords = torch.arange(W, device=device)
    z_coords = torch.arange(Z, device=device)

    grid_h, grid_w, grid_z = torch.meshgrid(h_coords, w_coords, z_coords, indexing="ij")
    grid_coords = torch.stack([grid_h, grid_w, grid_z], dim=-1).unsqueeze(0).expand(B, -1, -1, -1, -1)

    batch_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, H, W, Z)

    flat_coords = grid_coords.reshape(-1, 3)
    flat_batch = batch_idx.reshape(-1)

    flat_feats = dense_input.permute(0, 2, 3, 4, 1).reshape(-1, C)

    valid_mask = flat_feats.abs().sum(dim=1) > 0
    flat_coords = flat_coords[valid_mask]
    flat_feats = flat_feats[valid_mask]
    flat_batch = flat_batch[valid_mask]

    offset = flat_batch

    return {
        "grid_coord": flat_coords,
        "feat": flat_feats,
        "offset": offset,
    }


def dense_to_fvdb(dense_tensor: torch.Tensor, voxel_sizes=(1.0, 1.0, 1.0)):
    """
    Converts dense [B, C, H, W, Z] into fvdb VDBTensor.

    fvdb commonly expects channels-last dense: [B, D, H, W, C]
    We map Z -> D:
      [B, C, H, W, Z] -> [B, Z, H, W, C]
    """
    if fvnn is None:
        raise ImportError("fvdb is not installed but dense_to_fvdb was called.")
    dense_ch_last = dense_tensor.permute(0, 4, 2, 3, 1).contiguous()  # [B, Z, H, W, C]
    return fvnn.vdbtensor_from_dense(dense_ch_last, voxel_sizes=list(voxel_sizes))


def to_sparse(
    tensor: torch.Tensor,
    backend: Literal["torchnn", "spconv", "fvdb"] = "torchnn",
    dtype=None,
    coords=None,
    voxel_sizes=(1.0, 1.0, 1.0),
) -> torch.Tensor:
    """
    Prepare tensor for backend:
      - torchnn: dense torch.Tensor
      - spconv: spconv.SparseConvTensor
      - fvdb: fvdb.nn.VDBTensor
    """
    if dtype is not None:
        tensor = tensor.to(dtype)

    if backend == "torchnn":
        return tensor
    elif backend == "spconv":
        return dense_to_spconv(tensor, coords)
    elif backend == "fvdb":
        return dense_to_fvdb(tensor, voxel_sizes=voxel_sizes)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def to_dense(tensor, shape: torch.Size = None) -> torch.Tensor:
    """
    Convert backend tensor to dense torch.Tensor.
      - spconv: .dense()
      - fvdb: .to_dense() then permute back to [B, C, H, W, Z]
    """
    if isinstance(tensor, torch.Tensor):
        return tensor

    if spconv is not None and isinstance(tensor, spconv.SparseConvTensor):
        return tensor.dense()

    if VDBTensor is not None and isinstance(tensor, VDBTensor):
        dense_ch_last = tensor.to_dense()                 # [B, D, H, W, C]
        dense = dense_ch_last.permute(0, 4, 2, 3, 1).contiguous()  # [B, C, H, W, Z]
        if shape is not None and dense.shape != shape:
            raise ValueError(f"fvdb to_dense produced {dense.shape}, expected {shape}")
        return dense

    raise TypeError(f"Unsupported tensor type: {type(tensor)}")


def to_dtype(tensor, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(tensor, torch.Tensor):
        return tensor.to(dtype)

    if spconv is not None and isinstance(tensor, spconv.SparseConvTensor):
        return tensor.replace_feature(tensor.features.to(dtype))

    if VDBTensor is not None and isinstance(tensor, VDBTensor):
        # Rebuild VDBTensor with cast jagged data
        return VDBTensor(tensor.grid, tensor.data.to(dtype))

    raise TypeError(f"Unsupported tensor type: {type(tensor)}")


def requires_grad(tensor, requires_grad: bool = True):
    if isinstance(tensor, torch.Tensor):
        tensor.requires_grad_(requires_grad)
        return tensor

    if spconv is not None and isinstance(tensor, spconv.SparseConvTensor):
        tensor.features.requires_grad_(requires_grad)
        return tensor

    if VDBTensor is not None and isinstance(tensor, VDBTensor):
        tensor.data.requires_grad_(requires_grad)
        return tensor

    raise TypeError(f"Unsupported tensor type: {type(tensor)}")


def get_dtype(tensor) -> torch.dtype:
    if isinstance(tensor, torch.Tensor):
        return tensor.dtype

    if spconv is not None and isinstance(tensor, spconv.SparseConvTensor):
        return tensor.features.dtype

    if VDBTensor is not None and isinstance(tensor, VDBTensor):
        return tensor.data.dtype

    raise TypeError(f"Unsupported tensor type: {type(tensor)}")