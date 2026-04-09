import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any, Literal

try:
    import fvdb
except Exception:
    fvdb = None


@dataclass
class FvdbTensor:
    grid: Any
    data: Any

    def replace_data(self, new_data):
        # The local wrapper keeps topology (`grid`) and features (`data`)
        # together, but most feature-space ops only replace `data`.
        if fvdb is not None and isinstance(new_data, fvdb.JaggedTensor):
            return FvdbTensor(self.grid, new_data)
        if isinstance(new_data, torch.Tensor):
            return FvdbTensor(self.grid, self.grid.jagged_like(new_data.contiguous()))
        return FvdbTensor(self.grid, new_data)


def blur_tensor(input_tensor, epsilon=1e-10, noise_rate=0.3, sparse=False, backend: Literal["torchnn", "fvdb"] = "fvdb"):
    """
    For each active voxel in a binary tensor of shape (B, C, H, W, Z),
    add a small value to its 3x3x3 neighborhood excluding itself.
    """
    kernel = torch.ones((1, 1, 3, 3, 3), device=input_tensor.device)
    kernel[0, 0, 1, 1, 1] = 0.0

    B, C, H, W, Z = input_tensor.shape
    x = input_tensor.view(B * C, 1, H, W, Z)

    result = F.conv3d(x, kernel, padding=1)

    if 1 >= noise_rate > 0:
        random_mask = torch.rand_like(result) < noise_rate
        result[random_mask] += epsilon

    noise_mask = (result > 0).float().view(B, C, H, W, Z)
    output = input_tensor + (1 - input_tensor) * noise_mask * epsilon

    if sparse:
        output = to_sparse(output, backend=backend)

    return output


def calculate_sparsity(sparse_tensor: FvdbTensor, original_size=256):
    """
    Compute effective sparsity ratio for the local fvdb sparse tensor wrapper.
    """
    if not isinstance(sparse_tensor, FvdbTensor):
        raise TypeError(f"Unsupported tensor type for calculate_sparsity: {type(sparse_tensor)}")

    batch_size = int(sparse_tensor.grid.grid_count) if hasattr(sparse_tensor.grid, "grid_count") else 1
    total_voxels = batch_size * (original_size ** 3)
    active_voxels = int(sparse_tensor.data.jdata.shape[0])
    return 1 - (active_voxels / total_voxels)


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
    return torch.zeros(shape).to(input.device).scatter_(1, input, 1)


def flatten(tensor):
    """(N, C, D, H, W) -> (C, N * D * H * W)"""
    C = tensor.size(1)
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    transposed = tensor.permute(axis_order)
    return transposed.contiguous().view(C, -1)


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

    # OA-CNN operates on explicit point/voxel lists, so we drop zero voxels and
    # keep only active coordinates plus their per-voxel features.
    counts = torch.bincount(flat_batch, minlength=B)
    offset = torch.cumsum(counts, dim=0).long()

    return {
        "grid_coord": flat_coords,
        "feat": flat_feats,
        "offset": offset,
    }


def dense_to_fvdb(dense_tensor: torch.Tensor, voxel_sizes=(1.0, 1.0, 1.0)):
    """
    Converts dense [B, C, H, W, Z] into a sparse fvdb representation.
    """
    if fvdb is None:
        raise ImportError("fvdb is not installed but dense_to_fvdb was called.")

    B, C, H, W, Z = dense_tensor.shape
    device = dense_tensor.device

    active_mask = torch.any(dense_tensor != 0, dim=1)

    ijk_list = []
    for b in range(B):
        coords = torch.nonzero(active_mask[b], as_tuple=False).to(
            device=device, dtype=torch.int32
        ).contiguous()
        if coords.numel() == 0:
            coords = torch.zeros((0, 3), device=device, dtype=torch.int32).contiguous()
        ijk_list.append(coords)

    ijk = fvdb.JaggedTensor.from_list_of_tensors(ijk_list)
    # Build the sparse topology first from active voxel coordinates, then inject
    # the dense features onto that topology.
    grid = fvdb.GridBatch.from_ijk(
        ijk=ijk,
        voxel_sizes=voxel_sizes,
        origins=[0.0, 0.0, 0.0],
        device=device,
    )
    data = grid.inject_from_dense_cmajor(dense_tensor)
    return FvdbTensor(grid=grid, data=data)


def to_sparse(
    tensor: torch.Tensor,
    backend: Literal["torchnn", "fvdb"] = "fvdb",
    dtype=None,
    coords=None,
    voxel_sizes=(1.0, 1.0, 1.0),
) -> Any:
    """
    Prepare tensor for backend:
      - torchnn: dense torch.Tensor
      - fvdb: local FvdbTensor wrapper
    """
    del coords
    if dtype is not None:
        tensor = tensor.to(dtype)

    if backend == "torchnn":
        return tensor
    if backend == "fvdb":
        return dense_to_fvdb(tensor, voxel_sizes=voxel_sizes)
    raise ValueError(f"Unsupported backend: {backend}")


def to_dense(tensor, shape: torch.Size = None) -> torch.Tensor:
    """
    Convert backend tensor to dense torch.Tensor.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor

    if isinstance(tensor, FvdbTensor):
        min_coord = [0, 0, 0] if shape is not None else None
        grid_size = list(shape[2:]) if shape is not None else None
        # fvdb densification needs an explicit dense output box when we want the
        # result to match a known [B, C, H, W, Z] tensor shape.
        dense = tensor.grid.inject_to_dense_cmajor(
            tensor.data,
            min_coord=min_coord,
            grid_size=grid_size,
        )
        if shape is not None and dense.shape != shape:
            raise ValueError(f"fvdb to_dense produced {dense.shape}, expected {shape}")
        return dense

    raise TypeError(f"Unsupported tensor type: {type(tensor)}")


def to_dtype(tensor, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(tensor, torch.Tensor):
        return tensor.to(dtype)
    if isinstance(tensor, FvdbTensor):
        return tensor.replace_data(tensor.data.to(dtype))
    raise TypeError(f"Unsupported tensor type: {type(tensor)}")


def requires_grad(tensor, requires_grad: bool = True):
    if isinstance(tensor, torch.Tensor):
        tensor.requires_grad_(requires_grad)
        return tensor
    if isinstance(tensor, FvdbTensor):
        tensor.data.requires_grad_(requires_grad)
        return tensor
    raise TypeError(f"Unsupported tensor type: {type(tensor)}")


def get_dtype(tensor) -> torch.dtype:
    if isinstance(tensor, torch.Tensor):
        return tensor.dtype
    if isinstance(tensor, FvdbTensor):
        return tensor.data.dtype
    raise TypeError(f"Unsupported tensor type: {type(tensor)}")
