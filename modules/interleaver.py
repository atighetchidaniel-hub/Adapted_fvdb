import torch
import torch.nn as nn

try:
    import fvdb
except Exception:
    fvdb = None

from utils.tensor import FvdbTensor


def interleaving_fn(x: torch.Tensor, r: int) -> torch.Tensor:
    """
    Rearranges a 5D tensor of shape (B, C, H, W, Z) into
    (B, C * r^3, H//r, W//r, Z//r).
    """
    B, C, H, W, Z = x.shape
    assert H % r == 0 and W % r == 0 and Z % r == 0, "Spatial dimensions H, W, and Z must be divisible by r"
    x = x.view(B, C, H // r, r, W // r, r, Z // r, r)
    x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
    return x.view(B, C * (r ** 3), H // r, W // r, Z // r)


def deinterleaving_fn(x: torch.Tensor, r: int) -> torch.Tensor:
    """
    Reverses the interleaving operation.
    """
    B, Cr3, H, W, Z = x.shape
    assert Cr3 % (r ** 3) == 0, "Channel dimension must be divisible by r^3"
    C = Cr3 // (r ** 3)
    x = x.view(B, C, r, r, r, H, W, Z)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    return x.view(B, C, H * r, W * r, Z * r)


def _fvdb_split_lists(x: FvdbTensor):
    ijk = x.grid.ijk
    offsets = ijk.joffsets.long()
    coords_list = []
    feats_list = []
    # fvdb batches store all active voxels in jagged form; split them back into
    # one coordinate/feature list per sample for easier coordinate transforms.
    for start, end in zip(offsets[:-1], offsets[1:]):
        s = int(start.item())
        e = int(end.item())
        coords_list.append(ijk.jdata[s:e].to(dtype=torch.int32).contiguous())
        feats_list.append(x.data.jdata[s:e].contiguous())
    return coords_list, feats_list


def _fvdb_tensor_from_lists(coord_list, feat_list, voxel_sizes, origins, device):
    if fvdb is None:
        raise ImportError("fvdb is not installed but FvdbTensor construction was requested.")

    coord_list = [coords.to(device=device, dtype=torch.int32).contiguous() for coords in coord_list]
    feat_list = [feats.to(device=device).contiguous() for feats in feat_list]

    ijk = fvdb.JaggedTensor.from_list_of_tensors(coord_list)
    grid = fvdb.GridBatch.from_ijk(
        ijk=ijk,
        voxel_sizes=voxel_sizes,
        origins=origins,
        device=device,
    )
    feat_jagged = fvdb.JaggedTensor.from_list_of_tensors(feat_list)
    # Re-inject the explicit feature lists onto the newly constructed topology.
    data = grid.inject_from_ijk(ijk, feat_jagged)
    return FvdbTensor(grid=grid, data=data)


def sparse_interleaving_fn(x: FvdbTensor, r: int, use_cuda: bool = False):
    """
    Rearranges an FvdbTensor into an interleaved sparse representation.
    """
    del use_cuda
    if not isinstance(x, FvdbTensor):
        raise TypeError("Input must be an FvdbTensor.")

    r2 = r * r
    r3 = r2 * r
    coord_list, feat_list = _fvdb_split_lists(x)
    new_coord_list = []
    new_feat_list = []
    device = x.data.jdata.device

    for coords, feats in zip(coord_list, feat_list):
        if coords.numel() == 0:
            C = feats.shape[1]
            new_coord_list.append(coords.new_zeros((0, 3), dtype=torch.int32))
            new_feat_list.append(feats.new_zeros((0, C * r3)))
            continue

        C = feats.shape[1]
        base_coords = coords // r
        offsets = coords % r
        offset_idx = offsets[:, 0] * r2 + offsets[:, 1] * r + offsets[:, 2]
        unique_coords, inv_idx = torch.unique(base_coords, dim=0, return_inverse=True)
        aggregated_feats = torch.zeros((unique_coords.size(0), C * r3), device=device, dtype=feats.dtype)

        # Pack the r^3 spatial offsets into channels so each coarse voxel stores
        # the fine sub-voxel pattern in its feature vector.
        offset_idx = offset_idx.unsqueeze(1)
        ch_range = torch.arange(C, device=device).unsqueeze(0)
        target_idx = offset_idx * C + ch_range
        row_idx = inv_idx.unsqueeze(1).expand(-1, C)
        flat_idx = row_idx * (C * r3) + target_idx
        aggregated_feats.view(-1).scatter_(0, flat_idx.reshape(-1), feats.reshape(-1))

        new_coord_list.append(unique_coords.to(dtype=torch.int32).contiguous())
        new_feat_list.append(aggregated_feats.contiguous())

    voxel_sizes = getattr(x.grid, "voxel_sizes", 1.0)
    origins = getattr(x.grid, "origins", 0.0)
    return _fvdb_tensor_from_lists(new_coord_list, new_feat_list, voxel_sizes * r, origins, device)


def sparse_deinterleaving_fn(x: FvdbTensor, r: int, prune_zeros: bool = False):
    """
    Reverses sparse interleaving for FvdbTensor inputs.
    """
    if not isinstance(x, FvdbTensor):
        raise TypeError("Input must be an FvdbTensor.")

    coord_list, feat_list = _fvdb_split_lists(x)
    new_coord_list = []
    new_feat_list = []
    device = x.data.jdata.device
    total = r ** 3
    indices = torch.arange(total, device=device)
    offset_x = indices // (r * r)
    offset_y = (indices % (r * r)) // r
    offset_z = indices % r
    offsets = torch.stack([offset_x, offset_y, offset_z], dim=1).to(torch.int32)

    for coords, aggregated_feats in zip(coord_list, feat_list):
        if coords.numel() == 0:
            channels = aggregated_feats.shape[1] // total if aggregated_feats.shape[1] else 0
            new_coord_list.append(coords.new_zeros((0, 3), dtype=torch.int32))
            new_feat_list.append(aggregated_feats.new_zeros((0, channels)))
            continue

        M = coords.shape[0]
        C = aggregated_feats.shape[1] // total
        feats_reshaped = aggregated_feats.view(M, total, C)
        base_expanded = coords * r
        new_spatial = base_expanded.unsqueeze(1) + offsets.unsqueeze(0)
        new_coords_flat = new_spatial.reshape(-1, 3)
        new_feats_flat = feats_reshaped.reshape(-1, C)

        # Interleaving can create explicit zero entries for empty sub-voxels;
        # pruning drops them so the reconstructed sparse tensor stays compact.
        if prune_zeros:
            mask = new_feats_flat.abs().sum(dim=1) != 0
            new_feats_flat = new_feats_flat[mask]
            new_coords_flat = new_coords_flat[mask]

        new_coord_list.append(new_coords_flat.to(dtype=torch.int32).contiguous())
        new_feat_list.append(new_feats_flat.contiguous())

    voxel_sizes = getattr(x.grid, "voxel_sizes", 1.0)
    origins = getattr(x.grid, "origins", 0.0)
    return _fvdb_tensor_from_lists(new_coord_list, new_feat_list, voxel_sizes / r, origins, device)


class Interleaver(nn.Module):
    """
    Apply interleaving to either dense tensors or FvdbTensor inputs.
    """

    def __init__(self, r: int, use_cuda: bool = False):
        super().__init__()
        self.r = r
        self.use_cuda = use_cuda

    def forward(self, x):
        if isinstance(x, FvdbTensor):
            return sparse_interleaving_fn(x, self.r, self.use_cuda)
        if isinstance(x, torch.Tensor):
            return interleaving_fn(x, self.r)
        raise TypeError("Input must be either a torch.Tensor or an FvdbTensor.")


class Deinterleaver(nn.Module):
    """
    Apply deinterleaving to either dense tensors or FvdbTensor inputs.
    """

    def __init__(self, r: int, prune_zeros: bool = False, use_cuda: bool = False):
        super().__init__()
        self.r = r
        self.prune_zeros = prune_zeros
        self.use_cuda = use_cuda

    def forward(self, x):
        if isinstance(x, FvdbTensor):
            return sparse_deinterleaving_fn(x, self.r, self.prune_zeros)
        if isinstance(x, torch.Tensor):
            return deinterleaving_fn(x, self.r)
        raise TypeError("Input must be either a torch.Tensor or an FvdbTensor.")
