import torch
import torch.nn.functional as F
import spconv.pytorch as spconv
from typing import Literal


def blur_tensor(input_tensor, epsilon=1e-10, noise_rate=0.3, sparse=False):
    """
    For each voxel in a binary tensor of shape (B, C, H, W, Z) that is 1,
    add a small value (epsilon) to its 3x3x3 surrounding neighborhood (excluding itself).
    Only zero-valued voxels receive the added epsilon, and each voxel receives epsilon
    at most once, regardless of how many active neighbors it has.

    Args:
        input_tensor (torch.Tensor): Binary input tensor of shape (B, C, H, W, Z).
        epsilon (float): The small value to add, default reduced to 1e-5.

    Returns:
        torch.Tensor: The updated tensor.
    """
    # Create a 3D convolution kernel: ones in a 3x3x3 block except center is 0.
    kernel = torch.ones((1, 1, 3, 3, 3), device=input_tensor.device)
    kernel[0, 0, 1, 1, 1] = 0.0  # Exclude the center voxel

    # Reshape input so that we apply the same kernel to each channel.
    B, C, H, W, Z = input_tensor.shape
    x = input_tensor.view(B * C, 1, H, W, Z)

    # Apply 3D convolution with padding=1 to keep the same spatial dimensions.
    # This gives us counts of active neighbors for each voxel
    result = F.conv3d(x, kernel, padding=1)

    if 1 >= noise_rate > 0:
        # Randomly set extra noise to the input tensor
        random_mask = torch.rand_like(result) < noise_rate
        result[random_mask] += epsilon

    # Create a binary mask where a voxel is 1 if it has at least one active neighbor
    # This ensures we only add epsilon once per voxel regardless of neighbor count
    noise_mask = (result > 0).float()

    # Reshape the mask back to (B, C, H, W, Z)
    noise_mask = noise_mask.view(B, C, H, W, Z)

    # Only add epsilon to voxels that are originally zero AND have at least one active neighbor
    # This ensures each qualifying voxel gets exactly one epsilon added
    output = input_tensor + (1 - input_tensor) * noise_mask * epsilon

    if sparse:
        # Convert to spconv sparse tensor
        output = dense_to_spconv(output)

    return output


def calculate_sparsity(sparse_tensor, original_size=256):
    """
    Compute effective sparsity ratio considering tensor stride
    Args:
        sparse_tensor: spconv.SparseConvTensor
        original_size: Initial spatial dimension (assuming cube)
    Returns:
        Sparsity ratio (0 = dense, 1 = completely empty)
    """
    if isinstance(sparse_tensor, spconv.SparseConvTensor):
        # Get current stride information (spconv doesn't have tensor_stride)
        # For spconv, assume stride of 1 unless specified otherwise
        stride = 1

        # Calculate current effective spatial size
        current_size = original_size // stride

        # Total possible coordinates at this resolution
        total_voxels = current_size ** 3

        # Active coordinates count
        active_voxels = len(sparse_tensor.indices)

        return 1 - (active_voxels / total_voxels)
    else:
        raise TypeError(f"Unsupported tensor type: {type(sparse_tensor)}")


def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxDxHxW label image to NxCxDxHxW, where each label gets converted to its corresponding one-hot vector
    :param input: 4D input image (NxDxHxW)
    :param C: number of channels/labels
    :param ignore_index: ignore index to be kept during the expansion
    :return: 5D output image (NxCxDxHxW)
    """
    if input.dim() == 5:
        return input
    assert input.dim() == 4

    # expand the input tensor to Nx1xDxHxW before scattering
    input = input.unsqueeze(1)
    # create result tensor shape (NxCxDxHxW)
    shape = list(input.size())
    shape[1] = C

    if ignore_index is not None:
        # create ignore_index mask for the result
        mask = input.expand(shape) == ignore_index
        # clone the lib tensor and zero out ignore_index in the input
        input = input.clone()
        input[input == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input.device).scatter_(1, input, 1)


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


def dense_to_spconv(dense_tensor: torch.Tensor, coords: torch.Tensor | None = None) -> spconv.SparseConvTensor:
    """
    Converts a dense tensor of shape [B, C, H, W, Z] into a spconv.SparseConvTensor.

    It uses a simple non-zero criterion: a voxel (i.e. a spatial location) is kept if the sum
    of the absolute values over its channels is greater than zero.

    Args:
        dense_tensor (torch.Tensor): Dense input with shape [B, C, H, W, Z].

    Returns:
        spconv.SparseConvTensor: Sparse tensor whose features have shape [N, C] and
            indices have shape [N, 4]. The indices are structured as 
            [batch_index, h, w, z]. The spatial shape used is [H, W, Z] and the batch size is B.
    """
    B, C, H, W, Z = dense_tensor.shape

    if coords is None:
        coords = dense_to_coords(dense_tensor)

    # Reshape dense tensor to [B*H*W*Z, C]
    features_flat = dense_tensor.permute(0, 2, 3, 4, 1).reshape(-1, C)

    # Find non-zero elements (any channel != 0)
    mask = torch.any(features_flat != 0, dim=1)

    # Extract valid indices and features
    valid_indices = coords[mask].int()
    valid_features = features_flat[mask]

    # Create sparse tensor
    sparse_tensor = spconv.SparseConvTensor(
        valid_features, valid_indices, [H, W, Z], B
    )

    return sparse_tensor


def dense_to_coords(dense_tensor: torch.Tensor) -> torch.Tensor:
    B, C, H, W, Z = dense_tensor.shape
    device = dense_tensor.device

    # Create coordinates for all voxels in the tensor
    # This creates a tensor of shape [B*H*W*Z, 4]
    b_indices = torch.arange(B, device=device)
    h_indices = torch.arange(H, device=device)
    w_indices = torch.arange(W, device=device)
    z_indices = torch.arange(Z, device=device)

    # Using meshgrid to create coordinates
    b_grid, h_grid, w_grid, z_grid = torch.meshgrid(
        b_indices, h_indices, w_indices, z_indices, indexing='ij')
    coords = torch.stack([b_grid, h_grid, w_grid, z_grid],
                         dim=-1).reshape(-1, 4)

    return coords


def dense_to_oacnns_input(dense_input):
    """
    Convert a dense tensor of shape [B, C, H, W, Z] into the dictionary format required by the model.

    Args:
        dense_input (torch.Tensor): Dense input tensor with shape [B, C, H, W, Z].

    Returns:
        input_dict (dict): A dictionary with keys "grid_coord", "feat", and "offset".
    """
    B, C, H, W, Z = dense_input.shape
    device = dense_input.device

    # Create coordinate grids for the spatial dimensions.
    # Note: Adjust order as needed (here we assume the voxel coordinate is in the order [H, W, Z])
    h_coords = torch.arange(H, device=device)
    w_coords = torch.arange(W, device=device)
    z_coords = torch.arange(Z, device=device)
    # Use meshgrid to create coordinate grids (note that indexing='ij' preserves the order)
    grid_h, grid_w, grid_z = torch.meshgrid(
        h_coords, w_coords, z_coords, indexing="ij")

    # Expand the grid to include the batch dimension.
    # grid_coords will be of shape [B, H, W, Z, 3]
    grid_coords = torch.stack(
        [grid_h, grid_w, grid_z], dim=-1).unsqueeze(0).expand(B, -1, -1, -1, -1)

    # Create a batch index tensor of shape [B, H, W, Z]
    batch_idx = torch.arange(B, device=device).view(
        B, 1, 1, 1).expand(B, H, W, Z)

    # Flatten spatial dimensions.
    flat_coords = grid_coords.reshape(-1, 3)   # [B*H*W*Z, 3]
    flat_batch = batch_idx.reshape(-1)           # [B*H*W*Z]

    # Permute dense input to shape [B, H, W, Z, C] and then flatten.
    flat_feats = dense_input.permute(
        0, 2, 3, 4, 1).reshape(-1, C)  # [B*H*W*Z, C]

    # Optional: Filter out voxels with (almost) zero features (if sparsity is desired)
    # For instance, only include voxels with nonzero feature sum.
    valid_mask = flat_feats.abs().sum(dim=1) > 0
    flat_coords = flat_coords[valid_mask]
    flat_feats = flat_feats[valid_mask]
    flat_batch = flat_batch[valid_mask]

    # For the "offset", in your model the function "offset2batch" is used.
    # Here we assume that flat_batch correctly indicates the batch ID for each voxel.
    # This can be refined if needed based on your offset2batch logic.
    offset = flat_batch

    # Build the dictionary expected by the forward routine.
    input_dict = {
        "grid_coord": flat_coords,  # Expected shape: [N, 3]
        "feat": flat_feats,         # Expected shape: [N, C]
        "offset": offset,           # Expected shape: [N]
    }
    return input_dict


def to_sparse(tensor: torch.Tensor, backend: Literal["torchnn", "spconv"] = "torchnn", dtype=None, coords=None) -> torch.Tensor:
    """
    Prepares a tensor for the specified backend by converting it to the appropriate format.
    Args:
        tensor (torch.Tensor): The input tensor to be prepared.
        backend (str): The backend for which the tensor should be prepared. Options are "torchnn" or "spconv".
    Returns:
        torch.Tensor: The prepared tensor.
    """
    if dtype is not None:
        tensor = tensor.to(dtype)
    if backend == "torchnn":
        return tensor
    elif backend == "spconv":
        return dense_to_spconv(tensor, coords)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def to_dense(tensor: torch.Tensor, shape: torch.Size = None) -> torch.Tensor:
    """
    Converts a sparse tensor to a dense tensor based on the specified backend.
    Args:
        tensor (torch.Tensor): The input tensor to be converted.
        shape (torch.Size): The shape of the dense tensor to be created.
            If None, the shape is inferred from the input tensor.
    Returns:
        torch.Tensor: The converted dense tensor.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor
    elif isinstance(tensor, spconv.SparseConvTensor):
        return tensor.dense()
    else:
        raise TypeError(f"Unsupported tensor type: {type(tensor)}")


def to_dtype(tensor: torch.Tensor | spconv.SparseConvTensor, dtype: torch.dtype) -> torch.Tensor:
    """
    Convert the input tensor to the specified dtype.
    Args:
        tensor (torch.Tensor | spconv.SparseConvTensor): The input tensor.
        dtype (torch.dtype): The target dtype.
    Returns:
        torch.Tensor: The input tensor converted to the specified dtype.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.to(dtype)
    elif isinstance(tensor, spconv.SparseConvTensor):
        return tensor.replace_feature(tensor.features.to(dtype))
    else:
        raise TypeError(f"Unsupported tensor type: {type(tensor)}")


def requires_grad(tensor: torch.Tensor | spconv.SparseConvTensor, requires_grad: bool = True) -> torch.Tensor:
    """
    Set requires_grad for the input tensor.
    Args:
        tensor (torch.Tensor | spconv.SparseConvTensor): The input tensor.
        requires_grad (bool): Whether to set requires_grad to True or False.
    Returns:
        torch.Tensor: The input tensor with requires_grad set.
    """
    if isinstance(tensor, torch.Tensor):
        tensor.requires_grad_(requires_grad)
    elif isinstance(tensor, spconv.SparseConvTensor):
        tensor.features.requires_grad_(requires_grad)
    else:
        raise TypeError(f"Unsupported tensor type: {type(tensor)}")
    return tensor


def get_dtype(tensor: torch.Tensor | spconv.SparseConvTensor) -> torch.dtype:
    """
    Get the dtype of the input tensor.
    Args:
        tensor (torch.Tensor | spconv.SparseConvTensor): The input tensor.
    Returns:
        torch.dtype: The dtype of the input tensor.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.dtype
    elif isinstance(tensor, spconv.SparseConvTensor):
        return tensor.features.dtype
    else:
        raise TypeError(f"Unsupported tensor type: {type(tensor)}")
