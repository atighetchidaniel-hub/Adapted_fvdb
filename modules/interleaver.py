import torch
import torch.nn as nn

try:
    import spconv.pytorch as spconv
except Exception:
    spconv = None

try:
    import fvdb
except Exception:
    fvdb = None

from utils.tensor import FvdbTensor
# --------------------------
# Dense Functions
# --------------------------


def interleaving_fn(x: torch.Tensor, r: int) -> torch.Tensor:
    """
    Rearranges a 5D tensor of shape (B, C, H, W, Z) into
    (B, C * r^3, H//r, W//r, Z//r) by splitting each spatial dimension
    into non-overlapping blocks of size r.
    """
    B, C, H, W, Z = x.shape
    assert H % r == 0 and W % r == 0 and Z % r == 0, \
        "Spatial dimensions H, W, and Z must be divisible by r"
    # Reshape: split each spatial dimension into (dim//r, r)
    x = x.view(B, C, H // r, r, W // r, r, Z // r, r)
    # Permute to bring the r factors into the channel dimension:
    x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
    # Flatten the r factors into the channel dimension:
    x = x.view(B, C * (r ** 3), H // r, W // r, Z // r)
    return x


def deinterleaving_fn(x: torch.Tensor, r: int) -> torch.Tensor:
    """
    Reverses the interleaving operation: rearranges a 5D tensor of shape 
    (B, C * r^3, H, W, Z) back into (B, C, H*r, W*r, Z*r).
    """
    B, Cr3, H, W, Z = x.shape
    assert Cr3 % (r ** 3) == 0, "Channel dimension must be divisible by r^3"
    C = Cr3 // (r ** 3)
    # Reshape to separate the r factors from channels:
    x = x.view(B, C, r, r, r, H, W, Z)
    # Permute to interleave the r factors with the spatial dimensions:
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    # Merge the r factors with the spatial dimensions:
    x = x.view(B, C, H * r, W * r, Z * r)
    return x


# # Define C++ source
# cpp_source = r"""
# #include <torch/extension.h>

# // Forward declarations for CUDA back-ends
# torch::Tensor interleaving_cuda(torch::Tensor x, int r);
# torch::Tensor deinterleaving_cuda(torch::Tensor x, int r);
# void sparse_interleave_cuda(
#     const at::Tensor& feats,
#     const at::Tensor& inv_idx,
#     const at::Tensor& offset_idx,
#     at::Tensor& out,
#     int C,
#     int r3);

# // Dense wrappers
# torch::Tensor interleaving(torch::Tensor x, int r) {
#     return interleaving_cuda(x, r);
# }

# torch::Tensor deinterleaving(torch::Tensor x, int r) {
#     return deinterleaving_cuda(x, r);
# }

# // Sparse wrapper
# torch::Tensor sparse_interleave(
#     torch::Tensor feats,
#     torch::Tensor inv_idx,
#     torch::Tensor offset_idx,
#     int M,
#     int C,
#     int r3) {
#     auto out = torch::zeros({M, C * r3}, feats.options());
#     sparse_interleave_cuda(feats, inv_idx, offset_idx, out, C, r3);
#     return out;
# }
# """

# # Define the CUDA source
# cuda_source = r"""
# #include <torch/extension.h>
# #include <ATen/Dispatch.h>
# #include <cuda_runtime.h>

# #define CUDA_CHECK()                                                  \
#   do {                                                                \
#     cudaError_t e1 = cudaGetLastError();                              \
#     cudaError_t e2 = cudaDeviceSynchronize();                         \
#     if (e1 != cudaSuccess || e2 != cudaSuccess)                       \
#       TORCH_CHECK(false, "CUDA error: ",                              \
#                   cudaGetErrorString(e1),                            \
#                   " / ",                                              \
#                   cudaGetErrorString(e2));                           \
#   } while (0)

# template <typename scalar_t>
# __global__ void interleave_kernel_1d(
#     const scalar_t* __restrict__ in,
#           scalar_t* __restrict__ out,
#     int B, int C, int H, int W, int Z, int r) {
#   int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
#   int64_t total = int64_t(B)*C*H*W*Z;
#   if (idx >= total) return;
#   int z = idx % Z;  int64_t tmp = idx / Z;
#   int w = tmp % W;  tmp /= W;
#   int h = tmp % H;  tmp /= H;
#   int c = tmp % C;  int b = tmp / C;
#   int hb = h/r, ho = h%r;
#   int wb = w/r, wo = w%r;
#   int zb = z/r, zo = z%r;
#   int channel_offset = ho*(r*r) + wo*r + zo;
#   int new_c = c*(r*r*r) + channel_offset;
#   int new_H = H/r, new_W = W/r, new_Z = Z/r;
#   int64_t out_idx = (((b*int64_t(C)*r*r*r + new_c)*new_H + hb)
#                     * new_W + wb)*new_Z + zb;
#   out[out_idx] = in[idx];
# }

# template <typename scalar_t>
# __global__ void deinterleave_kernel_1d(
#     const scalar_t* __restrict__ in,
#           scalar_t* __restrict__ out,
#     int B, int Cr3, int H, int W, int Z, int r) {
#   int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
#   int64_t total = int64_t(B)*Cr3*H*W*Z;
#   if (idx >= total) return;
#   int z = idx % Z;  int64_t tmp = idx / Z;
#   int w = tmp % W;  tmp /= W;
#   int h = tmp % H;  tmp /= H;
#   int cr3 = tmp % Cr3;  int b = tmp / Cr3;
#   int C = Cr3/(r*r*r);
#   int c = cr3/(r*r*r);
#   int off = cr3 % (r*r*r);
#   int ho = off/(r*r);
#   int wo = (off/r) % r;
#   int zo = off % r;
#   int orig_h = h*r + ho;
#   int orig_w = w*r + wo;
#   int orig_z = z*r + zo;
#   int64_t out_idx = ((((b*C + c)*H*r + orig_h)*W*r + orig_w)*Z*r + orig_z);
#   out[out_idx] = in[idx];
# }

# torch::Tensor interleaving_cuda(torch::Tensor x, int r) {
#   auto B = x.size(0), C = x.size(1), H = x.size(2),
#        W = x.size(3), Z = x.size(4);
#   auto out = torch::empty({B, C*r*r*r, H/r, W/r, Z/r}, x.options());
#   int64_t total = int64_t(B)*C*H*W*Z;
#   const int threads = 256;
#   const int blocks  = (total + threads - 1) / threads;

#   AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "interleave1d", [&] {
#     interleave_kernel_1d<scalar_t><<<blocks,threads>>>(
#       x.data_ptr<scalar_t>(),
#       out.data_ptr<scalar_t>(),
#       B, C, H, W, Z, r
#     );
#   });
#   CUDA_CHECK();
#   return out;
# }

# torch::Tensor deinterleaving_cuda(torch::Tensor x, int r) {
#   auto B   = x.size(0), Cr3 = x.size(1),
#        H   = x.size(2), W   = x.size(3), Z = x.size(4);
#   auto out = torch::empty({B, Cr3/(r*r*r), H*r, W*r, Z*r}, x.options());
#   int64_t total = int64_t(B)*Cr3*H*W*Z;
#   const int threads = 256;
#   const int blocks  = (total + threads - 1) / threads;

#   AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "deinterleave1d", [&] {
#     deinterleave_kernel_1d<scalar_t><<<blocks,threads>>>(
#       x.data_ptr<scalar_t>(),
#       out.data_ptr<scalar_t>(),
#       B, Cr3, H, W, Z, r
#     );
#   });
#   CUDA_CHECK();
#   return out;
# }

# #define DISPATCH_INDEX_TYPE(idx_scalar_type, NAME, ...)                 \
#   [&] {                                                                 \
#     switch (idx_scalar_type) {                                          \
#       case at::ScalarType::Int: {                                       \
#         using index_t = int32_t;                                        \
#         __VA_ARGS__                                                    \
#         break;                                                          \
#       }                                                                  \
#       case at::ScalarType::Long: {                                      \
#         using index_t = int64_t;                                        \
#         __VA_ARGS__                                                    \
#         break;                                                          \
#       }                                                                  \
#       default:                                                           \
#         TORCH_CHECK(false, NAME,                                       \
#                     " not implemented for index type ",                \
#                     toString(idx_scalar_type));                        \
#     }                                                                    \
#   }()

# template <typename scalar_t, typename index_t>
# __global__ void sparse_interleave_kernel(
#     const scalar_t* __restrict__ feats,
#     const index_t*  __restrict__ inv_idx,
#     const index_t*  __restrict__ offset_idx,
#     scalar_t*       __restrict__ out,
#     int             N,
#     int             C,
#     int             r3) {
#   int idx = blockIdx.x * blockDim.x + threadIdx.x;
#   if (idx >= N*C) return;
#   int n   = idx / C;
#   int c   = idx % C;
#   int row = static_cast<int>(inv_idx[n]);
#   int off = static_cast<int>(offset_idx[n]);
#   int dst = row * (C*r3) + off*C + c;
#   out[dst] = feats[n*C + c];
# }

# void sparse_interleave_cuda(
#     const at::Tensor& feats,
#     const at::Tensor& inv_idx,
#     const at::Tensor& offset_idx,
#     at::Tensor&       out,
#     int               C,
#     int               r3) {
#   int N = feats.size(0);
#   const int threads = 256;
#   const int blocks  = (N*C + threads - 1) / threads;

#   TORCH_CHECK(inv_idx.scalar_type() == offset_idx.scalar_type(),
#               "inv_idx and offset_idx must have the same dtype");

#   DISPATCH_INDEX_TYPE(inv_idx.scalar_type(), "sparse_interleave", {
#     AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half,
#                               feats.scalar_type(),
#                               "sparse_interleave",
#                               [&] {
#       sparse_interleave_kernel<scalar_t, index_t><<<blocks,threads>>>(
#         feats.data_ptr<scalar_t>(),
#         inv_idx.data_ptr<index_t>(),
#         offset_idx.data_ptr<index_t>(),
#         out.data_ptr<scalar_t>(),
#         N, C, r3
#       );
#     });
#     CUDA_CHECK();
#   });
# }
# """

# # Create the extension module
# interleave_ext = load_inline(
#     name="interleave_ext",
#     cpp_sources=cpp_source,
#     cuda_sources=cuda_source,
#     functions=["interleaving", "deinterleaving", "sparse_interleave"],
#     extra_cflags=['-DTORCH_USE_CUDA_DSA'],
#     extra_cuda_cflags=["-DTORCH_USE_CUDA_DSA", "-O2"],
#     verbose=True
# )


# def interleaving_fn_cuda(x: torch.Tensor, r: int) -> torch.Tensor:
#     """
#     Rearranges a 5D tensor of shape (B, C, H, W, Z) into (B, C * r^3, H//r, W//r, Z//r)
#     by splitting each spatial dimension into non-overlapping blocks of size r.
#     Implemented with direct CUDA kernel.

#     Args:
#         x: Input tensor of shape (B, C, H, W, Z)
#         r: Block size for rearrangement

#     Returns:
#         Rearranged tensor of shape (B, C * r^3, H//r, W//r, Z//r)
#     """
#     B, C, H, W, Z = x.shape
#     assert H % r == 0 and W % r == 0 and Z % r == 0, "Spatial dimensions H, W, and Z must be divisible by r"
#     assert x.is_cuda, "Input tensor must be on CUDA device"
#     assert x.is_contiguous(), "Input tensor must be contiguous"

#     return interleave_ext.interleaving(x, r)


# def deinterleaving_fn_cuda(x: torch.Tensor, r: int) -> torch.Tensor:
#     """
#     Reverses the interleaving operation: rearranges a 5D tensor of shape 
#     (B, C * r^3, H, W, Z) back into (B, C, H*r, W*r, Z*r).
#     Implemented with direct CUDA kernel.

#     Args:
#         x: Input tensor of shape (B, C * r^3, H, W, Z)
#         r: Block size used in the original interleaving

#     Returns:
#         Deinterleaved tensor of shape (B, C, H*r, W*r, Z*r)
#     """
#     B, Cr3, H, W, Z = x.shape
#     assert Cr3 % (r ** 3) == 0, "Channel dimension must be divisible by r^3"
#     assert x.is_cuda, "Input tensor must be on CUDA device"
#     assert x.is_contiguous(), "Input tensor must be contiguous"

#     return interleave_ext.deinterleaving(x, r)


def _fvdb_split_lists(x: FvdbTensor):
    ijk = x.grid.ijk
    offsets = ijk.joffsets.long()
    coords_list = []
    feats_list = []
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
    data = grid.inject_from_ijk(ijk, feat_jagged)
    return FvdbTensor(grid=grid, data=data)


def sparse_interleaving_fn(x, r: int, use_cuda: bool = False):
    """
    Rearranges a sparse tensor into an interleaved sparse representation.
    Supports both spconv SparseConvTensor and local FvdbTensor inputs.
    """
    r2 = r * r
    r3 = r2 * r

    if spconv is not None and isinstance(x, spconv.SparseConvTensor):
        coords = x.indices
        feats = x.features
        device = feats.device
        _, C = feats.shape

        base_coords = coords[:, 1:] // r
        new_coords = torch.cat([coords[:, :1], base_coords], dim=1)
        offsets = coords[:, 1:] % r
        offset_idx = offsets[:, 0] * r2 + offsets[:, 1] * r + offsets[:, 2]
        unique_coords, inv_idx = torch.unique(new_coords, dim=0, return_inverse=True)
        M = unique_coords.size(0)
        assert M > 0, "No unique coordinates found."

        if use_cuda:
            pass
        else:
            aggregated_feats = torch.zeros((M, C * r3), device=device, dtype=feats.dtype)
            offset_idx = offset_idx.unsqueeze(1)
            ch_range = torch.arange(C, device=device).unsqueeze(0)
            target_idx = offset_idx * C + ch_range
            row_idx = inv_idx.unsqueeze(1).expand(-1, C)
            flat_idx = row_idx * (C * r3) + target_idx
            aggregated_feats.view(-1).scatter_(0, flat_idx.reshape(-1), feats.reshape(-1))

        H, W, D = x.spatial_shape
        d0, d1, d2 = H // r, W // r, D // r
        return spconv.SparseConvTensor(
            aggregated_feats,
            unique_coords,
            spatial_shape=(d0, d1, d2),
            batch_size=x.batch_size,
        )

    if isinstance(x, FvdbTensor):
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

            offset_idx = offset_idx.unsqueeze(1)
            ch_range = torch.arange(C, device=device).unsqueeze(0)
            target_idx = offset_idx * C + ch_range
            row_idx = inv_idx.unsqueeze(1).expand(-1, C)
            flat_idx = row_idx * (C * r3) + target_idx
            aggregated_feats.view(-1).scatter_(0, flat_idx.reshape(-1), feats.reshape(-1))

            new_coord_list.append(unique_coords.to(dtype=torch.int32).contiguous())
            new_feat_list.append(aggregated_feats.contiguous())

        voxel_sizes = getattr(x.grid, 'voxel_sizes', 1.0)
        origins = getattr(x.grid, 'origins', 0.0)
        return _fvdb_tensor_from_lists(new_coord_list, new_feat_list, voxel_sizes * r, origins, device)

    raise TypeError("Input must be either a spconv.SparseConvTensor or an FvdbTensor.")


def sparse_deinterleaving_fn(x, r: int, prune_zeros: bool = False):
    """
    Reverses sparse interleaving for both spconv SparseConvTensor and FvdbTensor inputs.
    """
    if spconv is not None and isinstance(x, spconv.SparseConvTensor):
        coords = x.indices
        aggregated_feats = x.features
        device = aggregated_feats.device
        M = coords.shape[0]
        total = r ** 3
        C = aggregated_feats.shape[1] // total

        feats_reshaped = aggregated_feats.view(M, total, C)
        indices = torch.arange(total, device=device)
        offset_x = indices // (r * r)
        offset_y = (indices % (r * r)) // r
        offset_z = indices % r
        offsets = torch.stack([offset_x, offset_y, offset_z], dim=1).to(coords.dtype)
        base_coords = coords[:, 1:]
        base_expanded = base_coords * r
        new_spatial = base_expanded.unsqueeze(1) + offsets.unsqueeze(0)
        batch = coords[:, :1]
        batch_expanded = batch.unsqueeze(1).expand(M, total, 1)
        new_coords = torch.cat([batch_expanded, new_spatial], dim=2)
        new_coords_flat = new_coords.reshape(-1, 4)
        new_feats_flat = feats_reshaped.reshape(-1, C)

        if prune_zeros:
            mask = new_feats_flat.abs().sum(dim=1) != 0
            new_feats_flat = new_feats_flat[mask]
            new_coords_flat = new_coords_flat[mask]

        H, W, D = x.spatial_shape
        return spconv.SparseConvTensor(new_feats_flat, new_coords_flat, (H * r, W * r, D * r), x.batch_size)

    if isinstance(x, FvdbTensor):
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

            if prune_zeros:
                mask = new_feats_flat.abs().sum(dim=1) != 0
                new_feats_flat = new_feats_flat[mask]
                new_coords_flat = new_coords_flat[mask]

            new_coord_list.append(new_coords_flat.to(dtype=torch.int32).contiguous())
            new_feat_list.append(new_feats_flat.contiguous())

        voxel_sizes = getattr(x.grid, 'voxel_sizes', 1.0)
        origins = getattr(x.grid, 'origins', 0.0)
        return _fvdb_tensor_from_lists(new_coord_list, new_feat_list, voxel_sizes / r, origins, device)

    raise TypeError("Input must be either a spconv.SparseConvTensor or an FvdbTensor.")


class Interleaver(nn.Module):
    """
    A unified interleaver that applies the interleaving operation based on the input type.
    If input is a torch.Tensor, it uses the dense function.
    If input is an spconv.SparseConvTensor, it uses the sparse function.
    """

    def __init__(self, r: int, use_cuda: bool = False):
        super().__init__()
        self.r = r
        self.use_cuda = use_cuda

    def forward(self, x):
        if isinstance(x, FvdbTensor) or (spconv is not None and isinstance(x, spconv.SparseConvTensor)):
            return sparse_interleaving_fn(x, self.r, self.use_cuda)
        if isinstance(x, torch.Tensor):
            if self.use_cuda:
                pass
            else:
                return interleaving_fn(x, self.r)
        raise TypeError(
            "Input must be either a torch.Tensor, a spconv.SparseConvTensor, or an FvdbTensor.")

class Deinterleaver(nn.Module):
    """
    A unified deinterleaver that applies the deinterleaving operation based on the input type.
    If input is a torch.Tensor, it uses the dense function.
    If input is an spconv.SparseConvTensor, it uses the sparse function.
    """

    def __init__(self, r: int, prune_zeros: bool = False, use_cuda: bool = False):
        super().__init__()
        self.r = r
        self.prune_zeros = prune_zeros
        self.use_cuda = use_cuda

    def forward(self, x):
        if isinstance(x, FvdbTensor) or (spconv is not None and isinstance(x, spconv.SparseConvTensor)):
            return sparse_deinterleaving_fn(x, self.r, self.prune_zeros)
        if isinstance(x, torch.Tensor):
            if self.use_cuda:
                pass
            else:
                return deinterleaving_fn(x, self.r)
        raise TypeError(
            "Input must be either a torch.Tensor, a spconv.SparseConvTensor, or an FvdbTensor.")
