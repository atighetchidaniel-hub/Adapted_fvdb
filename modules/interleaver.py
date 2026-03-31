import torch
import torch.nn as nn

try:
    import spconv.pytorch as spconv
except Exception:
    spconv = None
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


def sparse_interleaving_fn(x, r: int, use_cuda: bool = False):
    """
    Rearranges a sparse 5D tensor of shape (B, C, H, W, Z) into
    (B, C * r^3, H//r, W//r, Z//r) by interleaving the r^3 neighborhood
    along the channel dimension via a fused CUDA kernel.
    """
    if spconv is None:
        raise ImportError("sparse_interleaving_fn requires spconv, but spconv is not installed.")

    if isinstance(x, spconv.SparseConvTensor):
        coords = x.indices        # (N, 4): [batch, h, w, z]
        feats = x.features        # (N, C)
        backend = 'spconv'
    else:
        raise TypeError(
            "Input must be a spconv.SparseConvTensor.")
    device = feats.device
    N, C = feats.shape
    r2 = r * r
    r3 = r2 * r

    # Compute shrunk (base) coordinates and new coordinates.
    base_coords = coords[:, 1:] // r
    new_coords = torch.cat([coords[:, :1], base_coords], dim=1)  # (N, 4)

    # Compute the offset and map the 3D offset into a single index.
    offsets = coords[:, 1:] % r  # (N, 3)
    offset_idx = offsets[:, 0] * r2 + offsets[:, 1] * r + offsets[:, 2]  # (N,)

    # Group points by the new (shrunk) coordinates.
    unique_coords, inv_idx = torch.unique(
        new_coords, dim=0, return_inverse=True)
    M = unique_coords.size(0)
    assert M > 0, "No unique coordinates found."

    if use_cuda:
        pass
        # aggregated_feats = interleave_ext.sparse_interleave(
        #     feats.contiguous(),
        #     inv_idx,
        #     offset_idx.to(inv_idx.dtype),
        #     M, C, r3
        # )  # (M, C*r3)
    else:
        aggregated_feats = torch.zeros(
            (M, C * r3), device=device, dtype=feats.dtype)

        # Compute target indices:
        #   Each point's feature of shape (C) should be placed in a slot starting at offset_idx[i] * C.
        offset_idx = offset_idx.unsqueeze(1)  # (N, 1)
        ch_range = torch.arange(C, device=device).unsqueeze(0)  # (1, C)
        target_idx = offset_idx * C + ch_range  # (N, C)
        row_idx = inv_idx.unsqueeze(1).expand(-1, C)  # (N, C)

        # Compute flat indices into the aggregated_feats (flattened view).
        flat_idx = row_idx * (C * r3) + target_idx  # (N, C)

        # Scatter features into the aggregated feature vector.
        aggregated_feats.view(-1).scatter_(0,
                                           flat_idx.reshape(-1), feats.reshape(-1))

    # Return a new sparse tensor using spconv constructor
    H, W, D = x.spatial_shape
    d0, d1, d2 = H//r, W//r, D//r
    return spconv.SparseConvTensor(aggregated_feats,
                                   unique_coords,
                                   spatial_shape=(d0, d1, d2),
                                   batch_size=x.batch_size)


def sparse_deinterleaving_fn(x, r: int, prune_zeros: bool = False):
    """
    Reverses the interleaving operation: rearranges a sparsed 5D tensor of shape 
    (B, C * r^3, H, W, Z) back into (B, C, H*r, W*r, Z*r).

    Assume that:
      - The input sparse tensor `x` has coordinates of shape (M, 4)
        (where each row is [batch, base_x, base_y, base_z]) and
      - Its features have shape (M, C * r^3) (with C being the per-point feature size).
    In sparse_interleaving_fn, for each unique (base) coordinate, features from
    the original r^3 points were placed contiguously (ordered by the index:
         offset_idx = offset_x * r^2 + offset_y * r + offset_z).
    This function inverts that process. For each unique base coordinate it:
      1. Splits the aggregated feature into r^3 sub-features (each of shape (C,)).
      2. Reconstructs the original coordinate as:
             [batch, base_x * r + offset_x, base_y * r + offset_y, base_z * r + offset_z]
         where the offsets for offset_idx i are computed as:
             offset_x = i // (r^2)
             offset_y = (i % (r^2)) // r
             offset_z = i % r
    The output is a SparseTensor with coordinates of shape (M * r^3, 4) and features of shape (M * r^3, C).
    """
    if spconv is None:
        raise ImportError("sparse_deinterleaving_fn requires spconv, but spconv is not installed.")

    # Get the aggregated coordinates and features.
    # shape: (M, 4), where columns: [batch, base_x, base_y, base_z]
    if isinstance(x, spconv.SparseConvTensor):
        coords = x.indices
        aggregated_feats = x.features
        backend = 'spconv'
    else:
        raise TypeError(
            "Input must be a spconv.SparseConvTensor.")
    device = aggregated_feats.device
    M = coords.shape[0]
    total = r ** 3
    C = aggregated_feats.shape[1] // total

    # Reshape the aggregated features to (M, r^3, C)
    feats_reshaped = aggregated_feats.view(M, total, C)

    # Create a tensor of offset indices [0, 1, ..., r^3-1] and convert to offsets.
    indices = torch.arange(total, device=device)
    offset_x = indices // (r * r)
    offset_y = (indices % (r * r)) // r
    offset_z = indices % r
    # Offsets have shape (r^3, 3)
    offsets = torch.stack([offset_x, offset_y, offset_z],
                          dim=1).to(coords.dtype)

    # Extract the base (shrunk) spatial coordinates (columns 1:4) and multiply by r.
    base_coords = coords[:, 1:]  # shape: (M, 3)
    base_expanded = base_coords * r  # shape: (M, 3)

    # For each base coordinate, add each offset:
    #   Expand base_expanded to shape (M, r^3, 3) and add offsets (broadcasted from (r^3, 3))
    new_spatial = base_expanded.unsqueeze(
        1) + offsets.unsqueeze(0)  # shape: (M, r^3, 3)

    # Batch indices are in the first column of coords.
    batch = coords[:, :1]  # shape: (M, 1)
    # Expand batch to (M, r^3, 1)
    batch_expanded = batch.unsqueeze(1).expand(M, total, 1)

    # Concatenate the batch indices and the new spatial coordinates to form full coordinates.
    # Resulting shape: (M, r^3, 4)
    new_coords = torch.cat([batch_expanded, new_spatial], dim=2)

    # Flatten the new coordinates and the de-interleaved features.
    new_coords_flat = new_coords.reshape(-1, 4)
    new_feats_flat = feats_reshaped.reshape(-1, C)

    # Optionally prune rows where the feature vector is all zeros.
    if prune_zeros:
        mask = new_feats_flat.abs().sum(dim=1) != 0
        new_feats_flat = new_feats_flat[mask]
        new_coords_flat = new_coords_flat[mask]

    # Return the new SparseTensor.
    H, W, D = x.spatial_shape
    # The spatial shape is the original shape multiplied by r.
    d0, d1, d2 = H * r, W * r, D * r
    return spconv.SparseConvTensor(new_feats_flat, new_coords_flat, (d0, d1, d2), x.batch_size)


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
        if spconv is not None and isinstance(x, spconv.SparseConvTensor):
            return sparse_interleaving_fn(x, self.r, self.use_cuda)
        elif isinstance(x, torch.Tensor):
            if self.use_cuda:
                pass
                # return interleaving_fn_cuda(x, self.r)
            else:
                return interleaving_fn(x, self.r)
        else:
            raise TypeError(
                "Input must be either a torch.Tensor or a spconv.SparseConvTensor.")

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
        if spconv is not None and isinstance(x, spconv.SparseConvTensor):
            return sparse_deinterleaving_fn(x, self.r, self.prune_zeros)
        elif isinstance(x, torch.Tensor):
            if self.use_cuda:
                pass
                # return deinterleaving_fn_cuda(x, self.r)
            else:
                return deinterleaving_fn(x, self.r)
        else:
            raise TypeError(
                "Input must be either a torch.Tensor or a spconv.SparseConvTensor.")
