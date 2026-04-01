from functools import partial

import torch
import torch.nn as nn

try:
    import spconv.pytorch as spconv
    from spconv.core import ConvAlgo
except Exception:
    spconv = None
    ConvAlgo = None

try:
    import fvdb
    import fvdb.nn as fvnn
except Exception:
    fvdb = None
    fvnn = None

from torch_geometric.nn.pool import voxel_grid
from torch_geometric.utils import scatter

from modules.interleaver import Deinterleaver, Interleaver
from utils.tensor import FvdbTensor, to_sparse
from utils.train import trunc_normal_
from utils.pointcept.builder import MODELS
from utils.pointcept.models_misc import offset2batch

"""
Implementation of OACNNs adapted from the Pointcept-based sparse model.
Both spconv and fvdb sparse backends are supported.
"""

algo = ConvAlgo.MaskImplicitGemm if ConvAlgo is not None else None


def _point_batches_from_offset(offset: torch.Tensor, total_points: int) -> torch.Tensor:
    offset = offset.long()
    if offset.numel() == 0:
        return torch.zeros((0,), device=offset.device, dtype=torch.long)
    if offset[-1].item() == total_points:
        return offset2batch(offset)
    if offset.numel() == total_points:
        return offset
    raise ValueError(
        f"Unsupported offset format with shape {tuple(offset.shape)} for {total_points} points."
    )


def _split_by_offset(values: torch.Tensor, offset: torch.Tensor) -> list[torch.Tensor]:
    offset = offset.long().tolist()
    start = 0
    chunks = []
    for end in offset:
        chunks.append(values[start:end].contiguous())
        start = end
    return chunks


def _dict_to_fvdb(input_dict: dict) -> FvdbTensor:
    if fvdb is None:
        raise ImportError("fvdb is required for the fvdb OACNN backend.")

    discrete_coord = input_dict["grid_coord"].to(dtype=torch.int32).contiguous()
    feat = input_dict["feat"].contiguous()
    offset = input_dict["offset"]

    if offset.numel() == feat.shape[0]:
        batch_ids = offset.long()
        batch_count = int(batch_ids.max().item()) + 1 if batch_ids.numel() > 0 else 1
        coord_list = [discrete_coord[batch_ids == i].contiguous() for i in range(batch_count)]
        feat_list = [feat[batch_ids == i].contiguous() for i in range(batch_count)]
    else:
        offset = offset.long()
        coord_list = _split_by_offset(discrete_coord, offset)
        feat_list = _split_by_offset(feat, offset)

    if len(coord_list) == 0:
        coord_list = [torch.zeros((0, 3), device=feat.device, dtype=torch.int32)]
        feat_list = [torch.zeros((0, feat.shape[1]), device=feat.device, dtype=feat.dtype)]

    ijk = fvdb.JaggedTensor.from_list_of_tensors(coord_list)
    grid = fvdb.GridBatch.from_ijk(
        ijk=ijk,
        voxel_sizes=[1.0, 1.0, 1.0],
        origins=[0.0, 0.0, 0.0],
        device=feat.device,
    )
    src = fvdb.JaggedTensor.from_list_of_tensors(feat_list)
    data = grid.inject_from_ijk(ijk, src)
    return FvdbTensor(grid=grid, data=data)


def _fvdb_plan(source_grid, target_grid, kernel_size, stride=1):
    return fvdb.ConvolutionPlan.from_grid_batch(
        kernel_size=kernel_size,
        stride=stride,
        source_grid=source_grid,
        target_grid=target_grid,
    )


def _fvdb_jagged_like(grid, data: torch.Tensor):
    return grid.jagged_like(data.contiguous())


def _fvdb_batch_ids(grid) -> torch.Tensor:
    offsets = grid.ijk.joffsets.long()
    counts = offsets[1:] - offsets[:-1]
    return torch.arange(counts.numel(), device=offsets.device, dtype=torch.long).repeat_interleave(counts)


def _fvdb_cat(a: FvdbTensor, b: FvdbTensor) -> FvdbTensor:
    if not a.grid.is_same(b.grid):
        raise ValueError("FvdbTensor grids must match before concatenation.")
    return FvdbTensor(a.grid, _fvdb_jagged_like(a.grid, torch.cat([a.data.jdata, b.data.jdata], dim=1)))


class FvdbPointwise(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x: FvdbTensor) -> FvdbTensor:
        return x.replace_data(self.linear(x.data.jdata))


class FvdbConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, bias: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = fvnn.SparseConv3d(in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=bias)
        self.bn = fvnn.BatchNorm(out_channels, momentum=0.01)
        self.act = nn.LeakyReLU()

    def forward(self, x: FvdbTensor) -> FvdbTensor:
        plan = _fvdb_plan(x.grid, x.grid, kernel_size=self.kernel_size, stride=1)
        data = self.conv(x.data, plan)
        data = self.bn(data, x.grid)
        return FvdbTensor(x.grid, _fvdb_jagged_like(x.grid, self.act(data.jdata)))


class SpconvBasicBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_fn=None,
        indice_key=None,
        depth=4,
        groups=None,
        grid_size=None,
        bias=False,
    ):
        super().__init__()
        assert embed_channels % groups == 0
        self.groups = groups
        self.embed_channels = embed_channels
        self.proj = nn.ModuleList()
        self.grid_size = grid_size
        self.weight = nn.ModuleList()
        self.l_w = nn.ModuleList()
        self.proj.append(
            nn.Sequential(
                nn.Linear(embed_channels, embed_channels, bias=False),
                norm_fn(embed_channels),
                nn.LeakyReLU(),
            )
        )
        for _ in range(depth - 1):
            self.proj.append(
                nn.Sequential(
                    nn.Linear(embed_channels, embed_channels, bias=False),
                    norm_fn(embed_channels),
                    nn.LeakyReLU(),
                )
            )
            self.l_w.append(
                nn.Sequential(
                    nn.Linear(embed_channels, embed_channels, bias=False),
                    norm_fn(embed_channels),
                    nn.LeakyReLU(),
                )
            )
            self.weight.append(nn.Linear(embed_channels, embed_channels, bias=False))

        self.adaptive = nn.Linear(embed_channels, depth - 1, bias=False)
        self.fuse = nn.Sequential(
            nn.Linear(embed_channels * 2, embed_channels, bias=False),
            norm_fn(embed_channels),
            nn.LeakyReLU(),
        )
        self.voxel_block = spconv.SparseSequential(
            spconv.SubMConv3d(
                embed_channels,
                embed_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                indice_key=indice_key,
                bias=bias,
                algo=algo,
            ),
            norm_fn(embed_channels),
            nn.LeakyReLU(),
            spconv.SubMConv3d(
                embed_channels,
                embed_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                indice_key=indice_key,
                bias=bias,
                algo=algo,
            ),
            norm_fn(embed_channels),
        )
        self.act = nn.LeakyReLU()

    def forward(self, x, clusters):
        feat = x.features
        feats = []
        for i, cluster in enumerate(clusters):
            pw = self.l_w[i](feat)
            pw = pw - scatter(pw, cluster, reduce="mean")[cluster]
            pw = self.weight[i](pw)
            pw = torch.exp(pw - pw.max())
            pw = pw / (scatter(pw, cluster, reduce="sum", dim=0)[cluster] + 1e-6)
            pfeat = self.proj[i](feat) * pw
            pfeat = scatter(pfeat, cluster, reduce="sum")[cluster]
            feats.append(pfeat)
        adp = self.adaptive(feat)
        adp = torch.softmax(adp, dim=1)
        feats = torch.stack(feats, dim=1)
        feats = torch.einsum("l n, l n c -> l c", adp, feats)
        feat = self.proj[-1](feat)
        feat = torch.cat([feat, feats], dim=1)
        feat = self.fuse(feat) + x.features
        res = feat
        x = x.replace_feature(feat)
        x = self.voxel_block(x)
        x = x.replace_feature(self.act(x.features + res))
        return x


class FvdbBasicBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_fn=None,
        indice_key=None,
        depth=4,
        groups=None,
        grid_size=None,
        bias=False,
    ):
        super().__init__()
        assert embed_channels % groups == 0
        self.groups = groups
        self.embed_channels = embed_channels
        self.grid_size = grid_size
        self.proj = nn.ModuleList()
        self.weight = nn.ModuleList()
        self.l_w = nn.ModuleList()
        self.proj.append(
            nn.Sequential(
                nn.Linear(embed_channels, embed_channels, bias=False),
                norm_fn(embed_channels),
                nn.LeakyReLU(),
            )
        )
        for _ in range(depth - 1):
            self.proj.append(
                nn.Sequential(
                    nn.Linear(embed_channels, embed_channels, bias=False),
                    norm_fn(embed_channels),
                    nn.LeakyReLU(),
                )
            )
            self.l_w.append(
                nn.Sequential(
                    nn.Linear(embed_channels, embed_channels, bias=False),
                    norm_fn(embed_channels),
                    nn.LeakyReLU(),
                )
            )
            self.weight.append(nn.Linear(embed_channels, embed_channels, bias=False))

        self.adaptive = nn.Linear(embed_channels, depth - 1, bias=False)
        self.fuse = nn.Sequential(
            nn.Linear(embed_channels * 2, embed_channels, bias=False),
            norm_fn(embed_channels),
            nn.LeakyReLU(),
        )
        self.conv1 = fvnn.SparseConv3d(embed_channels, embed_channels, kernel_size=3, stride=1, bias=bias)
        self.bn1 = fvnn.BatchNorm(embed_channels, momentum=0.01)
        self.conv2 = fvnn.SparseConv3d(embed_channels, embed_channels, kernel_size=3, stride=1, bias=bias)
        self.bn2 = fvnn.BatchNorm(embed_channels, momentum=0.01)
        self.act = nn.LeakyReLU()

    def forward(self, x: FvdbTensor, clusters):
        feat = x.data.jdata
        feats = []
        for i, cluster in enumerate(clusters):
            pw = self.l_w[i](feat)
            pw = pw - scatter(pw, cluster, reduce="mean")[cluster]
            pw = self.weight[i](pw)
            pw = torch.exp(pw - pw.max())
            pw = pw / (scatter(pw, cluster, reduce="sum", dim=0)[cluster] + 1e-6)
            pfeat = self.proj[i](feat) * pw
            pfeat = scatter(pfeat, cluster, reduce="sum")[cluster]
            feats.append(pfeat)

        adp = self.adaptive(feat)
        adp = torch.softmax(adp, dim=1)
        feats = torch.stack(feats, dim=1)
        feats = torch.einsum("l n, l n c -> l c", adp, feats)
        feat = self.proj[-1](feat)
        feat = torch.cat([feat, feats], dim=1)
        feat = self.fuse(feat) + x.data.jdata
        residual = feat
        x = x.replace_data(feat)

        plan = _fvdb_plan(x.grid, x.grid, kernel_size=3, stride=1)
        data = self.conv1(x.data, plan)
        data = self.bn1(data, x.grid)
        data = _fvdb_jagged_like(x.grid, self.act(data.jdata))
        data = self.conv2(data, plan)
        data = self.bn2(data, x.grid)
        data = _fvdb_jagged_like(x.grid, self.act(data.jdata + residual))
        return FvdbTensor(x.grid, data)


class SpconvDownBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        embed_channels,
        depth,
        sp_indice_key,
        point_grid_size,
        num_ref=16,
        groups=None,
        norm_fn=None,
        sub_indice_key=None,
    ):
        super().__init__()
        self.num_ref = num_ref
        self.depth = depth
        self.point_grid_size = point_grid_size
        self.down = spconv.SparseSequential(
            spconv.SparseConv3d(
                in_channels,
                embed_channels,
                kernel_size=2,
                stride=2,
                indice_key=sp_indice_key,
                bias=False,
                algo=algo,
            ),
            norm_fn(embed_channels),
            nn.LeakyReLU(),
        )
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                SpconvBasicBlock(
                    in_channels=embed_channels,
                    embed_channels=embed_channels,
                    depth=len(point_grid_size) + 1,
                    groups=groups,
                    grid_size=point_grid_size,
                    norm_fn=norm_fn,
                    indice_key=sub_indice_key,
                )
            )

    def forward(self, x):
        x = self.down(x)
        coord = x.indices[:, 1:].float()
        batch = x.indices[:, 0]
        clusters = []
        for grid_size in self.point_grid_size:
            cluster = voxel_grid(pos=coord, size=grid_size, batch=batch)
            _, cluster = torch.unique(cluster, return_inverse=True)
            clusters.append(cluster)
        for block in self.blocks:
            x = block(x, clusters)
        return x


class FvdbDownBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        embed_channels,
        depth,
        sp_indice_key,
        point_grid_size,
        num_ref=16,
        groups=None,
        norm_fn=None,
        sub_indice_key=None,
    ):
        super().__init__()
        self.num_ref = num_ref
        self.depth = depth
        self.point_grid_size = point_grid_size
        self.down = FvdbPointwise(in_channels, embed_channels, bias=False)
        self.down_bn = fvnn.BatchNorm(embed_channels, momentum=0.01)
        self.down_act = nn.LeakyReLU()
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                FvdbBasicBlock(
                    in_channels=embed_channels,
                    embed_channels=embed_channels,
                    depth=len(point_grid_size) + 1,
                    groups=groups,
                    grid_size=point_grid_size,
                    norm_fn=norm_fn,
                    indice_key=sub_indice_key,
                )
            )

    def forward(self, x: FvdbTensor):
        coarse_grid = x.grid.coarsened_grid(2)
        data, coarse_grid = x.grid.max_pool(pool_factor=2, data=x.data, coarse_grid=coarse_grid)
        x = FvdbTensor(coarse_grid, data)
        x = self.down(x)
        data = self.down_bn(x.data, x.grid)
        x = FvdbTensor(x.grid, _fvdb_jagged_like(x.grid, self.down_act(data.jdata)))

        coord = x.grid.ijk.jdata.float()
        batch = _fvdb_batch_ids(x.grid)
        clusters = []
        for grid_size in self.point_grid_size:
            cluster = voxel_grid(pos=coord, size=grid_size, batch=batch)
            _, cluster = torch.unique(cluster, return_inverse=True)
            clusters.append(cluster)
        for block in self.blocks:
            x = block(x, clusters)
        return x


class SpconvUpBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        embed_channels,
        depth,
        sp_indice_key,
        norm_fn=None,
        down_ratio=2,
        sub_indice_key=None,
    ):
        super().__init__()
        assert depth > 0
        self.up = spconv.SparseSequential(
            spconv.SparseInverseConv3d(
                in_channels,
                embed_channels,
                kernel_size=down_ratio,
                indice_key=sp_indice_key,
                bias=False,
                algo=algo,
            ),
            norm_fn(embed_channels),
            nn.LeakyReLU(),
        )
        self.blocks = nn.ModuleList()
        self.fuse = nn.Sequential(
            nn.Linear(skip_channels + embed_channels, embed_channels),
            norm_fn(embed_channels),
            nn.LeakyReLU(),
            nn.Linear(embed_channels, embed_channels),
            norm_fn(embed_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x, skip_x):
        x = self.up(x)
        x = x.replace_feature(
            self.fuse(torch.cat([x.features, skip_x.features], dim=1)) + x.features
        )
        return x


class FvdbUpBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        embed_channels,
        depth,
        sp_indice_key,
        norm_fn=None,
        down_ratio=2,
        sub_indice_key=None,
    ):
        super().__init__()
        assert depth > 0
        self.down_ratio = down_ratio
        self.up = FvdbPointwise(in_channels, embed_channels, bias=False)
        self.up_bn = fvnn.BatchNorm(embed_channels, momentum=0.01)
        self.up_act = nn.LeakyReLU()
        self.fuse = nn.Sequential(
            nn.Linear(skip_channels + embed_channels, embed_channels),
            norm_fn(embed_channels),
            nn.LeakyReLU(),
            nn.Linear(embed_channels, embed_channels),
            norm_fn(embed_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x: FvdbTensor, skip_x: FvdbTensor):
        x = self.up(x)
        data = self.up_bn(x.data, x.grid)
        x = FvdbTensor(x.grid, _fvdb_jagged_like(x.grid, self.up_act(data.jdata)))
        refined_data, fine_grid = x.grid.refine(subdiv_factor=self.down_ratio, data=x.data, fine_grid=skip_x.grid)
        x = FvdbTensor(fine_grid, refined_data)
        fused = self.fuse(torch.cat([x.data.jdata, skip_x.data.jdata], dim=1)) + x.data.jdata
        return FvdbTensor(fine_grid, _fvdb_jagged_like(fine_grid, fused))


class FvdbStem(nn.Module):
    def __init__(self, in_channels, embed_channels):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                FvdbConvNormAct(in_channels, embed_channels, kernel_size=3, bias=False),
                FvdbConvNormAct(embed_channels, embed_channels, kernel_size=3, bias=False),
                FvdbConvNormAct(embed_channels, embed_channels, kernel_size=3, bias=False),
            ]
        )

    def forward(self, x: FvdbTensor):
        for block in self.blocks:
            x = block(x)
        return x


@MODELS.register_module()
class _OACNNs(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        embed_channels=64,
        enc_num_ref=[16, 16, 16, 16],
        enc_channels=[64, 64, 128, 256],
        groups=[2, 4, 8, 16],
        enc_depth=[2, 3, 6, 4],
        down_ratio=[2, 2, 2, 2],
        dec_channels=[96, 96, 128, 256],
        point_grid_size=[[16, 32, 64], [8, 16, 24], [4, 8, 12], [2, 4, 6]],
        dec_depth=[2, 2, 2, 2],
        backend_type="spconv",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_stages = len(enc_channels)
        self.embed_channels = embed_channels
        self.backend_type = backend_type
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        if backend_type == "spconv":
            if spconv is None:
                raise ImportError("spconv is required for the spconv OACNN backend.")
            self.stem = spconv.SparseSequential(
                spconv.SubMConv3d(
                    in_channels,
                    embed_channels,
                    kernel_size=3,
                    padding=1,
                    indice_key="stem",
                    bias=False,
                    algo=algo,
                ),
                norm_fn(embed_channels),
                nn.LeakyReLU(),
                spconv.SubMConv3d(
                    embed_channels,
                    embed_channels,
                    kernel_size=3,
                    padding=1,
                    indice_key="stem",
                    bias=False,
                    algo=algo,
                ),
                norm_fn(embed_channels),
                nn.LeakyReLU(),
                spconv.SubMConv3d(
                    embed_channels,
                    embed_channels,
                    kernel_size=3,
                    padding=1,
                    indice_key="stem",
                    bias=False,
                    algo=algo,
                ),
                norm_fn(embed_channels),
                nn.LeakyReLU(),
            )
            down_cls = SpconvDownBlock
            up_cls = SpconvUpBlock
            self.final = spconv.SubMConv3d(dec_channels[0], num_classes, kernel_size=1, algo=algo)
        elif backend_type == "fvdb":
            if fvdb is None:
                raise ImportError("fvdb is required for the fvdb OACNN backend.")
            self.stem = FvdbStem(in_channels, embed_channels)
            down_cls = FvdbDownBlock
            up_cls = FvdbUpBlock
            self.final = FvdbPointwise(dec_channels[0], num_classes, bias=True)
        else:
            raise ValueError(f"Unsupported backend_type: {backend_type}")

        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(self.num_stages):
            self.enc.append(
                down_cls(
                    in_channels=embed_channels if i == 0 else enc_channels[i - 1],
                    embed_channels=enc_channels[i],
                    depth=enc_depth[i],
                    norm_fn=norm_fn,
                    groups=groups[i],
                    point_grid_size=point_grid_size[i],
                    num_ref=enc_num_ref[i],
                    sp_indice_key=f"spconv{i}",
                    sub_indice_key=f"subm{i + 1}",
                )
            )
            self.dec.append(
                up_cls(
                    in_channels=enc_channels[-1] if i == self.num_stages - 1 else dec_channels[i + 1],
                    skip_channels=embed_channels if i == 0 else enc_channels[i - 1],
                    embed_channels=dec_channels[i],
                    depth=dec_depth[i],
                    norm_fn=norm_fn,
                    sp_indice_key=f"spconv{i}",
                    sub_indice_key=f"subm{i}",
                )
            )

        self.apply(self._init_weights)

    def _prepare_input(self, input):
        if self.backend_type == "spconv":
            if spconv is not None and isinstance(input, spconv.SparseConvTensor):
                return input
            if isinstance(input, dict):
                discrete_coord = input["grid_coord"]
                feat = input["feat"]
                offset = input["offset"]
                batch = _point_batches_from_offset(offset, feat.shape[0])
                return spconv.SparseConvTensor(
                    features=feat,
                    indices=torch.cat([batch.unsqueeze(-1), discrete_coord], dim=1).int().contiguous(),
                    spatial_shape=torch.add(torch.max(discrete_coord, dim=0).values, 1).tolist(),
                    batch_size=batch[-1].tolist() + 1 if batch.numel() > 0 else 1,
                )
            if isinstance(input, torch.Tensor):
                return to_sparse(input, "spconv")
            raise ValueError(f"Unsupported spconv OACNN input type: {type(input)}")

        if isinstance(input, FvdbTensor):
            return input
        if isinstance(input, dict):
            return _dict_to_fvdb(input)
        if isinstance(input, torch.Tensor):
            return to_sparse(input, "fvdb")
        raise ValueError(f"Unsupported fvdb OACNN input type: {type(input)}")

    def forward(self, input, data: dict = {}):
        x = self._prepare_input(input)
        x = self.stem(x)
        skips = [x]
        for i in range(self.num_stages):
            x = self.enc[i](x)
            skips.append(x)
        x = skips.pop(-1)
        for i in reversed(range(self.num_stages)):
            skip = skips.pop(-1)
            x = self.dec[i](x, skip)
        x = self.final(x)
        return x

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif spconv is not None and isinstance(m, spconv.SubMConv3d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class OACNNs(_OACNNs):
    def __init__(self, in_channels=1, classes=1, backend_type="spconv", depth=3):
        enc_num_ref = [16, 16, 16, 16]
        enc_channels = [64, 64, 128, 256]
        groups = [2, 4, 8, 16]
        enc_depth = [2, 3, 6, 4]
        down_ratio = [2, 2, 2, 2]
        dec_channels = [96, 96, 128, 256]
        point_grid_size = [[16, 32, 64], [8, 16, 24], [4, 8, 12], [2, 4, 6]]
        dec_depth = [2, 2, 2, 2]

        enc_num_ref = enc_num_ref[:depth]
        enc_channels = enc_channels[:depth]
        groups = groups[:depth]
        enc_depth = enc_depth[:depth]
        down_ratio = down_ratio[:depth]
        dec_channels = dec_channels[:depth]
        point_grid_size = point_grid_size[:depth]
        dec_depth = dec_depth[:depth]

        super().__init__(
            in_channels=in_channels,
            num_classes=classes,
            embed_channels=64,
            enc_num_ref=enc_num_ref,
            enc_channels=enc_channels,
            groups=groups,
            enc_depth=enc_depth,
            down_ratio=down_ratio,
            dec_channels=dec_channels,
            point_grid_size=point_grid_size,
            dec_depth=dec_depth,
            backend_type=backend_type,
        )


class OACNNsInterleaved(_OACNNs):
    def __init__(self, in_channels=1, classes=1, r=2, backend_type="spconv", depth=3):
        self.r = r
        interleaved_in_channels = in_channels * (self.r ** 3)
        interleaved_classes = classes * (self.r ** 3)

        enc_num_ref = [16, 16, 16, 16]
        enc_channels = [64, 64, 128, 256]
        groups = [2, 4, 8, 16]
        enc_depth = [2, 3, 6, 4]
        down_ratio = [2, 2, 2, 2]
        dec_channels = [96, 96, 128, 256]
        point_grid_size = [[16, 32, 64], [8, 16, 24], [4, 8, 12], [2, 4, 6]]
        dec_depth = [2, 2, 2, 2]

        enc_num_ref = enc_num_ref[:depth]
        enc_channels = enc_channels[:depth]
        groups = groups[:depth]
        enc_depth = enc_depth[:depth]
        down_ratio = down_ratio[:depth]
        dec_channels = dec_channels[:depth]
        point_grid_size = point_grid_size[:depth]
        dec_depth = dec_depth[:depth]

        super().__init__(
            in_channels=interleaved_in_channels,
            num_classes=interleaved_classes,
            embed_channels=64,
            enc_num_ref=enc_num_ref,
            enc_channels=enc_channels,
            groups=groups,
            enc_depth=enc_depth,
            down_ratio=down_ratio,
            dec_channels=dec_channels,
            point_grid_size=point_grid_size,
            dec_depth=dec_depth,
            backend_type=backend_type,
        )
        self.interleaver = Interleaver(self.r)
        self.deinterleaver = Deinterleaver(self.r)
        self.apply(self._init_weights)

    def forward(self, input, data={}):
        x = self.interleaver(input)
        out = super().forward(x, data)
        out = self.deinterleaver(out)
        return out
