import setup_paths  # noqa: F401

import gzip
import os
from matplotlib import pyplot as plt
import numpy as np
import open3d as o3d
from open3d.visualization import gui, rendering

from losses.metrics import Metrics

os.environ["XDG_SESSION_TYPE"] = "x11"


def read_binary_file(file_path: str, size: int = 256, depth: int = 256) -> np.ndarray:
    with gzip.open(file_path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.int32)
    data = data.byteswap().view(np.uint8)
    return np.unpackbits(data).reshape((size, size, depth))


def make_pcd_with_cmap(
    grid: np.ndarray, cmap_name: str, coordinate_axis: int = 0, voxel_size: float = 1.0
) -> o3d.geometry.PointCloud:
    """
    Create a PCD from a binary grid, coloring each point by a matplotlib colormap
    based on its index along `coordinate_axis` (0=z, 1=y, 2=x).
    """
    occupied = np.argwhere(grid == 1)  # (N,3) in (z, y, x)
    dims = np.array(grid.shape)
    # center & scale coords
    coords = (occupied - dims / 2).astype(np.float32) * voxel_size

    # use normalized index along chosen axis for colormap scalar
    idx = occupied[:, coordinate_axis].astype(np.float32)
    idx_norm = idx / (dims[coordinate_axis] - 1)

    cmap = plt.get_cmap(cmap_name)
    colors = cmap(idx_norm)[:, :3]  # drop alpha

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # normals for true lighting
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
    pcd.normalize_normals()

    return pcd


def make_diff_pcd_with_cmap(
    grid1: np.ndarray,
    grid2: np.ndarray,
    cmap_name: str = "cividis",
    voxel_size: float = 1.0,
) -> o3d.geometry.PointCloud:
    """
    Create a PCD of voxels where grid1 != grid2, coloring by radial distance colormap.
    """
    diff_idx = np.argwhere(grid1 != grid2)
    if len(diff_idx) == 0:
        print("No differences found between the two grids.")
        return o3d.geometry.PointCloud()

    dims = np.array(grid1.shape)
    coords = (diff_idx - dims / 2).astype(np.float32) * voxel_size

    # radial distance from center
    r = np.linalg.norm(coords, axis=1)
    r_norm = r / r.max()

    cmap = plt.get_cmap(cmap_name)
    colors = cmap(r_norm)[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
    pcd.normalize_normals()

    return pcd


def visualize_three_panes_with_cmaps(
    grid1: np.ndarray,
    grid2: np.ndarray,
    grid3: np.ndarray,
    voxel_size: float = 1.0,
    window_title: str = "GV | PVV | Diff",
):
    gui.Application.instance.initialize()
    width, height = 2400, 600  # three panes, each 800×600
    window = gui.Application.instance.create_window(window_title, width, height)

    # prepare 3 SceneWidgets
    scenes = [gui.SceneWidget() for _ in range(3)]
    for s in scenes:
        window.add_child(s)
        s.scene = rendering.Open3DScene(window.renderer)
        s.scene.set_background([0, 0, 0, 1])
        sun = np.array([0.577, -0.577, 0.577], dtype=np.float32)
        s.scene.set_lighting(rendering.Open3DScene.LightingProfile.MED_SHADOWS, sun)

    # build colored point clouds with different colormaps
    pcd_gv = make_pcd_with_cmap(
        grid1, cmap_name="viridis", coordinate_axis=0, voxel_size=voxel_size
    )
    pcd_pvv = make_pcd_with_cmap(
        grid2, cmap_name="plasma", coordinate_axis=0, voxel_size=voxel_size
    )
    pcd_diff = make_pcd_with_cmap(
        grid3, cmap_name="cividis", coordinate_axis=0, voxel_size=voxel_size
    )

    # prepare PBR materials
    def make_mat():
        m = rendering.MaterialRecord()
        m.shader = "defaultLit"
        m.point_size = 3.0
        return m

    mats = [make_mat(), make_mat(), make_mat()]

    # add each to its scene
    for scene, pcd, mat, title in zip(
        scenes,
        [pcd_gv, pcd_pvv, pcd_diff],
        mats,
        ["GV (Viridis)", "PVV (Plasma)", "Diff (Cividis)"],
    ):
        scene.scene.add_geometry(title, pcd, mat)
        aabb = pcd.get_axis_aligned_bounding_box()
        scene.setup_camera(60.0, aabb, aabb.get_center())

    # layout callback: split into 3 equal columns
    def on_layout(ctx):
        r = window.content_rect
        w3 = r.width // 3
        for i, s in enumerate(scenes):
            s.frame = gui.Rect(r.x + i * w3, r.y, w3, r.height)

    window.set_on_layout(on_layout)
    gui.Application.instance.run()


def revert_viewport_grid_to_world(
    grid: np.ndarray, fov_x: float, near: float, far: float
):
    """
    Unproject a binary 3D grid in viewport‐space back into camera‐space.

    Args:
        grid   : (H, W, Z) binary array (e.g. occupancy from your renderer)
        fov_x  : horizontal field‐of‐view in radians
        near   : distance to near plane
        far    : distance to far plane

    Returns:
        pts    : (N,3) float32 array of (x,y,z) camera‐space coords
                 for all voxels where grid>0
    """
    H, W, Z = grid.shape
    # Recover vertical fov from aspect
    aspect = W / H
    fov_y = 2 * np.arctan(np.tan(fov_x / 2) / aspect)

    # create arrays of pixel‐centers and depths
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    z_vals = np.linspace(near, far, Z, dtype=np.float32)
    uu, vv, zz = np.meshgrid(u, v, z_vals, indexing="xy")

    # unproject to camera‐space
    x = (uu - (W - 1) / 2) / (W - 1) * 2 * zz * np.tan(fov_x / 2)
    y = ((H - 1) / 2 - vv) / (H - 1) * 2 * zz * np.tan(fov_y / 2)

    # stack into (H, W, Z, 3)
    coords = np.stack((x, y, zz), axis=-1)

    # mask out only the occupied voxels
    mask = grid.astype(bool)
    pts = coords[mask]

    return pts.astype(np.float32)


import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter


def threshold3d(probs: torch.Tensor, thresh: float) -> torch.Tensor:
    """
    Binarize a probability map by threshold, boosting recall when thresh < 0.5.
    Input: [B,1,H,W,Z] float in [0,1].  Output: same shape, dtype of probs.
    """
    if probs.dim() != 5 or probs.size(1) != 1:
        raise ValueError(
            f"Expected input of shape [B,1,H,W,Z], got {tuple(probs.shape)}"
        )
    return (probs > thresh).to(probs.dtype)


def dilate3d(x: torch.Tensor, n: int) -> torch.Tensor:
    """
    Dilate a binary volume by an n×n×n cube, ensuring original points remain.
    """
    if x.dim() != 5 or x.size(1) != 1:
        raise ValueError(f"Expected input of shape [B,1,H,W,Z], got {tuple(x.shape)}")
    pad = n // 2
    x_float = x.to(torch.float32)
    dilated = (F.max_pool3d(x_float, kernel_size=n, stride=1, padding=pad) > 0).to(
        x.dtype
    )
    return torch.max(dilated, x)


def erode3d(x: torch.Tensor, n: int) -> torch.Tensor:
    """
    Erode a binary volume by an n×n×n cube (shrinks away small islands).
    """
    if x.dim() != 5 or x.size(1) != 1:
        raise ValueError(f"Expected input of shape [B,1,H,W,Z], got {tuple(x.shape)}")
    pad = n // 2
    x_float = x.to(torch.float32)
    # invert mask, dilate, then invert back → erosion
    inv = 1.0 - x_float
    inv_dil = F.max_pool3d(inv, kernel_size=n, stride=1, padding=pad)
    eroded = (1.0 - inv_dil > 0).to(x.dtype)
    return eroded


def close3d(x: torch.Tensor, n: int) -> torch.Tensor:
    """
    Morphological closing: dilation followed by erosion.
    """
    return erode3d(dilate3d(x, n), n)


# def fill_holes3d(x: torch.Tensor) -> torch.Tensor:
#     """
#     Fill interior holes in each volume slice-by-slice in 3D.
#     Uses scipy.ndimage.binary_fill_holes on CPU.
#     """
#     if x.dim() != 5 or x.size(1) != 1:
#         raise ValueError(f"Expected input of shape [B,1,H,W,Z], got {tuple(x.shape)}")
#     device, dtype = x.device, x.dtype
#     x_np = x.cpu().numpy()
#     out_np = np.zeros_like(x_np)
#     B = x_np.shape[0]
#     for i in range(B):
#         out_np[i,0] = binary_fill_holes(x_np[i,0])
#     return torch.from_numpy(out_np).to(dtype).to(device)

from scipy.ndimage import label, generate_binary_structure, find_objects


def fill_holes3d(
    x: torch.Tensor,
    *,
    slice_by_slice: bool = True,
    connectivity: int = 1,
    max_hole_size: int = None,
) -> torch.Tensor:
    """
    Fill interior holes in a binary volume.

    Args:
      x:        [B,1,H,W,Z] binary tensor
      slice_by_slice:
               if True, fill holes independently in each HxW slice along Z;
               if False, treat the whole HxWxZ as one volume.
      connectivity: 1..3, how to define neighbors in 3D (1=faces,2=edges,3=corners)
      max_hole_size: if not None, only fill holes with <= this many voxels.

    Returns:
      [B,1,H,W,Z] binary tensor with holes filled.
    """
    if x.dim() != 5 or x.size(1) != 1:
        raise ValueError(f"Expected [B,1,H,W,Z], got {tuple(x.shape)}")

    device, dtype = x.device, x.dtype
    x_np = x.cpu().numpy().astype(bool)
    B, _, H, W, Z = x_np.shape
    struct3d = generate_binary_structure(rank=3, connectivity=connectivity)

    out = np.empty_like(x_np)
    for b in range(B):
        vol = x_np[b, 0]
        if slice_by_slice:
            # fill holes per (H,W) slice
            for z in range(Z):
                slice2d = vol[:, :, z]
                inv = ~slice2d
                struct2d = generate_binary_structure(2, connectivity)
                lbl, num = label(inv, structure=struct2d)
                for lab in range(1, num + 1):
                    mask = lbl == lab
                    size = mask.sum()
                    # only holes (not background): check touches border
                    if not (
                        mask[0, :].any()
                        or mask[-1, :].any()
                        or mask[:, 0].any()
                        or mask[:, -1].any()
                    ):
                        if max_hole_size is None or size <= max_hole_size:
                            slice2d[mask] = True
                vol[:, :, z] = slice2d
        else:
            # fill true 3D holes
            inv = ~vol
            lbl, num = label(inv, structure=struct3d)
            # find bounding boxes for speed
            bbs = find_objects(lbl)
            for lab, bb in enumerate(bbs, start=1):
                if bb is None:
                    continue
                mask = lbl[bb] == lab
                size = mask.sum()
                # hole if it doesn’t touch any face of the full volume
                # check if label appears on any bounding box that hits a volume face
                # but simpler: reject if any voxel has coordinate index==0 or max
                coords = np.where(lbl == lab)
                if (
                    coords[0].min() == 0
                    or coords[0].max() == H - 1
                    or coords[1].min() == 0
                    or coords[1].max() == W - 1
                    or coords[2].min() == 0
                    or coords[2].max() == Z - 1
                ):
                    continue
                if max_hole_size is None or size <= max_hole_size:
                    vol[coords] = True

        out[b, 0] = vol

    return torch.from_numpy(out).to(device).to(dtype)


def smooth_threshold3d(
    probs: torch.Tensor, sigma: float, thresh: float
) -> torch.Tensor:
    """
    Gaussian-smooth a probability map, then re-threshold.
    """
    if probs.dim() != 5 or probs.size(1) != 1:
        raise ValueError(
            f"Expected input of shape [B,1,H,W,Z], got {tuple(probs.shape)}"
        )
    device, dtype = probs.device, probs.dtype
    p_np = probs.cpu().numpy()
    out_np = np.zeros_like(p_np)
    B = p_np.shape[0]
    for i in range(B):
        sm = gaussian_filter(p_np[i, 0], sigma=sigma)
        out_np[i, 0] = sm > thresh
    return torch.from_numpy(out_np).to(dtype).to(device)


def fill_surface_from_y(
    x: torch.Tensor, y: torch.Tensor, n: int, include_ratio: float = 0.9
) -> torch.Tensor:
    """
    For each n×n×n sub-block of y that is 'largely included' in x (i.e.
    overlap >= include_ratio * block_volume), but still missing some voxels,
    fill those missing y-voxels into x.

    Args:
        x:            [B,1,H,W,Z] binary mask (0/1) – your prediction
        y:            [B,1,H,W,Z] binary mask (0/1) – reference geometry
        n:            window size (must be odd or even; we pad by n//2)
        include_ratio: float in (0,1], fraction of y-voxels in each block
                       that must already be present in x to trigger filling.

    Returns:
        A new tensor of same shape & dtype as `x`, with those missing
        boundary voxels of y filled into x.
    """
    if x.shape != y.shape or x.dim() != 5 or x.size(1) != 1:
        raise ValueError(
            f"Expected x,y both of shape [B,1,H,W,Z], got {tuple(x.shape)} vs {tuple(y.shape)}"
        )

    # prepare
    B, _, H, W, Z = x.shape
    pad = n // 2
    device, dtype = x.device, x.dtype

    # float versions for convs
    xf = x.to(torch.float32)
    yf = y.to(torch.float32)
    xy = xf * yf

    # build all-ones kernel
    kernel = torch.ones(1, 1, n, n, n, device=device, dtype=torch.float32)

    # sliding sums
    sum_y = F.conv3d(yf, kernel, padding=pad)
    sum_xy = F.conv3d(xy, kernel, padding=pad)

    # windows where x already covers at least include_ratio of y, but misses some
    enough_overlap = sum_xy >= (include_ratio * sum_y)
    still_missing = sum_xy < sum_y
    trigger_windows = (enough_overlap & still_missing).to(torch.float32)

    # now any voxel within *any* triggered window should be considered
    # we dilate that trigger mask back out with the same kernel
    # (equivalent to marking all voxels in those windows)
    trigger_broadcast = (
        F.max_pool3d(trigger_windows, kernel_size=n, stride=1, padding=pad) > 0
    )

    # finally, only fill the y-voxels that x was missing
    missing_voxels = (yf > 0) & (xf == 0)
    to_fill = trigger_broadcast & missing_voxels

    # return x with those filled
    return torch.where(to_fill, torch.ones_like(x), x)


# def tta_predict3d(
#     model: Callable[[torch.Tensor], torch.Tensor],
#     x: torch.Tensor,
#     transforms: List[Tuple[Callable, Callable]],
#     thresh: float = 0.5
# ) -> torch.Tensor:
#     """
#     Test-time augmentation: apply each (forward, inverse) pair, OR the results, threshold.
#     - model: maps [B,1,H,W,Z]→[B,1,H,W,Z] probability tensor
#     - transforms: list of (apply_fn, invert_fn) pairs
#     """
#     preds = []
#     for apply_tf, invert_tf in transforms:
#         xi = apply_tf(x)
#         pi = model(xi)
#         pi_inv = invert_tf(pi)
#         preds.append(pi_inv)
#     stacked = torch.stack(preds, dim=0)      # [T,B,1,H,W,Z]
#     maxed   = stacked.max(dim=0)[0]          # [B,1,H,W,Z]
#     return (maxed > thresh).to(x.dtype)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize and evaluate 3D geometry predictions."
    )
    parser.add_argument("--id", type=int, default=25, help="Sample ID to visualize.")
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the dataset directory."
    )
    parser.add_argument(
        "--exp_path",
        type=str,
        required=True,
        help="Path to the experiment output directory.",
    )
    args = parser.parse_args()

    id = args.id
    dataset_path = args.dataset_path
    exp_path = args.exp_path

    gv_path = f"{dataset_path}/gv/{id}_gv.bin.gz"
    gt_pvv_path = f"{dataset_path}/pvv/{id}_pvv.bin.gz"
    pvv_path = f"{exp_path}/inference/0/{id}_predicted_pvv.bin.gz"

    gv = read_binary_file(gv_path, size=256, depth=256)
    pvv = read_binary_file(gt_pvv_path, size=256, depth=256)
    pred = read_binary_file(pvv_path, size=256, depth=256)

    # Mask PVV by GV occupancy if needed
    pred = np.where(gv > 0, pred, np.zeros_like(pred))

    pvv = (pvv > 0.5).astype(gv.dtype)
    pvv = np.where(gv > 0, pvv, np.zeros_like(pvv))

    diff_gv = np.abs(gv - pred)
    diff_pvv = np.abs(pvv - pred)

    print(diff_gv.sum(), diff_pvv.sum())

    gv_torch = torch.from_numpy(gv).unsqueeze(0).unsqueeze(0).to("cuda")
    pred_torch = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).to("cuda")
    pvv_torch = torch.from_numpy(pvv).unsqueeze(0).unsqueeze(0).to("cuda")

    metrics = Metrics()
    results = metrics.forward(pred_torch, pvv_torch, {"gv": gv_torch, "pvv": pvv_torch})
    print("Metrics: ", results)

    visualize_three_panes_with_cmaps(
        np.abs(gv - pred), np.abs(pvv - pred), pred, voxel_size=1.0
    )

    methods = {
        # 'Threshold (0.3)': lambda x: threshold3d(x, 0.3),
        # 'dilate + dilate (11)': lambda x: dilate3d(dilate3d(x, 9), 9),
        "fill_surface_from_y  + dilate (11)": lambda x: dilate3d(
            fill_surface_from_y(x, gv_torch, 11), 11
        ),
        "fill_surface_from_y": lambda x: fill_surface_from_y(x, gv_torch, 11),
        "dilate + fill_surface_from_y": lambda x: fill_surface_from_y(
            dilate3d(x, 11), gv_torch, 5, 0.8
        ),
        "Dilate (11)": lambda x: dilate3d(x, 11),
        # 'Closing (11)': lambda x: close3d(x, 11),
        # 'dilate + close (11)': lambda x: close3d(dilate3d(x, 11), 11),
        # 'close + dilate (11)': lambda x: dilate3d(close3d(x, 11), 11),
        # 'dilate + fill (11)': lambda x: fill_holes3d(dilate3d(x, 11), slice_by_slice=False, connectivity=2, max_hole_size=100),
        # 'Fill Holes': lambda x: fill_holes3d(x, slice_by_slice=False, connectivity=2, max_hole_size=100),
        # 'Smooth + Threshold (σ=1, thr=0.3)': lambda x: smooth_threshold3d(x, sigma=1.0, thresh=0.3),
        # 'TTA (max-OR)': lambda x: tta_predict3d(model, x, transforms, thresh=0.5),
    }

    # Evaluation loop
    for name, postprocess in methods.items():
        pred_processed = postprocess(pred_torch)
        # Mask out regions outside ground truth validity (if gv_torch indicates valid mask)
        pred_processed = torch.where(
            gv_torch > 0, pred_processed, torch.zeros_like(pred_processed)
        )
        # Convert to NumPy for visualization
        pred_np = pred_processed.squeeze(0).squeeze(0).cpu().numpy()

        # Compute metrics
        result = metrics.forward(
            pred_processed, pvv_torch, {"gv": gv_torch, "pvv": pvv_torch}
        )
        print(f"{name}: {result}")

        # Visualize differences and predictions
        visualize_three_panes_with_cmaps(
            np.abs(gv - pred_np), np.abs(pvv - pred_np), pred_np, voxel_size=1.0
        )
