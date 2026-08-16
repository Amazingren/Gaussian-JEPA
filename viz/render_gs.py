"""Self-contained GPU 3DGS rasteriser: colored .ply -> PNG.

No gsplat / diff-gaussian-rasterization / CUDA compilation needed -- pure PyTorch
EWA splatting on the GPU. Meant for previewing the part-discovery / PCA .ply files
(flat per-part colours), not photorealism: uses order-independent normalised
weighted blending, which is clean and fast for these small objects.

Reads raw 3DGS attributes (opacity=logit, scale=log, rot=unnormalised quat) and
colour from f_dc (color = 0.5 + SH_C0 * f_dc), matching write_colored_ply().

Usage:
  python viz/render_gs.py --ply experiments/part_discovery/*.ply
  python viz/render_gs.py --ply a.ply --elev 20 --azim 135 --res 800
  python viz/render_gs.py --ply a.ply --azim 0 90 180 270      # multi-view grid
"""

import argparse
import math
import os

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData

SH_C0 = 0.28209479177387814


def load_gaussians(path, device):
    v = PlyData.read(path)["vertex"].data
    g = lambda k: torch.from_numpy(np.ascontiguousarray(v[k])).float().to(device)
    xyz = torch.stack([g("x"), g("y"), g("z")], -1)                 # (N,3)
    scale = torch.stack([g(f"scale_{i}") for i in range(3)], -1).exp()
    quat = torch.stack([g(f"rot_{i}") for i in range(4)], -1)        # (N,4) wxyz
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    opacity = torch.sigmoid(g("opacity"))                            # (N,)
    fdc = torch.stack([g(f"f_dc_{i}") for i in range(3)], -1)
    color = (0.5 + SH_C0 * fdc).clamp(0, 1)                          # (N,3)
    return xyz, scale, quat, opacity, color


def quat_to_rotmat(q):
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], -1).reshape(-1, 3, 3)
    return R


def look_at(center, elev, azim, dist, up_axis="y"):
    """Camera position on a sphere around center; returns C (3,), R_wc (3,3).

    ``up_axis`` makes the object convention explicit. ModelNet assets in the
    existing paper figures are Y-up, while the native ShapeNet cars used for
    the framework illustration are Z-up.
    """
    e, a = math.radians(elev), math.radians(azim)
    if up_axis == "y":
        d = np.array([
            math.cos(e) * math.sin(a),
            math.sin(e),
            math.cos(e) * math.cos(a),
        ])
        up_values = [0.0, 1.0, 0.0]
    elif up_axis in {"z", "neg_z"}:
        sign = 1.0 if up_axis == "z" else -1.0
        d = np.array([
            math.cos(e) * math.sin(a),
            math.cos(e) * math.cos(a),
            sign * math.sin(e),
        ])
        up_values = [0.0, 0.0, sign]
    else:
        raise ValueError(f"unsupported up axis: {up_axis}")
    C = center + torch.from_numpy(d).float().to(center.device) * dist
    z = (center - C); z = z / z.norm()                              # forward (+z cam)
    up = torch.tensor(up_values, device=C.device)
    x = torch.cross(up, z); x = x / x.norm().clamp_min(1e-8)
    y = torch.cross(z, x)
    R_wc = torch.stack([x, y, z], 0)                                 # rows
    return C, R_wc


@torch.no_grad()
def render(xyz, scale, quat, opacity, color, elev, azim, dist_mul, res, fov, bg, device,
           center=None, radius=None, up_axis="y"):
    # pass a shared center/radius to keep multiple objects on the same camera
    if center is None:
        center = 0.5 * (xyz.amin(0) + xyz.amax(0))
    if radius is None:
        radius = (xyz - center).norm(dim=-1).max()
    C, R_wc = look_at(
        center,
        elev,
        azim,
        dist=radius * dist_mul,
        up_axis=up_axis,
    )

    # camera-space means + depth
    pcam = (xyz - C) @ R_wc.t()                                     # (N,3)
    Z = pcam[:, 2]
    front = Z > 1e-4
    focal = 0.5 * res / math.tan(0.5 * math.radians(fov))
    u = focal * pcam[:, 0] / Z + res / 2
    vv = focal * pcam[:, 1] / Z + res / 2
    mean2d = torch.stack([u, vv], -1)                               # (N,2)

    # 3D covariance -> camera -> 2D (EWA)
    Rm = quat_to_rotmat(quat)
    M = Rm * scale.unsqueeze(1)                                     # R @ diag(s)
    Sigma = M @ M.transpose(1, 2)                                   # (N,3,3)
    Sig_c = R_wc @ Sigma @ R_wc.t()
    Zc = Z.clamp_min(1e-4)
    J = torch.zeros(xyz.size(0), 2, 3, device=device)
    J[:, 0, 0] = focal / Zc; J[:, 0, 2] = -focal * pcam[:, 0] / (Zc * Zc)
    J[:, 1, 1] = focal / Zc; J[:, 1, 2] = -focal * pcam[:, 1] / (Zc * Zc)
    cov2d = J @ Sig_c @ J.transpose(1, 2)                           # (N,2,2)
    cov2d[:, 0, 0] += 0.3; cov2d[:, 1, 1] += 0.3                    # low-pass
    det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] * cov2d[:, 1, 0]
    det = det.clamp_min(1e-8)
    inv = torch.stack([cov2d[:, 1, 1], -cov2d[:, 0, 1],
                       -cov2d[:, 1, 0], cov2d[:, 0, 0]], -1).reshape(-1, 2, 2) / det.view(-1, 1, 1)
    # splat radius (3 sigma) from max eigenvalue
    tr = cov2d[:, 0, 0] + cov2d[:, 1, 1]
    lam = 0.5 * tr + (0.25 * tr * tr - det).clamp_min(0).sqrt()
    rad = (3.0 * lam.sqrt()).clamp(1, 24)

    inb = front & (u > -30) & (u < res + 30) & (vv > -30) & (vv < res + 30)
    idx = inb.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        raise RuntimeError("no gaussians in view — check camera params")
    mean2d, inv, opacity, color, rad = mean2d[idx], inv[idx], opacity[idx], color[idx], rad[idx]

    R = int(rad.max().ceil().item())
    off = torch.arange(-R, R + 1, device=device)
    oy, ox = torch.meshgrid(off, off, indexing="ij")
    ox, oy = ox.reshape(-1), oy.reshape(-1)                         # (W2,)

    px = mean2d[:, 0].round().long().unsqueeze(1) + ox.unsqueeze(0)  # (M,W2)
    py = mean2d[:, 1].round().long().unsqueeze(1) + oy.unsqueeze(0)
    dx = px.float() + 0.5 - mean2d[:, 0:1]
    dy = py.float() + 0.5 - mean2d[:, 1:2]
    # d^T inv d
    a, b, c, dd = inv[:, 0, 0:1], inv[:, 0, 1:2], inv[:, 1, 0:1], inv[:, 1, 1:2]
    power = -0.5 * (a * dx * dx + (b + c) * dx * dy + dd * dy * dy)
    w = opacity.unsqueeze(1) * torch.exp(power.clamp(max=0.0))       # (M,W2)

    valid = (px >= 0) & (px < res) & (py >= 0) & (py < res) & (w > 1e-4)
    flat = (py * res + px)[valid]
    wv = w[valid]
    col = color.unsqueeze(1).expand(-1, ox.numel(), 3)[valid]        # (K,3)

    num = torch.zeros(res * res, 3, device=device)
    den = torch.zeros(res * res, device=device)
    num.index_add_(0, flat, wv.unsqueeze(1) * col)
    den.index_add_(0, flat, wv)
    den = den.clamp_min(1e-8)
    img = num / den.unsqueeze(1)
    bgc = torch.tensor(bg, device=device).float()
    mask = (den < 1e-3).unsqueeze(1)
    img = torch.where(mask, bgc, img)
    return (img.reshape(res, res, 3).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", nargs="+", required=True)
    ap.add_argument("--out", default=None, help="output dir (default: alongside each ply)")
    ap.add_argument("--elev", type=float, default=20.0)
    ap.add_argument("--azim", type=float, nargs="+", default=[135.0])
    ap.add_argument("--dist", type=float, default=2.6, help="distance = dist * object_radius")
    ap.add_argument("--res", type=int, default=800)
    ap.add_argument("--fov", type=float, default=40.0)
    ap.add_argument("--up-axis", choices=("y", "z", "neg_z"), default="y")
    ap.add_argument("--bg", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for p in args.ply:
        gs = load_gaussians(p, dev)
        views = [render(
                    *gs,
                    args.elev,
                    az,
                    args.dist,
                    args.res,
                    args.fov,
                    args.bg,
                    dev,
                    up_axis=args.up_axis,
                 )
                 for az in args.azim]
        grid = np.concatenate(views, axis=1) if len(views) > 1 else views[0]
        out_dir = args.out or os.path.dirname(p)
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, os.path.splitext(os.path.basename(p))[0] + ".png")
        Image.fromarray(grid).save(out)
        print(f"  {os.path.basename(p)} -> {out}  ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    main()
