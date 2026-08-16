"""Depth-sorted alpha-compositing renderer for paper-quality 3DGS figures.

This renderer is intentionally independent of the fast normalized-blending
preview renderer in :mod:`viz.render_gs`. It processes depth-ordered Gaussian
chunks and performs front-to-back alpha compositing per pixel, retaining the
same camera and EWA covariance projection used by the preview path.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from viz.render_gs import look_at, quat_to_rotmat


def _composite_chunk(pixel, local_index, alpha, color, pixel_count):
    """Return premultiplied RGB and transmittance for one depth-sorted chunk."""
    if pixel.numel() == 0:
        rgb = torch.zeros(pixel_count, 3, device=color.device, dtype=color.dtype)
        return rgb, torch.ones(pixel_count, device=color.device, dtype=color.dtype)

    chunk_size = color.shape[0]
    key = pixel.to(torch.int64) * (chunk_size + 1) + local_index.to(torch.int64)
    order = torch.argsort(key)
    pixel = pixel[order]
    local_index = local_index[order]
    alpha = alpha[order].clamp(0.0, 0.995)

    log_remaining = torch.log1p(-alpha)
    cumulative = torch.cumsum(log_remaining, dim=0)
    before = cumulative - log_remaining
    starts = torch.ones_like(pixel, dtype=torch.bool)
    starts[1:] = pixel[1:] != pixel[:-1]
    start_indices = starts.nonzero(as_tuple=True)[0]
    end_indices = torch.cat(
        [start_indices[1:], torch.tensor([pixel.numel()], device=pixel.device)]
    )
    counts = end_indices - start_indices
    offsets = torch.repeat_interleave(before[start_indices], counts)
    transmittance_before = torch.exp(before - offsets)
    weights = transmittance_before * alpha

    rgb = torch.zeros(pixel_count, 3, device=color.device, dtype=color.dtype)
    rgb.index_add_(0, pixel, weights.unsqueeze(1) * color[local_index])
    log_transmittance = torch.zeros(pixel_count, device=color.device, dtype=color.dtype)
    log_transmittance.index_add_(0, pixel, log_remaining)
    return rgb, torch.exp(log_transmittance)


@torch.no_grad()
def render_alpha(
    xyz,
    scale,
    quat,
    opacity,
    color,
    elev,
    azim,
    dist_mul,
    res,
    fov,
    bg,
    device,
    center=None,
    radius=None,
    chunk_size=1024,
    max_radius=32.0,
    alpha_threshold=1.0 / 255.0,
    up_axis="y",
):
    """Render Gaussians with depth-sorted front-to-back alpha compositing.

    Gaussian chunks are globally ordered by center depth. Compositing each
    chunk over the accumulated image preserves that order while bounding peak
    memory for Full-GS assets containing tens of thousands of primitives.
    """
    xyz = xyz.to(device=device, dtype=torch.float32)
    scale = scale.to(device=device, dtype=torch.float32).clamp_min(1e-6)
    quat = quat.to(device=device, dtype=torch.float32)
    opacity = opacity.to(device=device, dtype=torch.float32).flatten().clamp(0.0, 1.0)
    color = color.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    if center is None:
        center = 0.5 * (xyz.amin(0) + xyz.amax(0))
    else:
        center = torch.as_tensor(center, device=device, dtype=torch.float32)
    if radius is None:
        radius = (xyz - center).norm(dim=-1).max()
    else:
        radius = torch.as_tensor(radius, device=device, dtype=torch.float32)
    radius = radius.clamp_min(1e-6)

    camera, rotation_wc = look_at(
        center,
        elev,
        azim,
        dist=radius * dist_mul,
        up_axis=up_axis,
    )
    camera_points = (xyz - camera) @ rotation_wc.t()
    depth = camera_points[:, 2]
    focal = 0.5 * res / math.tan(0.5 * math.radians(fov))
    safe_depth = depth.clamp_min(1e-4)
    mean_x = focal * camera_points[:, 0] / safe_depth + res / 2
    mean_y = focal * camera_points[:, 1] / safe_depth + res / 2

    rotation = quat_to_rotmat(quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8))
    transform = rotation * scale.unsqueeze(1)
    covariance = transform @ transform.transpose(1, 2)
    camera_covariance = rotation_wc @ covariance @ rotation_wc.t()
    jacobian = torch.zeros(xyz.shape[0], 2, 3, device=device)
    jacobian[:, 0, 0] = focal / safe_depth
    jacobian[:, 0, 2] = -focal * camera_points[:, 0] / safe_depth.square()
    jacobian[:, 1, 1] = focal / safe_depth
    jacobian[:, 1, 2] = -focal * camera_points[:, 1] / safe_depth.square()
    covariance_2d = jacobian @ camera_covariance @ jacobian.transpose(1, 2)
    covariance_2d[:, 0, 0] += 0.3
    covariance_2d[:, 1, 1] += 0.3
    determinant = (
        covariance_2d[:, 0, 0] * covariance_2d[:, 1, 1]
        - covariance_2d[:, 0, 1] * covariance_2d[:, 1, 0]
    ).clamp_min(1e-8)
    inverse = torch.stack(
        [
            covariance_2d[:, 1, 1],
            -covariance_2d[:, 0, 1],
            -covariance_2d[:, 1, 0],
            covariance_2d[:, 0, 0],
        ],
        dim=-1,
    ).reshape(-1, 2, 2) / determinant[:, None, None]
    trace = covariance_2d[:, 0, 0] + covariance_2d[:, 1, 1]
    max_eigenvalue = 0.5 * trace + (0.25 * trace.square() - determinant).clamp_min(0).sqrt()
    splat_radius = (3.0 * max_eigenvalue.sqrt()).clamp(1.0, max_radius)

    visible = (
        (depth > 1e-4)
        & (opacity > alpha_threshold)
        & (mean_x > -max_radius)
        & (mean_x < res + max_radius)
        & (mean_y > -max_radius)
        & (mean_y < res + max_radius)
    )
    indices = visible.nonzero(as_tuple=True)[0]
    if indices.numel() == 0:
        raise RuntimeError("no Gaussians in view; check camera parameters")
    indices = indices[torch.argsort(depth[indices])]

    pixel_count = res * res
    image = torch.zeros(pixel_count, 3, device=device)
    transmittance = torch.ones(pixel_count, device=device)
    background = torch.as_tensor(bg, device=device, dtype=torch.float32)

    for begin in range(0, indices.numel(), chunk_size):
        selected = indices[begin : begin + chunk_size]
        local_count = selected.numel()
        radius_px = int(splat_radius[selected].max().ceil().item())
        offsets = torch.arange(-radius_px, radius_px + 1, device=device)
        offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing="ij")
        offset_x = offset_x.flatten()
        offset_y = offset_y.flatten()

        pixel_x = mean_x[selected].round().long().unsqueeze(1) + offset_x
        pixel_y = mean_y[selected].round().long().unsqueeze(1) + offset_y
        delta_x = pixel_x.float() + 0.5 - mean_x[selected, None]
        delta_y = pixel_y.float() + 0.5 - mean_y[selected, None]
        inverse_selected = inverse[selected]
        exponent = -0.5 * (
            inverse_selected[:, 0, 0, None] * delta_x.square()
            + (inverse_selected[:, 0, 1, None] + inverse_selected[:, 1, 0, None])
            * delta_x
            * delta_y
            + inverse_selected[:, 1, 1, None] * delta_y.square()
        )
        alpha = opacity[selected, None] * torch.exp(exponent.clamp(max=0.0))
        inside_radius = (
            offset_x.square()[None] + offset_y.square()[None]
            <= splat_radius[selected, None].square()
        )
        valid = (
            inside_radius
            & (pixel_x >= 0)
            & (pixel_x < res)
            & (pixel_y >= 0)
            & (pixel_y < res)
            & (alpha > alpha_threshold)
        )
        pixel = (pixel_y * res + pixel_x)[valid]
        local_index = (
            torch.arange(local_count, device=device)[:, None]
            .expand(-1, offset_x.numel())[valid]
        )
        chunk_rgb, chunk_transmittance = _composite_chunk(
            pixel,
            local_index,
            alpha[valid],
            color[selected],
            pixel_count,
        )
        image += transmittance.unsqueeze(1) * chunk_rgb
        transmittance *= chunk_transmittance

    image += transmittance.unsqueeze(1) * background
    image = image.reshape(res, res, 3).clamp(0.0, 1.0)
    return (image.cpu().numpy() * 255.0).round().astype(np.uint8)
