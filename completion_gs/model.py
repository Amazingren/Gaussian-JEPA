"""Frozen-backbone Gaussian completion with learned queries only."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Gaussian_JEPA_ExpMultiScale import _EncoderCore
from models.transformer import Group


class _NS(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__dict__ = self


def encoder_config():
    transformer = _NS(
        trans_dim=384,
        encoder_dims=384,
        depth=12,
        drop_path_rate=0.1,
        num_heads=6,
    )
    return _NS(
        attribute=["xyz", "opacity", "scale", "rotation", "sh"],
        group_attribute=["xyz"],
        transformer_config=transformer,
    )


def checkpoint_state(path: str):
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("base_model", checkpoint.get("model", checkpoint))
    prefix = "module."
    return {
        (key[len(prefix) :] if key.startswith(prefix) else key): value
        for key, value in state.items()
    }


class FrozenGaussianBackbone(nn.Module):
    """Common inference wrapper for architecture-matched JEPA and MAE encoders."""

    def __init__(self, checkpoint: str, prefix: str, num_group: int = 32, group_size: int = 32):
        super().__init__()
        self.group = Group(num_group, group_size, attribute=["xyz"], soft_knn=False)
        self.encoder = _EncoderCore(encoder_config(), soft_knn=False)
        state = checkpoint_state(checkpoint)
        selected = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        if not selected:
            raise KeyError(f"no checkpoint keys found for prefix {prefix!r} in {checkpoint}")
        incompatible = self.encoder.load_state_dict(selected, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"incompatible encoder state: {incompatible}")
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.encoder.eval()
        self.loaded_keys = len(selected)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def forward(self, partial: torch.Tensor) -> torch.Tensor:
        neighborhood, center = self.group(partial)
        tokens = self.encoder.encoder(neighborhood)
        position = self.encoder.pos_embed(center[..., :3])
        return self.encoder.norm(self.encoder.blocks(tokens, position))


class CompletionDecoder(nn.Module):
    """Predict a complete 1K Gaussian set without ground-truth spatial queries."""

    def __init__(
        self,
        token_dim: int = 384,
        decoder_dim: int = 256,
        coarse_points: int = 256,
        up_factor: int = 4,
        depth: int = 4,
        heads: int = 8,
    ):
        super().__init__()
        self.coarse_points = coarse_points
        self.up_factor = up_factor
        self.output_points = coarse_points * up_factor
        self.memory_proj = nn.Linear(token_dim, decoder_dim)
        self.global_proj = nn.Sequential(
            nn.Linear(token_dim * 2, decoder_dim), nn.GELU(), nn.Linear(decoder_dim, decoder_dim)
        )
        self.queries = nn.Parameter(torch.empty(1, coarse_points, decoder_dim))
        layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=heads,
            dim_feedforward=decoder_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=depth)
        self.coarse_xyz = nn.Linear(decoder_dim, 3)
        grid = torch.tensor(
            [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]], dtype=torch.float32
        )
        if up_factor != 4:
            raise ValueError("the current leakage-free decoder uses a fixed 2x2 folding grid")
        self.register_buffer("folding_grid", grid, persistent=False)
        refine_in = decoder_dim * 2 + 3 + 2
        self.refine = nn.Sequential(
            nn.Linear(refine_in, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, decoder_dim),
            nn.GELU(),
        )
        self.xyz_residual = nn.Linear(decoder_dim, 3)
        self.attr_head = nn.Linear(decoder_dim, 11)
        nn.init.trunc_normal_(self.queries, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch = tokens.shape[0]
        pooled = torch.cat([tokens.mean(1), tokens.amax(1)], dim=-1)
        global_feature = self.global_proj(pooled)
        memory = self.memory_proj(tokens)
        queries = self.queries.expand(batch, -1, -1) + global_feature.unsqueeze(1)
        coarse_feature = self.decoder(queries, memory)
        coarse_xyz = torch.tanh(self.coarse_xyz(coarse_feature))

        feature = coarse_feature.unsqueeze(2).expand(-1, -1, self.up_factor, -1)
        global_expanded = global_feature[:, None, None, :].expand_as(feature)
        xyz = coarse_xyz.unsqueeze(2).expand(-1, -1, self.up_factor, -1)
        grid = self.folding_grid[None, None].expand(batch, self.coarse_points, -1, -1)
        refined_feature = self.refine(
            torch.cat([feature, global_expanded, xyz, grid], dim=-1)
        )
        refined_xyz = torch.tanh(xyz + 0.1 * self.xyz_residual(refined_feature))
        attributes = self.attr_head(refined_feature)
        opacity = torch.tanh(attributes[..., 0:1])
        scale = torch.tanh(attributes[..., 1:4])
        rotation = attributes[..., 4:8]
        rotation = rotation / rotation.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        rotation = rotation * torch.where(rotation[..., :1] < 0, -1.0, 1.0)
        sh = torch.tanh(attributes[..., 8:11])
        output = torch.cat([refined_xyz, opacity, scale, rotation, sh], dim=-1)
        return output.reshape(batch, self.output_points, 14)


class GaussianCompletion(nn.Module):
    def __init__(self, encoder_checkpoint: str, encoder_prefix: str):
        super().__init__()
        self.backbone = FrozenGaussianBackbone(encoder_checkpoint, encoder_prefix)
        self.decoder = CompletionDecoder()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, partial: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            tokens = self.backbone(partial)
        return self.decoder(tokens)


def chamfer_and_matches(pred_xyz: torch.Tensor, target_xyz: torch.Tensor):
    # Direct squared-L2 Chamfer matches the repository's Gaussian-MAE setup
    # and avoids an unnecessary square root in the training objective.
    pred_xyz = pred_xyz.float()
    target_xyz = target_xyz.float()
    pred_sq = pred_xyz.square().sum(dim=-1, keepdim=True)
    target_sq = target_xyz.square().sum(dim=-1).unsqueeze(1)
    squared = (pred_sq + target_sq - 2.0 * pred_xyz @ target_xyz.transpose(1, 2)).clamp_min(0.0)
    pred_min_sq, pred_match = squared.min(dim=2)
    target_min_sq = squared.min(dim=1).values
    chamfer = pred_min_sq.mean() + target_min_sq.mean()
    pred_dist = pred_min_sq.clamp_min(1e-12).sqrt()
    target_dist = target_min_sq.clamp_min(1e-12).sqrt()
    return chamfer, pred_dist, target_dist, pred_match


def completion_loss(pred: torch.Tensor, target: torch.Tensor, attr_weight: float = 0.1):
    chamfer, pred_dist, target_dist, match = chamfer_and_matches(pred[..., :3], target[..., :3])
    gather = match.unsqueeze(-1).expand(-1, -1, 11)
    target_attr = torch.gather(target[..., 3:], 1, gather)
    opacity = F.l1_loss(pred[..., 3:4], target_attr[..., 0:1])
    scale = F.l1_loss(pred[..., 4:7], target_attr[..., 1:4])
    pred_quat = F.normalize(pred[..., 7:11], dim=-1)
    target_quat = F.normalize(target_attr[..., 4:8], dim=-1)
    rotation = (1.0 - (pred_quat * target_quat).sum(-1).abs()).mean()
    sh = F.l1_loss(pred[..., 11:14], target_attr[..., 8:11])
    attr = opacity + scale + rotation + sh
    loss = chamfer + attr_weight * attr
    metrics = {
        "loss": loss.detach(),
        "chamfer": chamfer.detach(),
        "opacity": opacity.detach(),
        "scale": scale.detach(),
        "rotation": rotation.detach(),
        "sh": sh.detach(),
        "fscore": fscore_from_distances(pred_dist.detach(), target_dist.detach()),
    }
    return loss, metrics


def fscore_from_distances(pred_dist: torch.Tensor, target_dist: torch.Tensor, threshold: float = 0.01):
    precision = (pred_dist < threshold).float().mean(dim=1)
    recall = (target_dist < threshold).float().mean(dim=1)
    return (2.0 * precision * recall / (precision + recall).clamp_min(1e-8)).mean()
