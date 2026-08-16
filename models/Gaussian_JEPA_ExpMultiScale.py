"""Gaussian-JEPA pretraining and ModelNet transfer models.

The pretraining model predicts four non-overlapping target blocks with sizes
``[11, 9, 7, 5]`` from their shared visible context. An EMA target encoder
provides stop-gradient targets in two learned latent spaces. Their projectors
are regularized in feature space with VISReg and cross-covariance objectives;
no raw Gaussian attribute is reconstructed.

The legacy class names are retained because they form part of the released
checkpoint state dictionary. Public configurations use concise registry aliases
defined in :mod:`models`.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .build import MODELS
from models.transformer import Encoder, Group, SoftEncoder, TransformerDecoder, TransformerEncoder
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import print_log


# 14-dim Gaussian center layout: [xyz(0-2) | opacity(3) | scale(4-6) | rotation(7-10) | sh(11-13)]
_GEO_DIMS = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]   # xyz + scale + rotation  (10 dims)
_APP_DIMS  = [3, 11, 12, 13]                     # opacity + sh/colour       (4 dims)


# ---------------------------------------------------------------------------
# VISReg feature-space grounding for the two target projections
# ---------------------------------------------------------------------------

class GeoAppGrounding(nn.Module):
    """Feature-space grounding via VISReg + cross-stream decorrelation.

    Distributional regularisation is applied to target-projector outputs.

    The module must be called with tensors that still have requires_grad=True
    (i.e. before any .detach()) so that gradients propagate into the projection
    heads that produced them.

    Args:
        num_projections: K random directions for the Sliced Wasserstein Distance.
    """

    def __init__(self, num_projections: int = 256):
        super().__init__()
        self.K = num_projections
        # Cache normal quantiles for the current batch size to avoid recomputation.
        self._cached_N = -1
        self._cached_target = None

    def _get_normal_quantiles(self, N: int, device, dtype) -> torch.Tensor:
        if self._cached_N != N:
            q = torch.linspace(1, N, N, device=device, dtype=torch.float32) / (N + 1)
            self._cached_target = torch.erfinv(2 * q - 1).mul_(math.sqrt(2))
            self._cached_N = N
        return self._cached_target.to(device=device, dtype=dtype)

    def _visreg(self, z: torch.Tensor) -> torch.Tensor:
        """VISReg loss on z of shape (N, D).

        Three terms matching the VISReg paper:
          center : penalise non-zero mean  → representations stay centred
          scale  : penalise ||·||/√N ≠ 1  → prevents norm collapse
          shape  : SWD to N(0,I)          → distributional alignment, stronger
                                             than variance alone (captures higher
                                             moments, gradient doesn't vanish)
        """
        N, D = z.shape
        z3 = z.unsqueeze(0)                              # (1, N, D) — V=1 convention

        mu   = z3.mean(dim=1, keepdim=True)              # (1, 1, D)
        center_loss = mu.pow(2).mean()

        z_c  = z3 - mu                                   # (1, N, D) centred
        std  = z_c.norm(dim=1).div(math.sqrt(N)) + 1e-6  # (1, D)
        scale_loss = (std - 1.0).pow(2).mean()

        z_n  = z_c / std.detach().unsqueeze(1)           # (1, N, D) normalised
        W    = F.normalize(
            torch.randn(D, self.K, device=z.device, dtype=z.dtype), dim=0
        )                                                # (D, K) random directions
        p_sorted = (z_n @ W).sort(dim=1).values         # (1, N, K) sorted projections
        target   = self._get_normal_quantiles(N, z.device, z.dtype).view(1, N, 1)
        shape_loss = (p_sorted - target).pow(2).mean()

        return center_loss + scale_loss + shape_loss

    def _cross_decouple(self, geo: torch.Tensor, app: torch.Tensor) -> torch.Tensor:
        """Cross-stream cross-covariance penalty (VICReg-style).

        Penalises linear correlation between the geo and app subspaces so that
        the two projection heads are forced to capture complementary information.
        geo, app: (N, D)
        """
        N, D = geo.shape
        geo_c = geo - geo.mean(0)
        app_c = app - app.mean(0)
        cross_cov = (geo_c.T @ app_c) / N    # (D, D)
        return (cross_cov ** 2).sum() / D

    def forward(
        self,
        geo_list: list[torch.Tensor],
        app_list: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute grounding loss from M blocks of pre-detach teacher projections.

        Args:
            geo_list: [tensor(B, N_tgt, D), ...] × M — geo projections before detach
            app_list: [tensor(B, N_tgt, D), ...] × M — app projections before detach

        Returns:
            Scalar grounding loss (has gradient w.r.t. projection head parameters).
        """
        geo_all = torch.cat(geo_list, dim=1)   # (B, M·N_tgt, D)
        app_all = torch.cat(app_list, dim=1)

        N = geo_all.shape[0] * geo_all.shape[1]
        D = geo_all.shape[2]

        geo_flat = geo_all.reshape(N, D)
        app_flat = app_all.reshape(N, D)

        return (
            self._visreg(geo_flat)
            + self._visreg(app_flat)
            + self._cross_decouple(geo_flat, app_flat)
        )


# ---------------------------------------------------------------------------
# Shared encoder backbone (student + teacher)
# ---------------------------------------------------------------------------

class _EncoderCore(nn.Module):

    def __init__(self, config, soft_knn=False):
        super().__init__()
        self.trans_dim      = config.transformer_config.trans_dim
        self.depth          = config.transformer_config.depth
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads      = config.transformer_config.num_heads
        self.encoder_dims   = config.transformer_config.encoder_dims

        self.encoder = (
            SoftEncoder(encoder_channel=self.encoder_dims, attribute=config.attribute)
            if soft_knn
            else Encoder(encoder_channel=self.encoder_dims, attribute=config.attribute)
        )

        self.pos_feature_dim = []
        if "xyz"      in config.group_attribute: self.pos_feature_dim.extend([0, 1, 2])
        if "opacity"  in config.group_attribute: self.pos_feature_dim.append(3)
        if "scale"    in config.group_attribute: self.pos_feature_dim.extend([4, 5, 6])
        if "rotation" in config.group_attribute: self.pos_feature_dim.extend([7, 8, 9, 10])
        if "sh"       in config.group_attribute: self.pos_feature_dim.extend([11, 12, 13])

        self.pos_embed = nn.Sequential(
            nn.Linear(len(self.pos_feature_dim), 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim, depth=self.depth,
            drop_path_rate=dpr, num_heads=self.num_heads,
        )
        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward_context(self, neighborhood, center, union_mask):
        all_tokens = self.encoder(neighborhood)
        B, G, C    = all_tokens.shape
        x_ctx      = all_tokens[~union_mask].reshape(B, -1, C)
        pos_ctx    = self.pos_embed(center[~union_mask].reshape(B, -1, len(self.pos_feature_dim)))
        return self.norm(self.blocks(x_ctx, pos_ctx))

    def forward_all(self, neighborhood, center):
        """Encode the complete token field once, without selecting target blocks."""
        all_tokens = self.encoder(neighborhood)
        return self.norm(self.blocks(all_tokens, self.pos_embed(center)))

    def forward_full(self, neighborhood, center, mask):
        all_tokens = self.encoder(neighborhood)
        x_all      = self.norm(self.blocks(all_tokens, self.pos_embed(center)))
        return x_all[mask].reshape(x_all.size(0), -1, x_all.size(2))


# ---------------------------------------------------------------------------
# Multi-target spatial block sampler
# ---------------------------------------------------------------------------

class MultiTargetMaskSampler(nn.Module):

    def __init__(
        self,
        mask_ratio: float,
        num_targets: int,
        target_sizes: list[int],
    ):
        super().__init__()
        self.mask_ratio   = mask_ratio
        self.num_targets  = num_targets
        self.target_sizes = tuple(int(size) for size in target_sizes)

        if len(self.target_sizes) != self.num_targets:
            raise ValueError(
                f"target_sizes must contain M={self.num_targets} entries, "
                f"got {self.target_sizes}"
            )
        if any(size <= 0 for size in self.target_sizes):
            raise ValueError(f"target_sizes must be positive, got {self.target_sizes}")

    def forward(self, center, noaug=False):
        B, G, _ = center.shape
        device  = center.device
        if noaug or self.mask_ratio == 0:
            empty = torch.zeros(B, G, dtype=torch.bool, device=device)
            return [empty] * self.num_targets, empty
        if sum(self.target_sizes) >= G:
            raise ValueError(
                f"target_sizes={self.target_sizes} mask {sum(self.target_sizes)} "
                f"of G={G} tokens, leaving no context"
            )
        return self._multi_block(center)

    def _multi_block(self, center):
        B, G, _ = center.shape
        device  = center.device
        pts     = center[..., :3]

        available = torch.ones(B, G, device=device)
        masks     = []

        # Large-to-small ordering is deliberate: large semantic regions are
        # selected before the available set becomes spatially fragmented.
        for per_mask_num in self.target_sizes:
            anchor_idx = torch.multinomial(available, 1).squeeze(-1)
            anchor_pts = pts[torch.arange(B, device=device), anchor_idx].unsqueeze(1)
            dist       = torch.norm(pts - anchor_pts, p=2, dim=-1)
            dist       = dist + (1.0 - available) * 1e9
            target_idx = torch.argsort(dist, dim=-1)[:, :per_mask_num]

            mask_t = torch.zeros(B, G, dtype=torch.bool, device=device)
            mask_t.scatter_(1, target_idx, True)
            available = available * (~mask_t).float()
            masks.append(mask_t)

        union_mask = torch.stack(masks, dim=0).any(dim=0)
        return masks, union_mask


# ---------------------------------------------------------------------------
# Gaussian-JEPA pretraining model
# ---------------------------------------------------------------------------

@MODELS.register_module()
class Gaussian_JEPA_ExpMultiScale(nn.Module):
    """Gaussian-JEPA with four heterogeneous, non-overlapping target blocks."""

    def __init__(self, config):
        super().__init__()
        print_log("[Gaussian-JEPA]", logger="Gaussian_JEPA_ExpMultiScale")
        self.config        = config
        self.soft_knn      = getattr(config, "soft_knn", False)
        self.trans_dim     = config.transformer_config.trans_dim
        self.group_size    = config.group_size
        self.num_group     = config.num_group
        self.pred_dim      = getattr(config.transformer_config, "pred_dim", 192)
        self.num_targets   = getattr(config, "num_targets", 4)
        self.target_sizes   = list(getattr(config, "target_sizes", [11, 9, 7, 5]))
        self.lambda_ground = getattr(config, "lambda_ground", 0.1)

        self.group_divider = Group(
            num_group=self.num_group, group_size=self.group_size,
            attribute=config.group_attribute, soft_knn=self.soft_knn,
        )

        self.mask_sampler = MultiTargetMaskSampler(
            mask_ratio=config.transformer_config.mask_ratio,
            num_targets=self.num_targets,
            target_sizes=self.target_sizes,
        )

        self.JEPA_encoder    = _EncoderCore(config, soft_knn=self.soft_knn)
        self.pos_feature_dim = self.JEPA_encoder.pos_feature_dim

        self.teacher_encoder = copy.deepcopy(self.JEPA_encoder)
        for p in self.teacher_encoder.parameters():
            p.requires_grad_(False)

        # Target projection heads are trained by feature-space grounding.
        self.teacher_geo_proj = nn.Sequential(
            nn.Linear(self.trans_dim, self.trans_dim),
            nn.GELU(),
            nn.Linear(self.trans_dim, self.trans_dim),
        )
        self.teacher_app_proj = nn.Sequential(
            nn.Linear(self.trans_dim, self.trans_dim),
            nn.GELU(),
            nn.Linear(self.trans_dim, self.trans_dim),
        )

        # VISReg feature-space grounding.
        num_swd = getattr(config, "num_swd_projections", 256)
        self.grounding = GeoAppGrounding(num_projections=num_swd)

        # Predictor
        pos_dim = len(self.pos_feature_dim)
        self.pred_input_proj     = nn.Linear(self.trans_dim, self.pred_dim)
        self.predictor_pos_embed = nn.Sequential(
            nn.Linear(pos_dim, 128), nn.GELU(), nn.Linear(128, self.pred_dim),
        )
        self.query_token = nn.Parameter(torch.zeros(1, 1, self.pred_dim))

        dpr = [x.item() for x in torch.linspace(
            0, config.transformer_config.drop_path_rate,
            config.transformer_config.decoder_depth,
        )]
        self.predictor = TransformerDecoder(
            embed_dim=self.pred_dim,
            depth=config.transformer_config.decoder_depth,
            drop_path_rate=dpr,
            num_heads=config.transformer_config.decoder_num_heads,
        )

        self.pred_geo_out = nn.Linear(self.pred_dim, self.trans_dim)
        self.pred_app_out = nn.Linear(self.pred_dim, self.trans_dim)

        trunc_normal_(self.query_token, std=0.02)
        self._init_extra_weights()

        print_log(
            f"[Gaussian-JEPA] G={self.num_group}×S={self.group_size}, "
            f"M={self.num_targets}, target_sizes={self.target_sizes}, "
            f"mask_ratio(actual)={sum(self.target_sizes) / self.num_group:.4f}, "
            f"pred_dim={self.pred_dim}, lambda_ground={self.lambda_ground} | "
            f"VISReg(K={num_swd}) + CrossCov grounding",
            logger="Gaussian_JEPA_ExpMultiScale",
        )

    def _init_extra_weights(self):
        for m in [self.pred_input_proj, self.pred_geo_out, self.pred_app_out]:
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)
        for seq in [self.teacher_geo_proj, self.teacher_app_proj, self.predictor_pos_embed]:
            for m in seq:
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None: nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def _update_teacher(self, momentum: float = 0.996):
        for s_p, t_p in zip(
            self.JEPA_encoder.parameters(), self.teacher_encoder.parameters()
        ):
            t_p.data.mul_(momentum).add_(s_p.data, alpha=1.0 - momentum)

    def load_model_from_ckpt(self, ckpt_path):
        if ckpt_path is None:
            print_log("Training from scratch!!!", logger="Gaussian_JEPA_ExpMultiScale")
            return
        ckpt      = torch.load(ckpt_path, map_location="cpu")
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt["base_model"].items()}
        incompatible = self.load_state_dict(base_ckpt, strict=False)
        if incompatible.missing_keys:
            print_log(get_missing_parameters_message(incompatible.missing_keys),
                      logger="Gaussian_JEPA_ExpMultiScale")
        if incompatible.unexpected_keys:
            print_log(get_unexpected_parameters_message(incompatible.unexpected_keys),
                      logger="Gaussian_JEPA_ExpMultiScale")
        print_log(f"[Gaussian-JEPA] Loaded from {ckpt_path}", logger="Gaussian_JEPA_ExpMultiScale")

    def forward(self, pts, noaug=False, **kwargs):
        neighborhood, center = self.group_divider(pts)
        center_pos = center[..., self.pos_feature_dim]   # (B, G, pos_dim)
        B      = center_pos.size(0)
        pos_dim = len(self.pos_feature_dim)

        masks, union_mask = self.mask_sampler(center_pos, noaug=noaug)

        # Student: one forward pass over context tokens (shared across all M blocks)
        x_ctx = self.JEPA_encoder.forward_context(neighborhood, center_pos, union_mask)

        pred_pos_ctx = self.predictor_pos_embed(
            center_pos[~union_mask].reshape(B, -1, pos_dim)
        )
        x_ctx_proj = self.pred_input_proj(x_ctx)

        total_geo = total_app = 0.0
        geo_list: list[torch.Tensor] = []   # pre-detach — for VISReg grounding
        app_list: list[torch.Tensor] = []

        for mask_i in masks:
            N_tgt = mask_i.sum(dim=1)[0].item()

            with torch.no_grad():
                x_tgt_full = self.teacher_encoder.forward_full(neighborhood, center_pos, mask_i)

            # Project into geo / app subspaces.
            # NOTE: x_tgt_full has no grad (teacher is EMA / no_grad), but
            # teacher_geo_proj.parameters() DO require grad, so the output
            # x_tgt_geo carries gradient w.r.t. the projection head weights.
            x_tgt_geo = self.teacher_geo_proj(x_tgt_full)   # (B, N_tgt, D)
            x_tgt_app = self.teacher_app_proj(x_tgt_full)   # (B, N_tgt, D)

            # Accumulate for VISReg grounding (must be BEFORE detach)
            geo_list.append(x_tgt_geo)
            app_list.append(x_tgt_app)

            # JEPA targets: stop-gradient + LayerNorm for stable training
            tgt_geo = F.layer_norm(x_tgt_geo.detach(), x_tgt_geo.shape[-1:])
            tgt_app = F.layer_norm(x_tgt_app.detach(), x_tgt_app.shape[-1:])

            # Predictor
            pred_pos_tgt = self.predictor_pos_embed(
                center_pos[mask_i].reshape(B, -1, pos_dim)
            )
            queries  = self.query_token.expand(B, N_tgt, -1)
            x_full   = torch.cat([x_ctx_proj, queries], dim=1)
            pos_full = torch.cat([pred_pos_ctx, pred_pos_tgt], dim=1)
            x_pred   = self.predictor(x_full, pos_full, N_tgt)

            total_geo += F.smooth_l1_loss(self.pred_geo_out(x_pred), tgt_geo)
            total_app += F.smooth_l1_loss(self.pred_app_out(x_pred), tgt_app)

        # Grounding operates on pre-detach target projections.
        # Provides gradient to teacher_geo_proj and teacher_app_proj via:
        #   VISReg(geo) + VISReg(app): each subspace spans the embedding space
        #   CrossCov(geo, app):        geo and app subspaces stay decorrelated
        loss_ground = self.grounding(geo_list, app_list)

        M = self.num_targets
        return {
            "geo":    total_geo / M,
            "app":    total_app / M,
            "ground": self.lambda_ground * loss_ground,
        }


# ---------------------------------------------------------------------------
# Linear-probe / finetune head
# ---------------------------------------------------------------------------

@MODELS.register_module()
class PointTransformer_JEPA_ExpMultiScale(nn.Module):
    """ModelNet classifier initialized from a Gaussian-JEPA encoder."""

    def __init__(self, config, **kwargs):
        super().__init__()
        self.trans_dim      = config.trans_dim
        self.depth          = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim        = config.cls_dim
        self.num_heads      = config.num_heads
        self.soft_knn       = getattr(config, "soft_knn", False)
        self.group_size     = config.group_size
        self.num_group      = config.num_group
        self.encoder_dims   = config.encoder_dims

        self.group_divider = Group(
            num_group=self.num_group, group_size=self.group_size,
            attribute=config.group_attribute, soft_knn=self.soft_knn,
        )
        self.encoder = (
            SoftEncoder(encoder_channel=self.encoder_dims, attribute=config.attribute)
            if self.soft_knn
            else Encoder(encoder_channel=self.encoder_dims, attribute=config.attribute)
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos   = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_feature_dim = []
        if "xyz"      in config.group_attribute: self.pos_feature_dim.extend([0, 1, 2])
        if "opacity"  in config.group_attribute: self.pos_feature_dim.append(3)
        if "scale"    in config.group_attribute: self.pos_feature_dim.extend([4, 5, 6])
        if "rotation" in config.group_attribute: self.pos_feature_dim.extend([7, 8, 9, 10])
        if "sh"       in config.group_attribute: self.pos_feature_dim.extend([11, 12, 13])

        self.pos_embed = nn.Sequential(
            nn.Linear(len(self.pos_feature_dim), 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim, depth=self.depth,
            drop_path_rate=dpr, num_heads=self.num_heads,
        )
        self.norm = nn.LayerNorm(self.trans_dim)

        if config.type == "linear":
            self.cls_head_finetune = nn.Linear(self.trans_dim * 2, self.cls_dim)
        else:
            self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * 2, 256), nn.BatchNorm1d(256),
                nn.ReLU(inplace=True), nn.Dropout(0.5),
                nn.Linear(256, 256), nn.BatchNorm1d(256),
                nn.ReLU(inplace=True), nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim),
            )

        self.loss_ce = nn.CrossEntropyLoss()
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.cls_pos,   std=0.02)

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc  = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, ckpt_path):
        if ckpt_path is None:
            print_log("Training from scratch!!!", logger="Gaussian_JEPA_ExpMultiScale")
            self.apply(self._init_weights)
            return
        ckpt      = torch.load(ckpt_path, map_location="cpu")
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt["base_model"].items()}
        remapped  = {
            k[len("JEPA_encoder."):]: v
            for k, v in base_ckpt.items()
            if k.startswith("JEPA_encoder.")
        }
        incompatible = self.load_state_dict(remapped, strict=False)
        if incompatible.missing_keys:
            print_log(get_missing_parameters_message(incompatible.missing_keys),
                      logger="Gaussian_JEPA_ExpMultiScale")
        print_log(f"[Gaussian_JEPA_ExpMultiScale] Loaded from {ckpt_path}",
                  logger="Gaussian_JEPA_ExpMultiScale")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)

    def forward(self, pts):
        neighborhood, center = self.group_divider(pts)
        tokens = self.encoder(neighborhood)

        cls_tokens = self.cls_token.expand(tokens.size(0), -1, -1)
        cls_pos    = self.cls_pos.expand(tokens.size(0),   -1, -1)
        pos        = self.pos_embed(center[..., self.pos_feature_dim])

        x   = torch.cat([cls_tokens, tokens], dim=1)
        pos = torch.cat([cls_pos,    pos],    dim=1)
        x   = self.norm(self.blocks(x, pos))
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        return self.cls_head_finetune(concat_f)
