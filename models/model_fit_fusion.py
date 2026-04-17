"""
FIT fusion model for the active 0414 line.

Design goals:
- Keep the numerical backbone unchanged.
- Keep an explicit numerical skip in `direct`.
- Keep `direct_v3` as the stable baseline.
- Support `residual_v3`, `residual_v4`, and `residual_v5` for comparison.
- Keep `residual_v4/v5` as trend-level, piecewise-constant corrections.
- Keep `disable_text=True` as a strict pure-numerical fallback.
"""

from copy import deepcopy

import torch
import torch.nn as nn

from models.model_fit_num_with_meta import Model_Fit_Num_With_Meta as FitNumericalModel


def _to_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "t", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "f", "off"}:
            return False
        return default
    return bool(value)


class Model_Fit_Fusion(nn.Module):
    """
    Active FIT fusion model for the 0414 branch.

    `direct`
        Numerical stream produces `y_num`.
        Text stream produces several candidate text forecasts.
        Text-conditioned routing mixes those candidates into `y_text`.
        Final prediction uses an explicit step-wise gate:
        y_final = (1 - gate) * y_num + gate * y_text

    `residual`
        `residual_v3`
            Numerical side produces several candidate correction experts.
            Text side produces routing weights and correction radius.
            Final correction must pass through text routing.

        `residual_v4`
            Text first reads numerical trend context, then emits a low-frequency
            piecewise-constant correction in forecast space.

        `residual_v5`
            Text uses segment queries to attend numerical patch memory, then
            predicts piecewise-constant corrections from text-aligned segment
            contexts. This is a stronger trend-routing variant of `v4`.
    """

    def __init__(self, configs, numerical_ckpt_path=None):
        super().__init__()

        self.configs = configs
        self.pred_len = int(configs.pred_len)
        self.num_channels = int(getattr(configs, "enc_in", 1))
        self.output_dim = self.pred_len * self.num_channels

        self.text_mode = getattr(configs, "text_mode", "direct")
        if self.text_mode not in {"direct", "residual"}:
            raise ValueError(f"[Fit-Fusion] Unsupported text_mode: {self.text_mode}")

        self.freeze_numerical = _to_bool(getattr(configs, "freeze_numerical", False), False)
        self.disable_text = _to_bool(getattr(configs, "disable_text", False), False)

        self.text_hidden = int(getattr(configs, "text_hidden", 128))
        self.num_experts = int(getattr(configs, "num_experts", getattr(configs, "residual_rank", 4)))
        self.residual_style = getattr(configs, "residual_style", "v5")
        if self.residual_style not in {"v3", "v4", "v5"}:
            raise ValueError(f"[Fit-Fusion] Unsupported residual_style: {self.residual_style}")
        self.trend_segments = int(getattr(configs, "trend_segments", 4))
        self.num_feat_dim = int(getattr(configs, "num_feat_dim", max(self.pred_len, 24)))
        self.fusion_hidden = int(getattr(configs, "fusion_hidden", max(self.pred_len * 4, 96)))
        self.fusion_dropout = float(getattr(configs, "fusion_dropout", 0.1))

        self.numerical_model = self._build_numerical_backbone(configs, numerical_ckpt_path)
        self.backbone_dim = int(self.numerical_model.d_model)

        self._init_shared_modules()
        self._init_direct_modules()
        self._init_residual_modules()

        self.aux_loss = None
        if self.disable_text:
            self._freeze_non_numerical_parameters()
        else:
            self._freeze_inactive_branch_parameters()

    def _build_numerical_backbone(self, configs, numerical_ckpt_path):
        numerical_cfg = deepcopy(configs)
        numerical_model = FitNumericalModel(numerical_cfg)

        if numerical_ckpt_path:
            ckpt = torch.load(numerical_ckpt_path, map_location="cpu")
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            numerical_model.load_state_dict(ckpt, strict=False)

        if self.freeze_numerical:
            for param in numerical_model.parameters():
                param.requires_grad = False
            numerical_model.eval()

        return numerical_model

    def _init_shared_modules(self):
        dropout = self.fusion_dropout

        # Shared text adapter for all text-conditioned heads.
        self.text_proj = nn.Sequential(
            nn.LazyLinear(self.text_hidden * 2),
            nn.GELU(),
            nn.LayerNorm(self.text_hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(self.text_hidden * 2, self.text_hidden),
            nn.GELU(),
            nn.LayerNorm(self.text_hidden),
        )

        # Numerical summaries used by both direct and residual.
        self.summary_proj = nn.Sequential(
            nn.Linear(self.backbone_dim, self.fusion_hidden),
            nn.GELU(),
            nn.LayerNorm(self.fusion_hidden),
            nn.Dropout(dropout),
        )
        self.y_num_proj = nn.Sequential(
            nn.Linear(self.output_dim, self.fusion_hidden),
            nn.GELU(),
            nn.LayerNorm(self.fusion_hidden),
            nn.Dropout(dropout),
        )
        self.hist_proj = nn.Sequential(
            nn.Linear(self.num_channels * 5, self.fusion_hidden),
            nn.GELU(),
            nn.LayerNorm(self.fusion_hidden),
            nn.Dropout(dropout),
        )
        self.encoded_proj = nn.Sequential(
            nn.LazyLinear(self.fusion_hidden),
            nn.GELU(),
            nn.LayerNorm(self.fusion_hidden),
            nn.Dropout(dropout),
        )

    def _init_direct_modules(self):
        dropout = self.fusion_dropout
        gate_in_dim = self.text_hidden + self.fusion_hidden * 3

        self.direct_text_expert_head = nn.Sequential(
            nn.Linear(self.text_hidden, self.fusion_hidden),
            nn.GELU(),
            nn.LayerNorm(self.fusion_hidden),
            nn.Dropout(dropout),
            nn.Linear(self.fusion_hidden, self.output_dim * self.num_experts),
        )

        self.direct_text_router_head = nn.Sequential(
            nn.Linear(gate_in_dim, self.fusion_hidden),
            nn.GELU(),
            nn.LayerNorm(self.fusion_hidden),
            nn.Dropout(dropout),
            nn.Linear(self.fusion_hidden, self.pred_len * self.num_experts),
        )

        self.direct_gate_head = nn.Sequential(
            nn.Linear(gate_in_dim, self.fusion_hidden),
            nn.GELU(),
            nn.LayerNorm(self.fusion_hidden),
            nn.Dropout(dropout),
            nn.Linear(self.fusion_hidden, self.output_dim),
        )

    def _init_residual_modules(self):
        dropout = self.fusion_dropout
        basis_in_dim = self.fusion_hidden * 4
        radius_in_dim = self.text_hidden + self.fusion_hidden

        # residual_v3: numerical side only builds candidate residual experts.
        self.residual_v3_expert_head = nn.Sequential(
            nn.Linear(basis_in_dim, self.fusion_hidden),
            nn.GELU(),
            nn.LayerNorm(self.fusion_hidden),
            nn.Dropout(dropout),
            nn.Linear(self.fusion_hidden, self.output_dim * self.num_experts),
        )

        self.residual_v3_text_router_head = nn.Sequential(
            nn.Linear(self.text_hidden, self.fusion_hidden),
            nn.GELU(),
            nn.LayerNorm(self.fusion_hidden),
            nn.Dropout(dropout),
            nn.Linear(self.fusion_hidden, self.pred_len * self.num_experts),
        )

        self.residual_v3_radius_head = nn.Sequential(
            nn.Linear(radius_in_dim, self.fusion_hidden),
            nn.GELU(),
            nn.LayerNorm(self.fusion_hidden),
            nn.Dropout(dropout),
            nn.Linear(self.fusion_hidden, self.output_dim),
        )

        # residual_v4: text reads numerical trend context first, then emits
        # a low-frequency piecewise-constant correction.
        self.residual_v4_num_trend_proj = nn.Sequential(
            nn.Linear(basis_in_dim, self.fusion_hidden),
            nn.GELU(),
            nn.LayerNorm(self.fusion_hidden),
            nn.Dropout(dropout),
        )
        self.residual_v4_text_gamma = nn.Linear(self.text_hidden, self.fusion_hidden)
        self.residual_v4_text_beta = nn.Linear(self.text_hidden, self.fusion_hidden)
        self.residual_v4_trend_norm = nn.LayerNorm(self.fusion_hidden)

        segment_in_dim = self.text_hidden + self.fusion_hidden * 2
        self.residual_v4_segment_head = nn.Sequential(
            nn.Linear(segment_in_dim, self.fusion_hidden),
            nn.GELU(),
            nn.LayerNorm(self.fusion_hidden),
            nn.Dropout(dropout),
            nn.Linear(self.fusion_hidden, self.trend_segments * self.num_channels),
        )
        self.residual_v4_segment_gate_head = nn.Sequential(
            nn.Linear(segment_in_dim, self.fusion_hidden),
            nn.GELU(),
            nn.LayerNorm(self.fusion_hidden),
            nn.Dropout(dropout),
            nn.Linear(self.fusion_hidden, self.trend_segments * self.num_channels),
        )

        # residual_v5: text-aligned segment queries attend numerical patch memory.
        self.residual_v5_trend_memory_proj = nn.Linear(self.backbone_dim, self.fusion_hidden)
        self.residual_v5_key_proj = nn.Linear(self.fusion_hidden, self.fusion_hidden)
        self.residual_v5_value_proj = nn.Linear(self.fusion_hidden, self.fusion_hidden)
        self.residual_v5_text_query_proj = nn.Linear(self.text_hidden, self.fusion_hidden)
        self.residual_v5_segment_queries = nn.Parameter(
            torch.randn(self.trend_segments, self.fusion_hidden) * 0.02
        )
        self.residual_v5_segment_norm = nn.LayerNorm(self.fusion_hidden)

        v5_basis_in_dim = self.fusion_hidden * 3
        self.residual_v5_basis_head = nn.Sequential(
            nn.Linear(v5_basis_in_dim, self.fusion_hidden),
            nn.GELU(),
            nn.LayerNorm(self.fusion_hidden),
            nn.Dropout(dropout),
            nn.Linear(self.fusion_hidden, self.num_channels * self.num_experts),
        )
        self.residual_v5_coeff_head = nn.Sequential(
            nn.Linear(self.text_hidden + self.fusion_hidden, self.fusion_hidden),
            nn.GELU(),
            nn.LayerNorm(self.fusion_hidden),
            nn.Dropout(dropout),
            nn.Linear(self.fusion_hidden, self.num_experts),
        )
        self.residual_v5_radius_head = nn.Sequential(
            nn.Linear(self.text_hidden + self.fusion_hidden * 2, self.fusion_hidden),
            nn.GELU(),
            nn.LayerNorm(self.fusion_hidden),
            nn.Dropout(dropout),
            nn.Linear(self.fusion_hidden, self.num_channels),
        )

    def _freeze_non_numerical_parameters(self):
        for name, param in self.named_parameters():
            if not name.startswith("numerical_model"):
                param.requires_grad = False

    @staticmethod
    def _freeze_module(module):
        for param in module.parameters():
            param.requires_grad = False

    def _freeze_inactive_branch_parameters(self):
        if self.text_mode == "direct":
            self._freeze_module(self.encoded_proj)
            self._freeze_module(self.residual_v3_expert_head)
            self._freeze_module(self.residual_v3_text_router_head)
            self._freeze_module(self.residual_v3_radius_head)
            self._freeze_module(self.residual_v4_num_trend_proj)
            self._freeze_module(self.residual_v4_text_gamma)
            self._freeze_module(self.residual_v4_text_beta)
            self._freeze_module(self.residual_v4_trend_norm)
            self._freeze_module(self.residual_v4_segment_head)
            self._freeze_module(self.residual_v4_segment_gate_head)
            self._freeze_module(self.residual_v5_trend_memory_proj)
            self._freeze_module(self.residual_v5_key_proj)
            self._freeze_module(self.residual_v5_value_proj)
            self._freeze_module(self.residual_v5_text_query_proj)
            self._freeze_module(self.residual_v5_segment_norm)
            self._freeze_module(self.residual_v5_basis_head)
            self._freeze_module(self.residual_v5_coeff_head)
            self._freeze_module(self.residual_v5_radius_head)
            self.residual_v5_segment_queries.requires_grad = False
            return

        self._freeze_module(self.direct_text_expert_head)
        self._freeze_module(self.direct_text_router_head)
        self._freeze_module(self.direct_gate_head)
        if self.residual_style == "v3":
            self._freeze_module(self.residual_v4_num_trend_proj)
            self._freeze_module(self.residual_v4_text_gamma)
            self._freeze_module(self.residual_v4_text_beta)
            self._freeze_module(self.residual_v4_trend_norm)
            self._freeze_module(self.residual_v4_segment_head)
            self._freeze_module(self.residual_v4_segment_gate_head)
            self._freeze_module(self.residual_v5_trend_memory_proj)
            self._freeze_module(self.residual_v5_key_proj)
            self._freeze_module(self.residual_v5_value_proj)
            self._freeze_module(self.residual_v5_text_query_proj)
            self._freeze_module(self.residual_v5_segment_norm)
            self._freeze_module(self.residual_v5_basis_head)
            self._freeze_module(self.residual_v5_coeff_head)
            self._freeze_module(self.residual_v5_radius_head)
            self.residual_v5_segment_queries.requires_grad = False
            return

        self._freeze_module(self.residual_v3_expert_head)
        self._freeze_module(self.residual_v3_text_router_head)
        self._freeze_module(self.residual_v3_radius_head)
        if self.residual_style == "v4":
            self._freeze_module(self.residual_v5_trend_memory_proj)
            self._freeze_module(self.residual_v5_key_proj)
            self._freeze_module(self.residual_v5_value_proj)
            self._freeze_module(self.residual_v5_text_query_proj)
            self._freeze_module(self.residual_v5_segment_norm)
            self._freeze_module(self.residual_v5_basis_head)
            self._freeze_module(self.residual_v5_coeff_head)
            self._freeze_module(self.residual_v5_radius_head)
            self.residual_v5_segment_queries.requires_grad = False
            return

        self._freeze_module(self.residual_v4_num_trend_proj)
        self._freeze_module(self.residual_v4_text_gamma)
        self._freeze_module(self.residual_v4_text_beta)
        self._freeze_module(self.residual_v4_trend_norm)
        self._freeze_module(self.residual_v4_segment_head)
        self._freeze_module(self.residual_v4_segment_gate_head)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_numerical:
            self.numerical_model.eval()
        return self

    @staticmethod
    def _sanitize_caption_emb(caption_emb, device):
        if not isinstance(caption_emb, torch.Tensor):
            caption_emb = torch.tensor(caption_emb)

        caption_emb = caption_emb.to(device=device)
        if caption_emb.dim() == 1:
            caption_emb = caption_emb.unsqueeze(0)
        if caption_emb.dtype != torch.float32:
            caption_emb = caption_emb.float()
        return caption_emb

    @staticmethod
    def _reshape_output(flat_output, batch_size, pred_len, num_channels, dtype):
        return flat_output.to(dtype=dtype).reshape(batch_size, pred_len, num_channels)

    @staticmethod
    def _expand_piecewise_constant(segment_values, target_len):
        batch_size, num_segments, num_channels = segment_values.shape
        base = target_len // num_segments
        remainder = target_len % num_segments

        expanded = []
        for idx in range(num_segments):
            seg_len = base + (1 if idx < remainder else 0)
            if seg_len <= 0:
                continue
            expanded.append(segment_values[:, idx : idx + 1, :].expand(batch_size, seg_len, num_channels))

        return torch.cat(expanded, dim=1)

    @staticmethod
    def _compute_history_features(x_enc):
        last_value = x_enc[:, -1, :]
        mean_value = x_enc.mean(dim=1)
        std_value = x_enc.std(dim=1, unbiased=False)
        slope_value = (x_enc[:, -1, :] - x_enc[:, 0, :]) / max(x_enc.shape[1] - 1, 1)
        range_value = x_enc.max(dim=1).values - x_enc.min(dim=1).values

        return {
            "last": last_value,
            "mean": mean_value,
            "std": std_value,
            "slope": slope_value,
            "range": range_value,
            "flat": torch.cat([last_value, mean_value, std_value, slope_value, range_value], dim=-1),
        }

    def _extract_numerical_features(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        city_id,
        gender_id,
        age_id,
        element_id,
    ):
        if self.freeze_numerical:
            self.numerical_model.eval()
            with torch.no_grad():
                return self.numerical_model.extract_features(
                    x_enc,
                    x_mark_enc,
                    x_dec,
                    x_mark_dec,
                    city_id,
                    gender_id,
                    age_id,
                    element_id,
                    caption_emb=None,
                )

        return self.numerical_model.extract_features(
            x_enc,
            x_mark_enc,
            x_dec,
            x_mark_dec,
            city_id,
            gender_id,
            age_id,
            element_id,
            caption_emb=None,
        )

    def _build_numerical_contexts(self, numerical_features, history_features):
        y_num = numerical_features["forecast"]
        encoded_tokens = numerical_features["encoded_tokens"]
        summary_state = numerical_features["summary_state"]
        batch_size = y_num.shape[0]

        y_num_ctx = self.y_num_proj(y_num.reshape(batch_size, self.output_dim))
        summary_ctx = self.summary_proj(summary_state)
        hist_ctx = self.hist_proj(history_features["flat"].float())

        # Keep encoded token information separate from pooled summary state.
        encoded_summary = encoded_tokens.mean(dim=2).reshape(batch_size, -1)
        encoded_ctx = self.encoded_proj(encoded_summary)

        return {
            "y_num_ctx": y_num_ctx,
            "summary_ctx": summary_ctx,
            "hist_ctx": hist_ctx,
            "encoded_ctx": encoded_ctx,
        }

    def _apply_direct(self, text_ctx, numerical_features, numerical_contexts):
        y_num = numerical_features["forecast"]
        batch_size = y_num.shape[0]

        gate_input = torch.cat(
            [
                text_ctx,
                numerical_contexts["summary_ctx"],
                numerical_contexts["y_num_ctx"],
                numerical_contexts["hist_ctx"],
            ],
            dim=-1,
        )
        direct_experts = self.direct_text_expert_head(text_ctx).reshape(
            batch_size,
            self.pred_len,
            self.num_channels,
            self.num_experts,
        )
        direct_route = torch.softmax(
            self.direct_text_router_head(gate_input).reshape(batch_size, self.pred_len, self.num_experts),
            dim=-1,
        )
        y_text = torch.einsum("blck,blk->blc", direct_experts, direct_route.to(dtype=direct_experts.dtype))
        gate = torch.sigmoid(
            self._reshape_output(
                self.direct_gate_head(gate_input),
                batch_size,
                self.pred_len,
                self.num_channels,
                y_num.dtype,
            )
        )

        self.aux_loss = None
        return (1.0 - gate) * y_num + gate * y_text

    def _apply_residual_v3(self, text_ctx, numerical_features, numerical_contexts):
        y_num = numerical_features["forecast"]
        batch_size = y_num.shape[0]

        basis_input = torch.cat(
            [
                numerical_contexts["encoded_ctx"],
                numerical_contexts["summary_ctx"],
                numerical_contexts["y_num_ctx"],
                numerical_contexts["hist_ctx"],
            ],
            dim=-1,
        )
        residual_experts = self.residual_v3_expert_head(basis_input).reshape(
            batch_size,
            self.pred_len,
            self.num_channels,
            self.num_experts,
        )

        residual_route = torch.softmax(
            self.residual_v3_text_router_head(text_ctx).reshape(
                batch_size,
                self.pred_len,
                self.num_experts,
            ),
            dim=-1,
        )
        delta_raw = torch.einsum(
            "blck,blk->blc",
            residual_experts,
            residual_route.to(dtype=residual_experts.dtype),
        )

        radius_input = torch.cat([text_ctx, numerical_contexts["hist_ctx"]], dim=-1)
        radius = torch.sigmoid(
            self._reshape_output(
                self.residual_v3_radius_head(radius_input),
                batch_size,
                self.pred_len,
                self.num_channels,
                y_num.dtype,
            )
        )

        delta = radius * torch.tanh(delta_raw.to(dtype=y_num.dtype))
        self.aux_loss = None
        return y_num + delta

    def _apply_residual_v4(self, text_ctx, numerical_features, numerical_contexts):
        y_num = numerical_features["forecast"]
        batch_size = y_num.shape[0]

        trend_input = torch.cat(
            [
                numerical_contexts["encoded_ctx"],
                numerical_contexts["summary_ctx"],
                numerical_contexts["y_num_ctx"],
                numerical_contexts["hist_ctx"],
            ],
            dim=-1,
        )
        num_trend_ctx = self.residual_v4_num_trend_proj(trend_input)
        gamma = torch.tanh(self.residual_v4_text_gamma(text_ctx))
        beta = self.residual_v4_text_beta(text_ctx)
        aligned_trend_ctx = self.residual_v4_trend_norm(num_trend_ctx * (1.0 + gamma) + beta)

        segment_input = torch.cat(
            [
                aligned_trend_ctx,
                text_ctx,
                numerical_contexts["hist_ctx"],
            ],
            dim=-1,
        )
        delta_segments = self.residual_v4_segment_head(segment_input).reshape(
            batch_size,
            self.trend_segments,
            self.num_channels,
        )
        segment_gate = torch.sigmoid(
            self.residual_v4_segment_gate_head(segment_input).reshape(
                batch_size,
                self.trend_segments,
                self.num_channels,
            )
        )

        delta_segments = segment_gate * torch.tanh(delta_segments.to(dtype=y_num.dtype))
        delta = self._expand_piecewise_constant(delta_segments, self.pred_len).to(dtype=y_num.dtype)
        self.aux_loss = None
        return y_num + delta

    def _apply_residual_v5(self, text_ctx, numerical_features, numerical_contexts):
        y_num = numerical_features["forecast"]
        encoded_tokens = numerical_features["encoded_tokens"]
        batch_size = y_num.shape[0]

        # [B, patch_count, d_model] -> [B, patch_count, fusion_hidden]
        trend_memory = encoded_tokens.mean(dim=1)
        trend_memory = self.residual_v5_trend_memory_proj(trend_memory)
        keys = self.residual_v5_key_proj(trend_memory)
        values = self.residual_v5_value_proj(trend_memory)

        text_query = self.residual_v5_text_query_proj(text_ctx).unsqueeze(1)
        segment_queries = self.residual_v5_segment_queries.unsqueeze(0) + text_query
        attn_scores = torch.einsum("bsh,bph->bsp", segment_queries, keys) / (self.fusion_hidden ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        segment_ctx = torch.einsum("bsp,bph->bsh", attn_weights, values)
        segment_ctx = self.residual_v5_segment_norm(segment_ctx)

        summary_seg = numerical_contexts["summary_ctx"].unsqueeze(1).expand(batch_size, self.trend_segments, -1)
        hist_seg = numerical_contexts["hist_ctx"].unsqueeze(1).expand(batch_size, self.trend_segments, -1)
        text_seg = text_ctx.unsqueeze(1).expand(batch_size, self.trend_segments, -1)

        basis_input = torch.cat([segment_ctx, summary_seg, hist_seg], dim=-1)
        trend_basis = self.residual_v5_basis_head(basis_input).reshape(
            batch_size,
            self.trend_segments,
            self.num_channels,
            self.num_experts,
        )

        coeff_input = torch.cat([text_seg, segment_ctx], dim=-1)
        coeff = torch.softmax(self.residual_v5_coeff_head(coeff_input), dim=-1)
        delta_raw = torch.einsum("bsck,bsk->bsc", trend_basis, coeff.to(dtype=trend_basis.dtype))

        radius_input = torch.cat([text_seg, segment_ctx, hist_seg], dim=-1)
        radius = torch.sigmoid(self.residual_v5_radius_head(radius_input)).to(dtype=y_num.dtype)

        delta_segments = radius * torch.tanh(delta_raw.to(dtype=y_num.dtype))
        delta = self._expand_piecewise_constant(delta_segments, self.pred_len).to(dtype=y_num.dtype)
        self.aux_loss = None
        return y_num + delta

    def _apply_residual(self, text_ctx, numerical_features, numerical_contexts):
        if self.residual_style == "v3":
            return self._apply_residual_v3(text_ctx, numerical_features, numerical_contexts)
        if self.residual_style == "v4":
            return self._apply_residual_v4(text_ctx, numerical_features, numerical_contexts)
        return self._apply_residual_v5(text_ctx, numerical_features, numerical_contexts)

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        city_id=None,
        gender_id=None,
        age_id=None,
        element_id=None,
        caption_emb=None,
        element_ids=None,
        group_ids=None,
        norms=None,
    ):
        del element_ids, group_ids, norms

        if caption_emb is None:
            raise ValueError("[Fit-Fusion] caption_emb must be provided")

        caption_emb = self._sanitize_caption_emb(caption_emb, device=x_enc.device)
        numerical_features = self._extract_numerical_features(
            x_enc,
            x_mark_enc,
            x_dec,
            x_mark_dec,
            city_id,
            gender_id,
            age_id,
            element_id,
        )
        y_num = numerical_features["forecast"]

        if self.disable_text:
            self.aux_loss = None
            return y_num

        history_features = self._compute_history_features(x_enc)
        numerical_contexts = self._build_numerical_contexts(numerical_features, history_features)
        text_ctx = self.text_proj(caption_emb)

        if self.text_mode == "direct":
            return self._apply_direct(text_ctx, numerical_features, numerical_contexts)
        return self._apply_residual(text_ctx, numerical_features, numerical_contexts)
