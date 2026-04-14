"""
FIT fusion model for the active 0414 line.

Design goals:
- Keep the numerical backbone unchanged.
- Keep an explicit numerical skip in `direct`.
- Force residual correction to pass through text-controlled coefficients.
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
        Numerical and text streams each produce a forecast.
        Final prediction uses an explicit step-wise gate:
        y_final = (1 - gate) * y_num + gate * y_text

    `residual`
        Numerical side only produces residual bases.
        Text side produces basis coefficients and correction radius.
        Final correction must pass through text:
        y_final = y_num + radius * tanh(sum_k basis_k * coeff_k)
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
        self.residual_rank = int(getattr(configs, "residual_rank", 8))
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

        self.direct_text_head = nn.Sequential(
            nn.Linear(self.text_hidden, self.fusion_hidden),
            nn.GELU(),
            nn.LayerNorm(self.fusion_hidden),
            nn.Dropout(dropout),
            nn.Linear(self.fusion_hidden, self.output_dim),
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

        # Numerical side only builds residual bases, not delta directly.
        self.residual_basis_head = nn.Sequential(
            nn.Linear(basis_in_dim, self.fusion_hidden),
            nn.GELU(),
            nn.LayerNorm(self.fusion_hidden),
            nn.Dropout(dropout),
            nn.Linear(self.fusion_hidden, self.output_dim * self.residual_rank),
        )

        self.residual_text_coeff_head = nn.Sequential(
            nn.Linear(self.text_hidden, self.fusion_hidden),
            nn.GELU(),
            nn.LayerNorm(self.fusion_hidden),
            nn.Dropout(dropout),
            nn.Linear(self.fusion_hidden, self.pred_len * self.residual_rank),
        )

        self.residual_radius_head = nn.Sequential(
            nn.Linear(radius_in_dim, self.fusion_hidden),
            nn.GELU(),
            nn.LayerNorm(self.fusion_hidden),
            nn.Dropout(dropout),
            nn.Linear(self.fusion_hidden, self.output_dim),
        )

    def _freeze_non_numerical_parameters(self):
        for name, param in self.named_parameters():
            if not name.startswith("numerical_model"):
                param.requires_grad = False

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

        y_text = self._reshape_output(
            self.direct_text_head(text_ctx),
            batch_size,
            self.pred_len,
            self.num_channels,
            y_num.dtype,
        )

        gate_input = torch.cat(
            [
                text_ctx,
                numerical_contexts["summary_ctx"],
                numerical_contexts["y_num_ctx"],
                numerical_contexts["hist_ctx"],
            ],
            dim=-1,
        )
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

    def _apply_residual(self, text_ctx, numerical_features, numerical_contexts):
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
        residual_basis = self.residual_basis_head(basis_input).reshape(
            batch_size,
            self.pred_len,
            self.num_channels,
            self.residual_rank,
        )

        text_coeff = self.residual_text_coeff_head(text_ctx).reshape(
            batch_size,
            self.pred_len,
            self.residual_rank,
        )
        delta_raw = torch.einsum("blcr,blr->blc", residual_basis, text_coeff.to(dtype=residual_basis.dtype))

        radius_input = torch.cat([text_ctx, numerical_contexts["hist_ctx"]], dim=-1)
        radius = torch.sigmoid(
            self._reshape_output(
                self.residual_radius_head(radius_input),
                batch_size,
                self.pred_len,
                self.num_channels,
                y_num.dtype,
            )
        )

        delta = radius * torch.tanh(delta_raw.to(dtype=y_num.dtype))
        self.aux_loss = None
        return y_num + delta

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
