"""
FIT 融合模型。

当前支持两套 fusion 头：

- modern:
  文本先调制数值 backbone 的中间特征，再走 direct / residual 两条路径。
- legacy:
  恢复旧版浅融合头。
  - direct: y_final = (1 - w) * y_num + w * y_text
  - residual: y_final = y_num + gain * delta_y

共同约束：
- 数值主干始终复用 `Model_Fit_Num_With_Meta`
- 文本输入始终是预处理好的 `caption_emb`
- `disable_text=True` 时统一退化为纯数值预测，并冻结所有非数值参数
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
    """FIT fusion 统一入口，内部按 `fusion_version` 选择 modern / legacy。"""

    def __init__(self, configs, numerical_ckpt_path=None):
        super().__init__()

        self.configs = configs
        self.pred_len = configs.pred_len
        self.num_channels = getattr(configs, "enc_in", 1)
        self.output_dim = self.pred_len * self.num_channels

        self.fusion_version = getattr(configs, "fusion_version", "modern")
        if self.fusion_version not in {"modern", "legacy"}:
            raise ValueError(f"[Fit-Fusion] Unsupported fusion_version: {self.fusion_version}")

        self.text_mode = getattr(configs, "text_mode", "direct")
        if self.text_mode not in {"direct", "residual"}:
            raise ValueError(f"[Fit-Fusion] Unsupported text_mode: {self.text_mode}")

        self.freeze_numerical = _to_bool(getattr(configs, "freeze_numerical", False), False)
        self.disable_text = _to_bool(getattr(configs, "disable_text", False), False)

        self.num_feat_dim = int(getattr(configs, "num_feat_dim", max(self.pred_len, 24)))
        self.fusion_hidden = int(getattr(configs, "fusion_hidden", max(self.pred_len * 4, 96)))
        self.fusion_dropout = float(getattr(configs, "fusion_dropout", 0.1))

        # legacy 兼容参数。modern 分支会忽略这些开关。
        self.llm_dim = int(getattr(configs, "llm_dim", 768))
        self.direct_w_mode = getattr(configs, "direct_w_mode", "learned")
        self.direct_w_fixed = float(getattr(configs, "direct_w_fixed", 0.5))
        self.use_vol_prior = _to_bool(getattr(configs, "use_vol_prior", True), True)
        self.force_gain = float(getattr(configs, "force_gain", -1.0))
        self.delta_scale = float(getattr(configs, "delta_scale", 1.0))
        self.alpha = float(getattr(configs, "alpha", 1.0))

        self.numerical_model = self._build_numerical_backbone(configs, numerical_ckpt_path)
        d_model = self.numerical_model.d_model

        if self.fusion_version == "modern":
            self._init_modern_modules(d_model)
        else:
            self._init_legacy_modules()

        self.aux_loss = None

        if self.disable_text:
            self._freeze_non_numerical_parameters()

    def _build_numerical_backbone(self, configs, numerical_ckpt_path):
        num_cfg = deepcopy(configs)
        numerical_model = FitNumericalModel(num_cfg)

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

    def _init_modern_modules(self, d_model):
        context_dim = self.num_feat_dim
        shared_hidden = self.fusion_hidden
        dropout = self.fusion_dropout

        # modern 分支允许预处理 embedding 维度自动适配。
        self.text_adapter = nn.Sequential(
            nn.LazyLinear(context_dim * 2),
            nn.GELU(),
            nn.LayerNorm(context_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(context_dim * 2, context_dim),
        )

        self.text_to_gamma = nn.Linear(context_dim, d_model)
        self.text_to_beta = nn.Linear(context_dim, d_model)
        self.latent_norm = nn.LayerNorm(d_model)

        self.summary_proj = nn.Sequential(
            nn.Linear(d_model * 2, context_dim * 2),
            nn.GELU(),
            nn.LayerNorm(context_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(context_dim * 2, context_dim),
        )

        self.y_num_proj = nn.Sequential(
            nn.Linear(self.output_dim, context_dim * 2),
            nn.GELU(),
            nn.LayerNorm(context_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(context_dim * 2, context_dim),
        )

        # 历史统计：last / mean / std / slope / range
        self.hist_proj = nn.Sequential(
            nn.Linear(self.num_channels * 5, context_dim * 2),
            nn.GELU(),
            nn.LayerNorm(context_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(context_dim * 2, context_dim),
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(context_dim * 4, shared_hidden),
            nn.GELU(),
            nn.LayerNorm(shared_hidden),
            nn.Dropout(dropout),
            nn.Linear(shared_hidden, shared_hidden),
            nn.GELU(),
            nn.LayerNorm(shared_hidden),
            nn.Dropout(dropout),
        )

        self.direct_head = nn.Sequential(
            nn.Linear(shared_hidden, shared_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(shared_hidden, self.output_dim),
        )

        self.residual_head = nn.Sequential(
            nn.Linear(shared_hidden, shared_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(shared_hidden, self.output_dim),
        )

        # modern residual 用历史波动为纠偏幅度做上界约束。
        self.radius_head = nn.Sequential(
            nn.Linear(shared_hidden, shared_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(shared_hidden, self.output_dim),
        )

    def _init_legacy_modules(self):
        dropout = self.fusion_dropout
        hidden = self.fusion_hidden

        self.text_pred_proj = nn.Sequential(
            nn.Linear(self.llm_dim, self.pred_len * 2),
            nn.ReLU(),
            nn.Linear(self.pred_len * 2, self.pred_len),
        )
        self.text_weight = nn.Parameter(torch.ones(self.num_channels) * -1.0)

        self.num_feat_proj = nn.Sequential(
            nn.Linear(self.output_dim, self.num_feat_dim),
            nn.GELU(),
            nn.LayerNorm(self.num_feat_dim),
            nn.Dropout(dropout),
        )

        self.fusion_in_dim = self.llm_dim + self.num_feat_dim + 1

        self.delta_proj = nn.Sequential(
            nn.Linear(self.fusion_in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.pred_len),
        )

        self.gain_net = nn.Sequential(
            nn.Linear(self.fusion_in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.pred_len),
        )

        # 旧版 residual 初始化得很保守，避免一开始就把数值主预测拉偏。
        nn.init.zeros_(self.gain_net[-1].weight)
        nn.init.constant_(self.gain_net[-1].bias, -4.0)
        nn.init.zeros_(self.delta_proj[-1].weight)
        nn.init.zeros_(self.delta_proj[-1].bias)

    def _freeze_non_numerical_parameters(self):
        for name, param in self.named_parameters():
            if not name.startswith("numerical_model"):
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_numerical:
            # 外层即使切回 train，冻结的数值 backbone 也保持 eval。
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

    def _extract_numerical_features(self, x_enc, x_mark_enc, x_dec, x_mark_dec, city_id, gender_id, age_id, element_id):
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

    def _modulate_latent(self, encoded_tokens, text_ctx):
        gamma = torch.tanh(self.text_to_gamma(text_ctx)).unsqueeze(1).unsqueeze(1)
        beta = self.text_to_beta(text_ctx).unsqueeze(1).unsqueeze(1)
        return self.latent_norm(encoded_tokens * (1.0 + gamma) + beta)

    def _build_modern_shared_context(self, caption_emb, numerical_features, x_enc):
        y_num = numerical_features["forecast"]
        encoded_tokens = numerical_features["encoded_tokens"]
        summary_state = numerical_features["summary_state"]

        batch_size = y_num.shape[0]
        history_features = self._compute_history_features(x_enc)

        text_ctx = self.text_adapter(caption_emb)
        modulated_tokens = self._modulate_latent(encoded_tokens, text_ctx)
        modulated_summary = modulated_tokens.mean(dim=(1, 2))

        summary_ctx = self.summary_proj(torch.cat([summary_state, modulated_summary], dim=-1))
        y_num_ctx = self.y_num_proj(y_num.reshape(batch_size, self.output_dim))
        hist_ctx = self.hist_proj(history_features["flat"].float())

        shared_context = self.fusion_mlp(torch.cat([text_ctx, summary_ctx, y_num_ctx, hist_ctx], dim=-1))
        return shared_context, y_num, history_features

    def _apply_modern_direct(self, shared_context, y_num):
        batch_size = y_num.shape[0]
        self.aux_loss = None
        return self._reshape_output(
            self.direct_head(shared_context),
            batch_size,
            self.pred_len,
            self.num_channels,
            y_num.dtype,
        )

    def _apply_modern_residual(self, shared_context, y_num, history_features):
        batch_size = y_num.shape[0]

        raw_delta = self._reshape_output(
            self.residual_head(shared_context),
            batch_size,
            self.pred_len,
            self.num_channels,
            y_num.dtype,
        )
        radius_gate = torch.sigmoid(
            self._reshape_output(
                self.radius_head(shared_context),
                batch_size,
                self.pred_len,
                self.num_channels,
                y_num.dtype,
            )
        )

        history_std = history_features["std"].unsqueeze(1)
        history_range = history_features["range"].unsqueeze(1)
        radius = (history_std + radius_gate * history_range).clamp_min(1e-4)
        delta = torch.tanh(raw_delta) * radius

        self.aux_loss = None
        return y_num + delta

    def _build_w(self, channel_count, device, dtype):
        if self.direct_w_mode == "fixed":
            w_scalar = max(0.0, min(1.0, float(self.direct_w_fixed)))
            return torch.full((1, 1, channel_count), w_scalar, device=device, dtype=dtype)

        w = torch.sigmoid(self.text_weight).to(device=device, dtype=dtype)
        if w.numel() == 1:
            return w.view(1, 1, 1).expand(1, 1, channel_count)
        return w[:channel_count].view(1, 1, channel_count)

    def _pad_or_trim_y_num_flat(self, y_num_flat):
        if y_num_flat.shape[1] == self.output_dim:
            return y_num_flat
        if y_num_flat.shape[1] > self.output_dim:
            return y_num_flat[:, : self.output_dim]
        pad_size = self.output_dim - y_num_flat.shape[1]
        return torch.cat([y_num_flat, y_num_flat.new_zeros(y_num_flat.shape[0], pad_size)], dim=1)

    def _build_legacy_vol(self, x_enc):
        batch_size = x_enc.shape[0]
        sample_stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        base_stdev = sample_stdev.mean(dim=2, keepdim=True)
        return base_stdev.view(batch_size, 1)

    def _apply_legacy_direct(self, caption_emb, y_num):
        batch_size, _, channel_count = y_num.shape

        y_text = self.text_pred_proj(caption_emb).to(dtype=y_num.dtype).unsqueeze(-1)
        if channel_count > 1:
            y_text = y_text.expand(-1, -1, channel_count)

        w = self._build_w(channel_count, device=y_num.device, dtype=y_num.dtype)
        self.aux_loss = None
        return (1.0 - w) * y_num + w * y_text

    def _apply_legacy_residual(self, caption_emb, y_num, x_enc):
        batch_size, _, channel_count = y_num.shape
        vol = self._build_legacy_vol(x_enc)

        y_num_flat = self._pad_or_trim_y_num_flat(y_num.reshape(batch_size, self.pred_len * channel_count))
        num_feat = self.num_feat_proj(y_num_flat.detach())

        vol_input = vol.float() if self.use_vol_prior else torch.zeros_like(vol, dtype=torch.float32)
        fusion_in = torch.cat([caption_emb, num_feat.float(), vol_input], dim=-1)

        delta_y = self.delta_proj(fusion_in) * self.delta_scale

        if self.force_gain >= 0:
            gain = torch.full(
                (batch_size, self.pred_len),
                self.force_gain,
                device=y_num.device,
                dtype=delta_y.dtype,
            )
        else:
            gain = torch.sigmoid(self.gain_net(fusion_in))

        delta_y = delta_y.to(dtype=y_num.dtype).unsqueeze(-1)
        gain = gain.to(dtype=y_num.dtype).unsqueeze(-1)

        if channel_count > 1:
            delta_y = delta_y.expand(-1, -1, channel_count)
            gain = gain.expand(-1, -1, channel_count)

        self.aux_loss = None
        return y_num + gain * delta_y

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

        if self.fusion_version == "modern":
            shared_context, y_num, history_features = self._build_modern_shared_context(
                caption_emb, numerical_features, x_enc
            )
            if self.text_mode == "direct":
                return self._apply_modern_direct(shared_context, y_num)
            return self._apply_modern_residual(shared_context, y_num, history_features)

        if self.text_mode == "direct":
            return self._apply_legacy_direct(caption_emb, y_num)
        return self._apply_legacy_residual(caption_emb, y_num, x_enc)
