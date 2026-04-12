"""
FIT 融合模型。

当前文本流继续使用预处理好的 `caption_emb`，不在训练时跑 raw text encoder。

和旧版的差别：
- 文本不再只在输出端接一个很浅的 head
- 文本先调制数值 backbone 的中间特征，再进入 direct / residual 两条路径
- residual 不再依赖 beta_delta / force_gain 这类补丁式超参
"""

from copy import deepcopy

import torch
import torch.nn as nn

from models.model_fit_num_with_meta import Model_Fit_Num_With_Meta as FitNumericalModel


def _to_bool(x, default=False):
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return x != 0
    if isinstance(x, str):
        normalized = x.strip().lower()
        if normalized in ["1", "true", "yes", "y", "t", "on"]:
            return True
        if normalized in ["0", "false", "no", "n", "f", "off"]:
            return False
        return default
    return bool(x)


class Model_Fit_Fusion(nn.Module):
    """
    FIT 融合模型。

    数值流：
    - `Model_Fit_Num_With_Meta`

    文本流：
    - 预处理好的 `caption_emb`

    两种模式：
    - `direct`：文本条件化后的中间特征直接输出最终预测
    - `residual`：文本条件化后的中间特征输出“有界纠偏量”
    """

    def __init__(self, configs, numerical_ckpt_path=None):
        super().__init__()

        self.configs = configs
        self.pred_len = configs.pred_len
        self.num_channels = getattr(configs, "enc_in", 1)
        self.output_dim = self.pred_len * self.num_channels

        self.freeze_numerical = _to_bool(getattr(configs, "freeze_numerical", False), False)
        self.disable_text = _to_bool(getattr(configs, "disable_text", False), False)
        self.text_mode = getattr(configs, "text_mode", "residual")

        context_dim = int(getattr(configs, "num_feat_dim", max(self.pred_len, 24)))
        shared_hidden = int(getattr(configs, "fusion_hidden", max(self.pred_len * 4, 96)))
        dropout = float(getattr(configs, "fusion_dropout", 0.1))

        # =========================================================
        # A. 数值 backbone
        # =========================================================
        num_cfg = deepcopy(configs)
        self.numerical_model = FitNumericalModel(num_cfg)

        if numerical_ckpt_path:
            ckpt = torch.load(numerical_ckpt_path, map_location="cpu")
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            self.numerical_model.load_state_dict(ckpt, strict=False)

        if self.freeze_numerical:
            for param in self.numerical_model.parameters():
                param.requires_grad = False
            self.numerical_model.eval()

        d_model = self.numerical_model.d_model

        # =========================================================
        # B. 文本适配器
        # =========================================================
        # 不再写死 llm_dim；首层用 LazyLinear 自动适配预处理 embedding 维度。
        self.text_adapter = nn.Sequential(
            nn.LazyLinear(context_dim * 2),
            nn.GELU(),
            nn.LayerNorm(context_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(context_dim * 2, context_dim),
        )

        # 文本生成一组条件调制参数，直接作用到数值 latent。
        self.text_to_gamma = nn.Linear(context_dim, d_model)
        self.text_to_beta = nn.Linear(context_dim, d_model)
        self.latent_norm = nn.LayerNorm(d_model)

        # =========================================================
        # C. 数值摘要、历史统计、预测摘要
        # =========================================================
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

        # =========================================================
        # D. direct / residual 头
        # =========================================================
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

        # residual 的幅度不再靠额外正则，而是由历史波动上界控制。
        self.radius_head = nn.Sequential(
            nn.Linear(shared_hidden, shared_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(shared_hidden, self.output_dim),
        )

        self.aux_loss = None

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_numerical:
            # 即使外层 exp 调了 model.train()，冻结的数值 backbone 也保持 eval。
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

    def _build_shared_context(self, caption_emb, numerical_features, x_enc):
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

    def _apply_direct_fusion(self, shared_context, y_num):
        batch_size = y_num.shape[0]
        self.aux_loss = None
        return self._reshape_output(
            self.direct_head(shared_context),
            batch_size,
            self.pred_len,
            self.num_channels,
            y_num.dtype,
        )

    def _apply_residual_fusion(self, shared_context, y_num, history_features):
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

        # 纠偏幅度直接受历史波动约束，不再依赖额外正则项。
        history_std = history_features["std"].unsqueeze(1)
        history_range = history_features["range"].unsqueeze(1)
        radius = (history_std + radius_gate * history_range).clamp_min(1e-4)
        delta = torch.tanh(raw_delta) * radius

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
            x_enc, x_mark_enc, x_dec, x_mark_dec, city_id, gender_id, age_id, element_id
        )
        y_num = numerical_features["forecast"]

        if self.disable_text:
            self.aux_loss = None
            return y_num

        shared_context, y_num, history_features = self._build_shared_context(caption_emb, numerical_features, x_enc)

        if self.text_mode == "direct":
            # direct：文本先调制数值 latent，再由 direct head 直接输出最终预测。
            return self._apply_direct_fusion(shared_context, y_num)

        if self.text_mode == "residual":
            # residual：文本只输出“有边界的纠偏量”，主预测仍然锚定在 y_num。
            return self._apply_residual_fusion(shared_context, y_num, history_features)

        raise ValueError(f"[Fit-Fusion] Unknown text_mode: {self.text_mode}")
