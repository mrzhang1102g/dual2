"""Geo 融合模型。

Geo 侧保留数值主干不变，文本分支支持两种融合方式：
- `direct`：文本直接给出一条预测，再与数值流做加权混合；
- `residual`：文本只产出修正量，在数值预测上做增量调整。
"""

from copy import deepcopy

import torch
import torch.nn as nn

from models.model_geo_num_with_meta import Model_Geo_Num_With_Meta as GeoNumericalModel


class Model_Geo_Fusion(nn.Module):
    """GeoStyle 融合模型主体。"""

    def __init__(self, configs, numerical_ckpt_path=None):
        super().__init__()

        self.configs = configs
        self.pred_len = configs.pred_len
        self.num_channels = getattr(configs, "enc_in", 1)

        self.freeze_numerical = getattr(configs, "freeze_numerical", False)
        self.disable_text = getattr(configs, "disable_text", False)
        self.text_mode = getattr(configs, "text_mode", "residual")

        # residual 模式下用于控制修正幅度。
        self.use_vol_prior = bool(getattr(configs, "use_vol_prior", True))
        self.force_gain = float(getattr(configs, "force_gain", -1))

        # direct 模式下用于控制文本流权重。
        self.direct_w_mode = getattr(configs, "direct_w_mode", "learned")
        self.direct_w_fixed = float(getattr(configs, "direct_w_fixed", 0.5))

        # 数值主干沿用带元数据的 Geo 模型。
        num_cfg = deepcopy(configs)
        num_cfg.use_element = True
        num_cfg.use_group = True
        num_cfg.num_element = getattr(configs, "num_element", 46)
        num_cfg.num_group = getattr(configs, "num_group", 44)

        self.numerical_model = GeoNumericalModel(num_cfg)

        if numerical_ckpt_path is not None:
            ckpt = torch.load(numerical_ckpt_path, map_location="cpu")
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            self.numerical_model.load_state_dict(ckpt, strict=False)

        if self.freeze_numerical:
            for param in self.numerical_model.parameters():
                param.requires_grad = False
            self.numerical_model.eval()

        self.llm_dim = getattr(configs, "llm_dim", 768)

        self.text_pred_proj = nn.Sequential(
            nn.Linear(self.llm_dim, self.pred_len * 2),
            nn.ReLU(),
            nn.Linear(self.pred_len * 2, self.pred_len),
        )

        self.delta_proj = nn.Sequential(
            nn.Linear(self.llm_dim, self.pred_len * 2),
            nn.ReLU(),
            nn.LayerNorm(self.pred_len * 2),
            nn.Linear(self.pred_len * 2, self.pred_len),
        )

        self.text_weight = nn.Parameter(torch.ones(self.num_channels) * -1.0)
        self.alpha = float(getattr(configs, "alpha", 1.0))
        self.delta_scale = float(getattr(configs, "delta_scale", 1.0))

        global_stdev_mean = float(getattr(configs, "global_stdev_mean", 0.17179356515407562))
        self.register_buffer("global_stdev_mean", torch.tensor(global_stdev_mean, dtype=torch.float32))

    def _sanitize_caption_emb(self, caption_emb, device):
        """把 caption embedding 规整为 `[B, D]` 的 float tensor。"""
        if not isinstance(caption_emb, torch.Tensor):
            caption_emb = torch.tensor(caption_emb)
        caption_emb = caption_emb.to(device).float()
        if caption_emb.dim() == 1:
            caption_emb = caption_emb.unsqueeze(0)
        return caption_emb

    def _build_w(self, channels, device):
        """为 direct 模式构造逐通道融合权重。"""
        if self.direct_w_mode == "fixed":
            weight = torch.full((1, 1, channels), self.direct_w_fixed, device=device)
        else:
            weight = torch.sigmoid(self.text_weight).to(device)
            if weight.numel() == 1:
                weight = weight.view(1, 1, 1).expand(1, 1, channels)
            else:
                weight = weight[:channels].view(1, 1, channels)
        return weight

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        element_ids=None,
        group_ids=None,
        norms=None,
        caption_emb=None,
    ):
        del norms
        if caption_emb is None:
            raise ValueError("[Geo-Fusion] caption_emb must be provided")

        device = x_enc.device
        _, _, channels = x_enc.shape
        caption_emb = self._sanitize_caption_emb(caption_emb, device)

        def _call_num():
            return self.numerical_model(
                x_enc,
                x_mark_enc,
                x_dec,
                x_mark_dec,
                element_ids=element_ids,
                group_ids=group_ids,
                caption_emb=None,
            )

        if self.freeze_numerical:
            with torch.no_grad():
                y_num = _call_num()
        else:
            y_num = _call_num()

        if self.disable_text:
            return y_num

        # direct：文本流直接给出一条预测，再与数值流做加权融合。
        if self.text_mode == "direct":
            y_text = self.text_pred_proj(caption_emb).unsqueeze(-1)
            if channels > 1:
                y_text = y_text.expand(-1, -1, channels)
            weight = self._build_w(channels, device)
            return (1.0 - weight) * y_num + weight * y_text

        # residual：文本流提供修正量，叠加到数值预测上。
        if self.text_mode == "residual":
            delta_y = self.delta_proj(caption_emb) * self.delta_scale
            delta_y = delta_y.unsqueeze(-1)
            if channels > 1:
                delta_y = delta_y.expand(-1, -1, channels)

            if self.force_gain == 0:
                gain = torch.zeros_like(delta_y)
            elif self.force_gain > 0:
                gain = torch.ones_like(delta_y) * self.force_gain
            else:
                if self.use_vol_prior:
                    sample_stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
                    base_stdev = sample_stdev.mean(dim=2, keepdim=True)
                    gain = 1.0 + self.alpha * (base_stdev / (self.global_stdev_mean + 1e-6))
                    gain = gain.expand(-1, self.pred_len, -1)
                else:
                    gain = torch.ones_like(delta_y)

            return y_num + gain * delta_y

        raise ValueError(f"[Geo-Fusion] Unknown text_mode: {self.text_mode}")
