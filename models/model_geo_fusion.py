import torch
import torch.nn as nn
from copy import deepcopy

from models.model_geo_num_with_meta import Model_Geo_Num_With_Meta as GeoNumericalModel


class Model_Geo_Fusion(nn.Module):
    """
    Model_Geo_Fusion（GeoStyle）
    """

    def __init__(self, configs, numerical_ckpt_path=None):
        super().__init__()

        self.configs = configs
        self.pred_len = configs.pred_len
        self.num_channels = getattr(configs, "enc_in", 1)

        # ================= 消融控制参数 =================
        self.freeze_numerical = getattr(configs, "freeze_numerical", False)
        self.disable_text = getattr(configs, "disable_text", False)
        self.text_mode = getattr(configs, "text_mode", "residual")

        # residual 相关
        self.use_vol_prior = bool(getattr(configs, "use_vol_prior", True))
        self.force_gain = float(getattr(configs, "force_gain", -1))

        # direct 相关
        self.direct_w_mode = getattr(configs, "direct_w_mode", "learned")
        self.direct_w_fixed = float(getattr(configs, "direct_w_fixed", 0.5))

        # =================================================
        # 1) 数值流
        # =================================================
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
            # print(f"[Geo-Fusion] Loaded numerical ckpt: {numerical_ckpt_path}")

        if self.freeze_numerical:
            for p in self.numerical_model.parameters():
                p.requires_grad = False
            self.numerical_model.eval()

        # =================================================
        # 2) 文本分支
        # =================================================
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

        # direct learned w
        self.text_weight = nn.Parameter(torch.ones(self.num_channels) * -1.0)

        # residual 控制参数
        self.alpha = float(getattr(configs, "alpha", 1.0))
        self.delta_scale = float(getattr(configs, "delta_scale", 1.0))

        global_stdev_mean = float(
            getattr(configs, "global_stdev_mean", 0.17179356515407562)
        )
        self.register_buffer(
            "global_stdev_mean",
            torch.tensor(global_stdev_mean, dtype=torch.float32)
        )

    # =====================================================
    # utils
    # =====================================================
    def _sanitize_caption_emb(self, caption_emb, device):
        if not isinstance(caption_emb, torch.Tensor):
            caption_emb = torch.tensor(caption_emb)
        caption_emb = caption_emb.to(device).float()
        if caption_emb.dim() == 1:
            caption_emb = caption_emb.unsqueeze(0)
        return caption_emb

    def _build_w(self, C, device):
        if self.direct_w_mode == "fixed":
            w = torch.full((1, 1, C), self.direct_w_fixed, device=device)
        else:
            w = torch.sigmoid(self.text_weight).to(device)
            if w.numel() == 1:
                w = w.view(1, 1, 1).expand(1, 1, C)
            else:
                w = w[:C].view(1, 1, C)
        return w

    # =====================================================
    # forward
    # =====================================================
    def forward(
        self,
        x_enc, x_mark_enc,
        x_dec, x_mark_dec,
        element_ids=None,
        group_ids=None,
        norms=None,
        caption_emb=None,
    ):
        if caption_emb is None:
            raise ValueError("[Geo-Fusion] caption_emb must be provided")

        device = x_enc.device
        B, _, C = x_enc.shape
        caption_emb = self._sanitize_caption_emb(caption_emb, device)

        def _call_num():
            return self.numerical_model(
                x_enc, x_mark_enc,
                x_dec, x_mark_dec,
                element_ids=element_ids,
                group_ids=group_ids,
                caption_emb=None
            )

        if self.freeze_numerical:
            with torch.no_grad():
                y_num = _call_num()
        else:
            y_num = _call_num()

        if self.disable_text:
            return y_num

        # ================= direct =================
        if self.text_mode == "direct":
            y_text = self.text_pred_proj(caption_emb).unsqueeze(-1)
            if C > 1:
                y_text = y_text.expand(-1, -1, C)
            w = self._build_w(C, device)
            return (1.0 - w) * y_num + w * y_text

        # ================= residual =================
        if self.text_mode == "residual":
            delta_y = self.delta_proj(caption_emb) * self.delta_scale
            delta_y = delta_y.unsqueeze(-1)
            if C > 1:
                delta_y = delta_y.expand(-1, -1, C)

            # --- gain 控制 ---
            if self.force_gain == 0:
                gain = torch.zeros_like(delta_y)
            elif self.force_gain > 0:
                gain = torch.ones_like(delta_y) * self.force_gain
            else:
                if self.use_vol_prior:
                    sample_stdev = torch.sqrt(
                        torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
                    ).detach()
                    base_stdev = sample_stdev.mean(dim=2, keepdim=True)
                    gain = 1.0 + self.alpha * (
                        base_stdev / (self.global_stdev_mean + 1e-6)
                    )
                    gain = gain.expand(-1, self.pred_len, -1)
                else:
                    gain = torch.ones_like(delta_y)

            return y_num + gain * delta_y

        raise ValueError(f"[Geo-Fusion] Unknown text_mode: {self.text_mode}")