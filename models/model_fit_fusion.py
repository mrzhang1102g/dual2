import torch
import torch.nn as nn
from copy import deepcopy

from models.model_fit_num_with_meta import Model_Fit_Num_With_Meta as FitNumericalModel


def _to_bool(x, default=False):
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return x != 0
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ["1", "true", "yes", "y", "t", "on"]:
            return True
        if s in ["0", "false", "no", "n", "f", "off"]:
            return False
        return default
    return bool(x)


class Model_Fit_Fusion(nn.Module):
    """
    Model_Fit_Fusion
    - 数值流: Model_Fit_Num_With_Meta
    - 文本流: caption_emb(离线生成)
    - 融合模式:
      A) direct: y_final=(1-w)*y_num+w*y_text
      B) residual: y_final=y_num+gain_total*delta_y
    """

    def __init__(self, configs, numerical_ckpt_path=None):
        super().__init__()

        self.configs = configs
        self.pred_len = configs.pred_len
        self.num_channels = getattr(configs, "enc_in", 1)

        # =========================================================
        # A. 融合/消融开关
        # =========================================================
        self.freeze_numerical = _to_bool(getattr(configs, "freeze_numerical", False), False)

        # 消融A：disable_text=True -> 纯数值流(w/o text)
        self.disable_text = _to_bool(getattr(configs, "disable_text", False), False)

        # 消融B：direct / residual
        self.text_mode = getattr(configs, "text_mode", "residual")

        # direct内部消融：learned-w vs fixed-w
        self.direct_w_mode = getattr(configs, "direct_w_mode", "learned")
        self.direct_w_fixed = float(getattr(configs, "direct_w_fixed", 0.5))

        # residual内部消融：是否启用波动先验prior_gain
        self.use_vol_prior = _to_bool(getattr(configs, "use_vol_prior", True), True)

        # residual内部消融：固定gain；-1表示学习gain
        self.force_gain = float(getattr(configs, "force_gain", -1.0))

        # 可选：Δy正则强度（默认0，不启用）
        self.beta_delta = float(getattr(configs, "beta_delta", 0.0))

        # 可选：调试信息
        self.debug_signature = _to_bool(getattr(configs, "debug_signature", False), False)
        self.debug_once = _to_bool(getattr(configs, "debug_once", False), False)
        self._debug_done = False

        # =========================================================
        # B. 数值流（FIT数值主干）
        # =========================================================
        num_cfg = deepcopy(configs)
        self.numerical_model = FitNumericalModel(num_cfg)

        # ---- 加载数值流checkpoint（可选）----
        if numerical_ckpt_path:
            ckpt = torch.load(numerical_ckpt_path, map_location="cpu")
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            self.numerical_model.load_state_dict(ckpt, strict=False)
            # print(f"[Fit-Fusion] Loaded numerical ckpt: {numerical_ckpt_path}")
        # else:
        #     print("[Fit-Fusion] No numerical checkpoint provided")

        # ---- 是否冻结数值流 ----
        if self.freeze_numerical:
            for p in self.numerical_model.parameters():
                p.requires_grad = False
            self.numerical_model.eval()
        else:
            self.numerical_model.train()

        # debug：打印一次数值流forward签名（可开关）
        if self.debug_signature:
            import inspect
            # print("[Fit-Fusion] numerical_model.forward =", inspect.signature(self.numerical_model.forward))

        # =========================================================
        # C. 文本分支（caption_emb离线生成）
        # =========================================================
        self.llm_dim = getattr(configs, "llm_dim", 768)

        # direct：caption_emb -> 直接预测y_text
        self.text_pred_proj = nn.Sequential(
            nn.Linear(self.llm_dim, self.pred_len * 2),
            nn.ReLU(),
            nn.Linear(self.pred_len * 2, self.pred_len),
        )

        # direct：通道级可学习融合权重w（sigmoid后在(0,1)）
        self.text_weight = nn.Parameter(torch.ones(self.num_channels) * -1.0)

        # =========================================================
        # D. residual纠偏头
        # =========================================================
        self.alpha = float(getattr(configs, "alpha", 1.0))
        self.delta_scale = float(getattr(configs, "delta_scale", 1.0))

        self.num_feat_dim = int(getattr(configs, "num_feat_dim", self.pred_len * 2))
        self.fusion_dropout = float(getattr(configs, "fusion_dropout", 0.1))
        hidden = int(getattr(configs, "fusion_hidden", self.pred_len * 4))

        # y_num压缩：phi(y_num)
        self.num_feat_proj = nn.Sequential(
            nn.Linear(self.pred_len * self.num_channels, self.num_feat_dim),
            nn.GELU(),
            nn.LayerNorm(self.num_feat_dim),
            nn.Dropout(self.fusion_dropout),
        )

        # 融合输入：caption_emb + phi(y_num) + vol(1维)
        self.fusion_in_dim = self.llm_dim + self.num_feat_dim + 1

        # Δy head
        self.delta_proj = nn.Sequential(
            nn.Linear(self.fusion_in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(self.fusion_dropout),
            nn.Linear(hidden, self.pred_len),
        )

        # gain head（sigmoid后(0,1)）
        self.gain_net = nn.Sequential(
            nn.Linear(self.fusion_in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(self.fusion_dropout),
            nn.Linear(hidden, self.pred_len),
        )

        # 初始化：让初始纠偏接近0
        nn.init.zeros_(self.gain_net[-1].weight)
        nn.init.constant_(self.gain_net[-1].bias, -4.0)  # sigmoid(-4)≈0.018
        nn.init.zeros_(self.delta_proj[-1].weight)
        nn.init.zeros_(self.delta_proj[-1].bias)

        # forward里会被Exp读取
        self.aux_loss = None

        # =========================================================
        # E. 参数级消融：disable_text时冻结所有非数值参数
        # =========================================================
        if self.disable_text:
            for name, p in self.named_parameters():
                if not name.startswith("numerical_model"):
                    p.requires_grad = False
            # print("[Fit-Fusion] disable_text=True -> freeze all non-numerical parameters")

    # =========================================================
    # 工具函数
    # =========================================================
    def _sanitize_caption_emb(self, caption_emb, device):
        if not isinstance(caption_emb, torch.Tensor):
            caption_emb = torch.tensor(caption_emb)

        caption_emb = caption_emb.to(device=device)
        if caption_emb.dim() == 1:
            caption_emb = caption_emb.unsqueeze(0)

        # 关键：保持float32
        if caption_emb.dtype != torch.float32:
            caption_emb = caption_emb.float()

        return caption_emb

    def _build_w(self, C, device, dtype):
        """
        direct融合权重w，输出[1,1,C]
        - learned：w=sigmoid(text_weight)
        - fixed：w=direct_w_fixed
        """
        if self.direct_w_mode == "fixed":
            w_scalar = max(0.0, min(1.0, float(self.direct_w_fixed)))
            return torch.full((1, 1, C), w_scalar, device=device, dtype=dtype)

        w = torch.sigmoid(self.text_weight).to(device=device, dtype=dtype)
        if w.numel() == 1:
            w = w.view(1, 1, 1).expand(1, 1, C)
        else:
            w = w[:C].view(1, 1, C)
        return w

    def _pad_or_trim_y_num_flat(self, y_num_flat, target_dim):
        if y_num_flat.shape[1] == target_dim:
            return y_num_flat
        if y_num_flat.shape[1] > target_dim:
            return y_num_flat[:, :target_dim]
        pad = target_dim - y_num_flat.shape[1]
        return torch.cat([y_num_flat, y_num_flat.new_zeros(y_num_flat.shape[0], pad)], dim=1)

    def _call_numerical_fit(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                            city_id, gender_id, age_id, element_id):
        return self.numerical_model(
            x_enc, x_mark_enc,
            x_dec, x_mark_dec,
            city_ids=city_id,
            gender_ids=gender_id,
            age_ids=age_id,
            element_ids=element_id,
            caption_emb=None
        )

    # =========================================================
    # forward
    # =========================================================
    def forward(
        self,
        x_enc, x_mark_enc,
        x_dec, x_mark_dec,
        city_id=None, gender_id=None, age_id=None, element_id=None,
        caption_emb=None,
        # 接口保留（不使用）
        element_ids=None, group_ids=None,
        norms=None,
    ):
        if caption_emb is None:
            raise ValueError("[Fit-Fusion] caption_emb must be provided")

        device = x_enc.device
        B, _, C = x_enc.shape

        caption_emb = self._sanitize_caption_emb(caption_emb, device=device)

        # debug：只打印一次输入信息
        if self.debug_once and (not self._debug_done):
            self._debug_done = True
            try:
                def _minmax(t):
                    if t is None:
                        return None
                    if isinstance(t, torch.Tensor) and t.numel() > 0:
                        return (t.min().item(), t.max().item())
                    return None
                # print("[Fit-Fusion] ids:",
                #       "city", _minmax(city_id),
                #       "gender", _minmax(gender_id),
                #       "age", _minmax(age_id),
                #       "element", _minmax(element_id))
            except Exception:
                # print("[Fit-Fusion] ids debug failed")
                pass
            # print("[Fit-Fusion] x_enc", x_enc.shape,
            #       "x_mark_enc", x_mark_enc.shape,
            #       "x_dec", x_dec.shape,
            #       "x_mark_dec", x_mark_dec.shape,
            #       "caption_emb", caption_emb.shape,
            #       "caption_dtype", caption_emb.dtype)

        # =================================================
        # 1) 数值流预测
        # =================================================
        def _run_num():
            return self._call_numerical_fit(
                x_enc, x_mark_enc,
                x_dec, x_mark_dec,
                city_id, gender_id, age_id, element_id
            )

        if self.freeze_numerical:
            with torch.no_grad():
                y_num = _run_num()
        else:
            y_num = _run_num()

        # =================================================
        # 2) 消融A：禁用文本流
        # =================================================
        if self.disable_text:
            self.aux_loss = None
            return y_num

        # =================================================
        # 3) direct：文本直接预测y_text
        # =================================================
        if self.text_mode == "direct":
            # 用float32 caption_emb做预测，然后cast到y_num.dtype以便融合
            y_text = self.text_pred_proj(caption_emb)  # [B,pred_len] float32
            y_text = y_text.to(dtype=y_num.dtype)
            y_text = y_text.unsqueeze(-1)              # [B,pred_len,1]
            if C > 1:
                y_text = y_text.expand(-1, -1, C)

            w = self._build_w(C, device=device, dtype=y_num.dtype)  # [1,1,C]
            self.aux_loss = None
            return (1.0 - w) * y_num + w * y_text

        # =================================================
        # 4) residual：Δy纠偏
        # =================================================
        if self.text_mode == "residual":
            # 4.1 计算vol（不反传）
            with torch.no_grad():
                sample_stdev = torch.sqrt(
                    torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
                )  # [B,1,C]
                base_stdev = sample_stdev.mean(dim=2, keepdim=True)  # [B,1,1]
                vol = base_stdev.view(B, 1)  # [B,1]

            # 4.2 y_num -> phi(y_num)
            y_num_flat = y_num.reshape(B, self.pred_len * C)  # [B,pred_len*C]

            # 维度对齐
            target_dim = self.pred_len * self.num_channels
            y_num_flat = self._pad_or_trim_y_num_flat(y_num_flat, target_dim)

            # detach：不让纠偏头梯度回传到数值流
            num_feat = self.num_feat_proj(y_num_flat.detach())  # [B,num_feat_dim] dtype=y_num.dtype

            # 4.3 预测Δy与gain
            # 注意：caption_emb是float32，num_feat/vol可能是半精度，cat会提升到float32，稳定训练
            fusion_in = torch.cat(
                [caption_emb, num_feat.float(), vol.float()],
                dim=-1
            )  # [B,fusion_in_dim] float32

            delta_y = self.delta_proj(fusion_in) * self.delta_scale  # [B,pred_len] float32

            if self.force_gain >= 0:
                gain = torch.full(
                    (B, self.pred_len),
                    self.force_gain,
                    device=device,
                    dtype=delta_y.dtype
                )
            else:
                gain = torch.sigmoid(self.gain_net(fusion_in))  # [B,pred_len] float32

            # cast到y_num.dtype再融合
            delta_y = delta_y.to(dtype=y_num.dtype).unsqueeze(-1)  # [B,pred_len,1]
            gain = gain.to(dtype=y_num.dtype).unsqueeze(-1)        # [B,pred_len,1]

            if C > 1:
                delta_y = delta_y.expand(-1, -1, C)
                gain = gain.expand(-1, -1, C)

            gain_total = gain

            y_final = y_num + gain_total * delta_y

            # 可选aux正则：限制纠偏幅度
            if self.beta_delta > 0:
                self.aux_loss = self.beta_delta * (gain_total * delta_y).pow(2).mean()
            else:
                self.aux_loss = None

            return y_final

        raise ValueError(f"[Fit-Fusion] Unknown text_mode: {self.text_mode}")