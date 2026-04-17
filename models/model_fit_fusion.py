"""
FIT fusion model for the active 0417 line.

Design goals:
- Keep the numerical backbone unchanged.
- Switch FIT fusion from offline `emb.pt` to raw text in json.
- Follow the DualSG spirit: numerical prediction first, text branch second,
  then fuse directly in forecast space.
- Keep the text encoder frozen.
- Keep `disable_text=True` as a strict pure-numerical fallback.
"""

from copy import deepcopy

import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer

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
    Active FIT fusion model for the 0417 branch.

    Numerical path:
        current `Model_Fit_Num_With_Meta` backbone -> `y_num`

    Text path:
        raw caption text -> frozen text encoder -> token pooling -> `y_text`

    Fusion:
        y_final = (1 - w_t) * y_num + w_t * y_text
    """

    def __init__(self, configs, numerical_ckpt_path=None):
        super().__init__()

        self.configs = configs
        self.pred_len = int(configs.pred_len)
        self.num_channels = int(getattr(configs, "enc_in", 1))
        self.output_dim = self.pred_len * self.num_channels

        self.freeze_numerical = _to_bool(getattr(configs, "freeze_numerical", False), False)
        self.disable_text = _to_bool(getattr(configs, "disable_text", False), False)

        self.text_model_type = str(getattr(configs, "text_model_type", "gpt2")).lower()
        self.text_model_path = str(getattr(configs, "text_model_path", "./weights/gpt2"))
        self.text_pool_type = str(getattr(configs, "text_pool_type", "avg")).lower()
        self.text_max_length = int(getattr(configs, "text_max_length", 256))
        self.fusion_hidden = int(getattr(configs, "fusion_hidden", 96))
        self.fusion_dropout = float(getattr(configs, "fusion_dropout", 0.1))

        self.numerical_model = self._build_numerical_backbone(configs, numerical_ckpt_path)

        self.text_tokenizer, self.text_encoder, self.text_dim = self._build_text_model()
        self.caption_proj = nn.Sequential(
            nn.Linear(self.text_dim, self.fusion_hidden),
            nn.GELU(),
            nn.LayerNorm(self.fusion_hidden),
            nn.Dropout(self.fusion_dropout),
            nn.Linear(self.fusion_hidden, self.output_dim),
        )

        # `sigmoid(0) = 0.5`, matching the requested default.
        self.fusion_weight = nn.Parameter(torch.zeros(self.pred_len, self.num_channels))
        self.aux_loss = None

        if self.disable_text:
            self._freeze_non_numerical_parameters()

    def train(self, mode=True):
        super().train(mode)
        # Keep the text encoder frozen and in eval mode even when the outer
        # fusion model enters train mode.
        self.text_encoder.eval()
        return self

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

    def _build_text_model(self):
        if self.text_model_type != "gpt2":
            raise ValueError(f"[Fit-Fusion] Unsupported text_model_type: {self.text_model_type}")

        tokenizer = GPT2Tokenizer.from_pretrained(self.text_model_path, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        text_encoder = GPT2Model.from_pretrained(self.text_model_path, local_files_only=True)
        text_encoder.config.use_cache = False
        text_encoder.eval()
        for param in text_encoder.parameters():
            param.requires_grad = False

        return tokenizer, text_encoder, int(text_encoder.config.hidden_size)

    def _freeze_non_numerical_parameters(self):
        for name, param in self.named_parameters():
            if name.startswith("numerical_model."):
                continue
            param.requires_grad = False

    def _normalize_caption_text(self, caption_text, batch_size):
        if caption_text is None:
            raise ValueError("[Fit-Fusion] caption_text must be provided when disable_text=False")

        if isinstance(caption_text, str):
            texts = [caption_text]
        else:
            texts = list(caption_text)

        texts = ["" if text is None else str(text) for text in texts]
        if len(texts) != batch_size:
            raise ValueError(
                f"[Fit-Fusion] raw text batch size mismatch: got {len(texts)} texts for batch={batch_size}"
            )
        return texts

    def _pool_hidden_states(self, hidden_states, attention_mask):
        if self.text_pool_type == "avg":
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            masked_hidden = hidden_states * mask
            denom = mask.sum(dim=1).clamp(min=1.0)
            return masked_hidden.sum(dim=1) / denom

        if self.text_pool_type == "max":
            mask = attention_mask.unsqueeze(-1).bool()
            masked_hidden = hidden_states.masked_fill(~mask, float("-inf"))
            return masked_hidden.max(dim=1).values

        if self.text_pool_type == "min":
            mask = attention_mask.unsqueeze(-1).bool()
            masked_hidden = hidden_states.masked_fill(~mask, float("inf"))
            return masked_hidden.min(dim=1).values

        raise ValueError(f"[Fit-Fusion] Unsupported text_pool_type: {self.text_pool_type}")

    def _encode_caption_text(self, caption_text, device):
        inputs = self.text_tokenizer(
            caption_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.text_max_length,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
        return self._pool_hidden_states(outputs.last_hidden_state, inputs["attention_mask"])

    def _call_numerical_backbone(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        city_ids,
        gender_ids,
        age_ids,
        element_ids,
    ):
        if self.freeze_numerical:
            with torch.no_grad():
                return self.numerical_model.extract_features(
                    x_enc,
                    x_mark_enc,
                    x_dec,
                    x_mark_dec,
                    city_ids,
                    gender_ids,
                    age_ids,
                    element_ids,
                    caption_emb=None,
                )

        return self.numerical_model.extract_features(
            x_enc,
            x_mark_enc,
            x_dec,
            x_mark_dec,
            city_ids,
            gender_ids,
            age_ids,
            element_ids,
            caption_emb=None,
        )

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        city_ids=None,
        gender_ids=None,
        age_ids=None,
        element_ids=None,
        caption_text=None,
    ):
        if city_ids is None or gender_ids is None or age_ids is None or element_ids is None:
            raise ValueError("[Fit-Fusion] city/gender/age/element ids must be provided")

        backbone_out = self._call_numerical_backbone(
            x_enc,
            x_mark_enc,
            x_dec,
            x_mark_dec,
            city_ids,
            gender_ids,
            age_ids,
            element_ids,
        )
        y_num = backbone_out["forecast"][:, -self.pred_len :, :]

        if self.disable_text:
            return y_num

        batch_size = y_num.shape[0]
        caption_text = self._normalize_caption_text(caption_text, batch_size)
        pooled_text = self._encode_caption_text(caption_text, x_enc.device)

        y_text = self.caption_proj(pooled_text).reshape(batch_size, self.pred_len, self.num_channels)
        weight = torch.sigmoid(self.fusion_weight).unsqueeze(0)
        return (1.0 - weight) * y_num + weight * y_text
