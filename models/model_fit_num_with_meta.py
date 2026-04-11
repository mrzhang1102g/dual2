"""
Model_Fit_Num_With_Meta
"""

import torch
from torch import nn
import torch.nn.functional as F

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims = dims
        self.contiguous = contiguous

    def forward(self, x):
        x = x.transpose(*self.dims)
        return x.contiguous() if self.contiguous else x


class FlattenHead(nn.Module):
    """
    输入:  [B, C, d_model, P]
    输出:  [B, C, pred_len] -> 外面 permute 到 [B, pred_len, C]
    """
    def __init__(self, nf, target_window, head_dropout=0.0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.LazyLinear(target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class AdaptiveImportanceMask(nn.Module):
    """
    输入:  [B, P, d_model]
    输出:  alpha [B, P]
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.msa = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )
        self.linear_alpha = nn.Linear(d_model, 1)

    def forward(self, x):
        attended, _ = self.msa(x, x, x)
        alpha = torch.sigmoid(self.linear_alpha(attended))
        return alpha.squeeze(-1)


class FitMetadataEmbedding(nn.Module):
    """
    FIT元数据嵌入
    """
    def __init__(self, d_model, num_cities=14, num_genders=2, num_ages=4, num_elements=173):
        super().__init__()
        self.city_embed = nn.Embedding(num_cities, d_model)
        self.gender_embed = nn.Embedding(num_genders, d_model)
        self.age_embed = nn.Embedding(num_ages, d_model)
        self.element_embed = nn.Embedding(num_elements, d_model)

        self.group_fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )

        nn.init.normal_(self.city_embed.weight, std=0.02)
        nn.init.normal_(self.gender_embed.weight, std=0.02)
        nn.init.normal_(self.age_embed.weight, std=0.02)
        nn.init.normal_(self.element_embed.weight, std=0.02)

    @staticmethod
    def _clamp_ids(ids, n):
        return ids.long().clamp(0, n - 1)

    def forward(self, city_ids, gender_ids, age_ids, element_ids):
        city_ids = self._clamp_ids(city_ids, self.city_embed.num_embeddings)
        gender_ids = self._clamp_ids(gender_ids, self.gender_embed.num_embeddings)
        age_ids = self._clamp_ids(age_ids, self.age_embed.num_embeddings)
        element_ids = self._clamp_ids(element_ids, self.element_embed.num_embeddings)

        city_emb = self.city_embed(city_ids)
        gender_emb = self.gender_embed(gender_ids)
        age_emb = self.age_embed(age_ids)

        group_emb = self.group_fusion(
            torch.cat([city_emb, gender_emb, age_emb], dim=-1)
        )
        element_emb = self.element_embed(element_ids)
        return group_emb, element_emb


class Model_Fit_Num_With_Meta(nn.Module):
    """
    Model_Fit_Num_With_Meta（变量级建模版，用于residual）
    """
    def __init__(self, configs):
        super().__init__()

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.d_model = configs.d_model
        self.n_heads = configs.n_heads
        self.d_ff = configs.d_ff
        self.e_layers = configs.e_layers
        self.dropout = configs.dropout
        self.factor = configs.factor
        self.activation = configs.activation
        self.enc_in = configs.enc_in

        # Patch Embedding
        patch_len = getattr(configs, "patch_len", 16)
        stride = getattr(configs, "stride", patch_len // 2)
        padding = stride

        self.patch_embedding = PatchEmbedding(
            self.d_model, patch_len, stride, padding, self.dropout
        )
        self.num_patches = int((self.seq_len - patch_len) / stride + 2)

        # Metadata
        self.meta_embed = FitMetadataEmbedding(
            self.d_model,
            getattr(configs, "num_city", 14),
            getattr(configs, "num_gender", 2),
            getattr(configs, "num_age", 4),
            getattr(configs, "num_element", 173),
        )

        self.meta_fusion = nn.Sequential(
            nn.Linear(self.d_model * 3, self.d_model * 2),
            nn.ReLU(),
            nn.LayerNorm(self.d_model * 2),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.Dropout(self.dropout),
        )

        self.aim = AdaptiveImportanceMask(self.d_model, self.n_heads)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, self.factor, attention_dropout=self.dropout),
                        self.d_model,
                        self.n_heads
                    ),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                )
                for _ in range(self.e_layers)
            ],
            norm_layer=nn.Sequential(
                Transpose(1, 2),
                nn.BatchNorm1d(self.d_model),
                Transpose(1, 2)
            )
        )

        self.head = FlattenHead(
            self.d_model * self.num_patches,
            self.pred_len,
            self.dropout
        )

    def forecast(
        self,
        x_enc, x_mark_enc, x_dec, x_mark_dec,
        city_ids, gender_ids, age_ids, element_ids,
        caption_emb=None
    ):
        B, L, C = x_enc.shape

        # 1) Instance Norm
        means = x_enc.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = (x_enc - means) / stdev

        # 2) Patch Embedding
        x = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x)  # [B*C,P,d]

        P = enc_out.shape[1]
        enc_out = enc_out.reshape(B, n_vars, P, self.d_model)  # ✅保留变量维

        # 3) Metadata Fusion（变量级）
        group_emb, element_emb = self.meta_embed(city_ids, gender_ids, age_ids, element_ids)
        group_exp = group_emb[:, None, None, :].expand(-1, n_vars, P, -1)
        elem_exp = element_emb[:, None, None, :].expand(-1, n_vars, P, -1)

        fused = torch.cat([enc_out, group_exp, elem_exp], dim=-1)
        fused = fused.reshape(B * n_vars * P, -1)
        enc_out = self.meta_fusion(fused).reshape(B, n_vars, P, self.d_model)

        # 4) AIM（变量独立）
        enc_for_aim = enc_out.reshape(B * n_vars, P, self.d_model)
        alpha = self.aim(enc_for_aim)
        enc_out = (enc_for_aim * alpha.unsqueeze(-1)).reshape(B, n_vars, P, self.d_model)

        # 5) Transformer Encoder
        enc_out = self.encoder(enc_out.reshape(B * n_vars, P, self.d_model))[0]
        enc_out = enc_out.reshape(B, n_vars, P, self.d_model)

        # 6) Head
        enc_out = enc_out.permute(0, 1, 3, 2)
        dec_out = self.head(enc_out).permute(0, 2, 1)

        # 7) De-Norm
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1) + means[:, 0, :].unsqueeze(1)
        return dec_out

    def forward(
        self,
        x_enc, x_mark_enc, x_dec, x_mark_dec,
        city_ids=None, gender_ids=None, age_ids=None, element_ids=None,
        caption_emb=None
    ):
        if self.task_name in ["long_term_forecast", "short_term_forecast", "long_term_forecast_meta", "fit_num_with_meta"]:
            if city_ids is None or gender_ids is None or age_ids is None or element_ids is None:
                raise ValueError("需要city/gender/age/element四个元数据输入")
            out = self.forecast(
                x_enc, x_mark_enc, x_dec, x_mark_dec,
                city_ids, gender_ids, age_ids, element_ids,
                caption_emb
            )
            return out[:, -self.pred_len:, :]
        return None