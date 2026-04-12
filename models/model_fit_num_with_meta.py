"""
FIT 数值流 + 元数据主干。

说明：
- 这个模型同时服务 `fit_num_with_meta` 和 `fit_fusion`
- 数值流原有训练逻辑保持不变
- 本轮新增 `extract_features()`，只给 fusion 读取中间特征用
"""

import torch
from torch import nn

from layers.Embed import PatchEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer


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
    输出:  [B, C, pred_len]，外面再 permute 成 [B, pred_len, C]
    """

    def __init__(self, target_window, head_dropout=0.0):
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
    为每个 patch 位置预测一个重要性系数 alpha。
    输入:  [B, P, d_model]
    输出:  [B, P]
    """

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.msa = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.linear_alpha = nn.Linear(d_model, 1)

    def forward(self, x):
        attended, _ = self.msa(x, x, x)
        alpha = torch.sigmoid(self.linear_alpha(attended))
        return alpha.squeeze(-1)


class FitMetadataEmbedding(nn.Module):
    """把 FIT 的 city / gender / age / element 映射到 d_model 空间。"""

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
        element_emb = self.element_embed(element_ids)

        group_emb = self.group_fusion(torch.cat([city_emb, gender_emb, age_emb], dim=-1))
        return group_emb, element_emb


class Model_Fit_Num_With_Meta(nn.Module):
    """
    FIT 数值主干 + meta 融合模型。

    流程保持原有语义：
    1. Instance Norm
    2. Patch Embedding
    3. 融合 group / element 元数据
    4. AIM
    5. Transformer Encoder
    6. Head 输出未来序列
    7. De-Norm

    新增：
    - `extract_features()`：给 fusion 读取中间特征，不改变数值流训练方式
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

        patch_len = getattr(configs, "patch_len", 16)
        stride = getattr(configs, "stride", patch_len // 2)
        padding = stride

        self.patch_embedding = PatchEmbedding(self.d_model, patch_len, stride, padding, self.dropout)
        self.num_patches = int((self.seq_len - patch_len) / stride + 2)

        # 若外部没有显式传入 meta 词表大小，则退回到 FIT 当前默认值。
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
                        self.n_heads,
                    ),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for _ in range(self.e_layers)
            ],
            norm_layer=nn.Sequential(
                Transpose(1, 2),
                nn.BatchNorm1d(self.d_model),
                Transpose(1, 2),
            ),
        )

        self.head = FlattenHead(self.pred_len, self.dropout)

    @staticmethod
    def _instance_normalize(x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        return (x_enc - means) / stdev, means, stdev

    def _patchify(self, x_enc):
        batch_size = x_enc.shape[0]
        x = x_enc.permute(0, 2, 1)
        patch_tokens, num_vars = self.patch_embedding(x)  # [B*C, P, d_model]
        patch_count = patch_tokens.shape[1]
        patch_tokens = patch_tokens.reshape(batch_size, num_vars, patch_count, self.d_model)
        return patch_tokens, num_vars, patch_count

    def _fuse_metadata(self, patch_tokens, city_ids, gender_ids, age_ids, element_ids):
        batch_size, num_vars, patch_count, _ = patch_tokens.shape
        group_emb, element_emb = self.meta_embed(city_ids, gender_ids, age_ids, element_ids)

        group_exp = group_emb[:, None, None, :].expand(-1, num_vars, patch_count, -1)
        element_exp = element_emb[:, None, None, :].expand(-1, num_vars, patch_count, -1)

        fused = torch.cat([patch_tokens, group_exp, element_exp], dim=-1)
        fused = fused.reshape(batch_size * num_vars * patch_count, -1)
        return self.meta_fusion(fused).reshape(batch_size, num_vars, patch_count, self.d_model)

    def _apply_aim(self, patch_tokens):
        batch_size, num_vars, patch_count, _ = patch_tokens.shape
        patch_tokens = patch_tokens.reshape(batch_size * num_vars, patch_count, self.d_model)
        alpha = self.aim(patch_tokens)
        patch_tokens = patch_tokens * alpha.unsqueeze(-1)
        return patch_tokens.reshape(batch_size, num_vars, patch_count, self.d_model)

    def _encode_patches(self, patch_tokens):
        batch_size, num_vars, patch_count, _ = patch_tokens.shape
        encoded = self.encoder(patch_tokens.reshape(batch_size * num_vars, patch_count, self.d_model))[0]
        return encoded.reshape(batch_size, num_vars, patch_count, self.d_model)

    def _decode_forecast(self, encoded_tokens, means, stdev):
        encoded_tokens = encoded_tokens.permute(0, 1, 3, 2)
        dec_out = self.head(encoded_tokens).permute(0, 2, 1)
        return dec_out * stdev[:, 0, :].unsqueeze(1) + means[:, 0, :].unsqueeze(1)

    @staticmethod
    def _pool_summary(encoded_tokens):
        # 对 patch 和变量维度做平均，得到给 fusion 使用的数值摘要。
        return encoded_tokens.mean(dim=(1, 2))

    def _run_backbone(self, x_enc, city_ids, gender_ids, age_ids, element_ids):
        x_norm, means, stdev = self._instance_normalize(x_enc)
        patch_tokens, _, _ = self._patchify(x_norm)
        patch_tokens = self._fuse_metadata(patch_tokens, city_ids, gender_ids, age_ids, element_ids)
        patch_tokens = self._apply_aim(patch_tokens)
        encoded_tokens = self._encode_patches(patch_tokens)
        summary_state = self._pool_summary(encoded_tokens)
        forecast = self._decode_forecast(encoded_tokens, means, stdev)[:, -self.pred_len :, :]

        return {
            "forecast": forecast,
            "encoded_tokens": encoded_tokens,
            "summary_state": summary_state,
            "means": means,
            "stdev": stdev,
        }

    def extract_features(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        city_ids,
        gender_ids,
        age_ids,
        element_ids,
        caption_emb=None,
    ):
        del x_mark_enc, x_dec, x_mark_dec, caption_emb
        return self._run_backbone(x_enc, city_ids, gender_ids, age_ids, element_ids)

    def forecast(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        city_ids,
        gender_ids,
        age_ids,
        element_ids,
        caption_emb=None,
    ):
        return self.extract_features(
            x_enc,
            x_mark_enc,
            x_dec,
            x_mark_dec,
            city_ids,
            gender_ids,
            age_ids,
            element_ids,
            caption_emb,
        )["forecast"]

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
        caption_emb=None,
    ):
        if self.task_name in [
            "long_term_forecast",
            "short_term_forecast",
            "long_term_forecast_meta",
            "fit_num_with_meta",
        ]:
            if city_ids is None or gender_ids is None or age_ids is None or element_ids is None:
                raise ValueError("需要 city / gender / age / element 四个元数据输入")

            out = self.forecast(
                x_enc,
                x_mark_enc,
                x_dec,
                x_mark_dec,
                city_ids,
                gender_ids,
                age_ids,
                element_ids,
                caption_emb,
            )
            return out[:, -self.pred_len :, :]

        return None
