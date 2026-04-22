"""Geo 纯数值流模型。"""

import torch
import torch.nn as nn

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
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class AdaptiveImportanceMask(nn.Module):
    """为每个 patch 学习一个重要性权重。"""

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.msa = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.linear_alpha = nn.Linear(d_model, 1)

    def forward(self, patch_tokens):
        attended, _ = self.msa(patch_tokens, patch_tokens, patch_tokens)
        alpha = torch.sigmoid(self.linear_alpha(attended))
        return alpha.squeeze(-1)


class Model_Geo_Num(nn.Module):
    """仅使用数值序列的 Geo 预测模型。"""

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.args = configs

        padding = stride
        patch_len = configs.patch_len

        if configs.patch_adaptive == 0:
            self.patch_embedding = PatchEmbedding(configs.d_model, patch_len, stride, padding, configs.dropout)
            self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        else:
            self.num_patches1 = int((configs.seq_len - 8) / 4 + 2)
            self.num_patches2 = int((configs.seq_len - 16) / 8 + 2)
            self.num_patches3 = int((configs.seq_len - 32) / 16 + 2)
            total_patches = self.num_patches2

            if self.pred_len <= 24:
                ratios = [1.0, 0.0, 0.0]
            elif self.pred_len == 96:
                ratios = [0.75, 0.25, 0.0]
            elif self.pred_len == 192:
                ratios = [5 / 12, 7 / 12, 0.0]
            elif self.pred_len == 336:
                ratios = [1 / 6, 1 / 2, 1 / 3]
            else:
                ratios = [0.0, 7 / 12, 5 / 12]

            self.k1 = int(ratios[0] * total_patches)
            self.k2 = int(ratios[1] * total_patches)
            self.k3 = total_patches - self.k1 - self.k2

            self.k3 = min(self.k3, self.num_patches3)
            self.k2 = min(self.k2, self.num_patches2)
            self.k1 = min(self.k1, self.num_patches1)

            self.patch_embedding1 = PatchEmbedding(configs.d_model, patch_len=8, stride=4, padding=2, dropout=configs.dropout)
            self.patch_embedding2 = PatchEmbedding(configs.d_model, patch_len=16, stride=8, padding=4, dropout=configs.dropout)
            self.patch_embedding3 = PatchEmbedding(configs.d_model, patch_len=32, stride=16, padding=8, dropout=configs.dropout)

            remaining = total_patches - (self.k1 + self.k2 + self.k3)
            if remaining > 0:
                if ratios[2] >= max(ratios):
                    self.k3 += remaining
                elif ratios[0] >= max(ratios):
                    self.k1 += remaining
                else:
                    self.k2 += remaining

            self.head_nf = configs.d_model * (self.k1 + self.k2 + self.k3)

        self.num_channels = configs.enc_in

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2)),
        )

        self.aim = AdaptiveImportanceMask(d_model=configs.d_model, num_heads=configs.n_heads)

        if self.task_name in {"long_term_forecast", "short_term_forecast", "geo_num"}:
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len, head_dropout=configs.dropout)

    def apply_patch_embeddings(self, x_enc):
        if self.args.patch_adaptive == 0:
            return self.patch_embedding(x_enc)

        enc1, n_vars1 = self.patch_embedding1(x_enc)
        enc2, _ = self.patch_embedding2(x_enc)
        enc3, _ = self.patch_embedding3(x_enc)

        enc3_part = enc3[:, : self.k3, :]
        start_idx = (self.num_patches2 - self.k2) // 2
        enc2_part = enc2[:, start_idx : start_idx + self.k2, :]
        enc1_part = enc1[:, -self.k1 :, :]

        if self.k1 == 0:
            combined = torch.cat([enc3_part, enc2_part], dim=1)
        elif self.k3 == 0:
            combined = torch.cat([enc2_part, enc1_part], dim=1)
        else:
            combined = torch.cat([enc3_part, enc2_part, enc1_part], dim=1)

        return combined, n_vars1

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        del x_mark_enc, x_dec, x_mark_dec

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.apply_patch_embeddings(x_enc)

        alpha = self.aim(enc_out)
        enc_out = enc_out * alpha.unsqueeze(-1)

        enc_out, _ = self.encoder(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        current_patch_num = enc_out.shape[-1]
        current_head_nf = self.args.d_model * current_patch_num
        if hasattr(self.head, "linear") and self.head.linear.in_features != current_head_nf:
            self.head.linear = nn.Linear(current_head_nf, self.pred_len).to(enc_out.device)

        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.task_name in {"long_term_forecast", "short_term_forecast", "geo_num"}:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len :, :]
        return None
