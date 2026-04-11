"""
Model_Geo_Num: 仅包含数值流的简化版本
去除了 TSCaption 和 LLM 语义流，专注于时序数值预测

架构:
    Input -> PatchEmbedding -> Transformer Encoder -> FlattenHead -> Output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import PatchEmbedding
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

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

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class AdaptiveImportanceMask(nn.Module):
    """自适应重要性掩码：为每个patch学习重要性权重"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.msa = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.linear_alpha = nn.Linear(d_model, 1)

    def forward(self, Ei):
        # Ei: [bs * nvars, patch_num, d_model]
        attended, _ = self.msa(Ei, Ei, Ei)
        alpha = torch.sigmoid(self.linear_alpha(attended))  # [bs * nvars, patch_num, 1]
        return alpha.squeeze(-1)  # [bs * nvars, patch_num]


class Model_Geo_Num(nn.Module):
    """
    Model_Geo_Num 数值流模型
    
    仅包含:
    1. Patch Embedding (支持标准和自适应两种模式)
    2. Transformer Encoder
    3. Adaptive Importance Mask
    4. Flatten Head 预测
    
    去除:
    - TSCaption 文本生成
    - LLM 语义编码
    - 双流融合
    """
    
    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.args = configs
        
        padding = stride
        patch_len = configs.patch_len
        
        # ============ Patch Embedding ============
        if configs.patch_adaptive == 0:
            # 标准 Patch Embedding
            self.patch_embedding = PatchEmbedding(
                configs.d_model, patch_len, stride, padding, configs.dropout)
            self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        else:
            # 自适应多尺度 Patch Embedding
            self.num_patches1 = int((configs.seq_len - 8) / 4 + 2)   # 细粒度 (8,4)
            self.num_patches2 = int((configs.seq_len - 16) / 8 + 2)  # 中粒度 (16,8)
            self.num_patches3 = int((configs.seq_len - 32) / 16 + 2) # 粗粒度 (32,16)
            
            total_patches = self.num_patches2
            
            # 根据预测长度选择不同尺度的比例
            if self.pred_len <= 24:
                ratios = [1.0, 0, 0]  # 短期：全部细粒度
            elif self.pred_len == 96:
                ratios = [0.75, 0.25, 0]
            elif self.pred_len == 192:
                ratios = [5/12, 7/12, 0]
            elif self.pred_len == 336:
                ratios = [1/6, 1/2, 1/3]
            else:  # 720+
                ratios = [0, 7/12, 5/12]

            self.k1 = int(ratios[0] * total_patches)
            self.k2 = int(ratios[1] * total_patches)
            self.k3 = total_patches - self.k1 - self.k2

            # 确保不超过可用 patch 数
            self.k3 = min(self.k3, self.num_patches3)
            self.k2 = min(self.k2, self.num_patches2)
            self.k1 = min(self.k1, self.num_patches1)
            
            self.patch_embedding1 = PatchEmbedding(
                configs.d_model, patch_len=8, stride=4, padding=2, dropout=configs.dropout)
            self.patch_embedding2 = PatchEmbedding(
                configs.d_model, patch_len=16, stride=8, padding=4, dropout=configs.dropout)
            self.patch_embedding3 = PatchEmbedding(
                configs.d_model, patch_len=32, stride=16, padding=8, dropout=configs.dropout)

            # 调整剩余 patch
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
        
        # ============ Transformer Encoder ============
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2))
        )
        
        # ============ Adaptive Importance Mask ============
        self.aim = AdaptiveImportanceMask(d_model=configs.d_model, num_heads=configs.n_heads)
        
        # ============ Prediction Head ============
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)
    
    def apply_patch_embeddings(self, x_enc):
        """应用 Patch Embedding"""
        if self.args.patch_adaptive == 0:
            return self.patch_embedding(x_enc)
        else:
            # 多尺度 patch embedding
            enc1, n_vars1 = self.patch_embedding1(x_enc)
            enc2, n_vars2 = self.patch_embedding2(x_enc)
            enc3, n_vars3 = self.patch_embedding3(x_enc)
            
            # 选择各尺度的 patches
            enc3_part = enc3[:, :self.k3, :]
            start_idx = (self.num_patches2 - self.k2) // 2
            enc2_part = enc2[:, start_idx:start_idx+self.k2, :]
            enc1_part = enc1[:, -self.k1:, :]

            # 按时间顺序拼接
            if self.k1 == 0:
                combined = torch.cat([enc3_part, enc2_part], dim=1)
            elif self.k3 == 0:
                combined = torch.cat([enc2_part, enc1_part], dim=1)
            else:
                combined = torch.cat([enc3_part, enc2_part, enc1_part], dim=1)
            
            return combined, n_vars1
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """数值流预测"""
        # 1. Instance Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        # 2. Patch Embedding
        x_enc = x_enc.permute(0, 2, 1)  # [B, C, L]
        enc_out, n_vars = self.apply_patch_embeddings(x_enc)
        
        # 3. Adaptive Importance Mask
        alpha = self.aim(enc_out)
        enc_out = enc_out * alpha.unsqueeze(-1)
        
        # 4. Transformer Encoder
        enc_out, attns = self.encoder(enc_out)
        
        # 5. Reshape for prediction head
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)  # [B, C, d_model, patch_num]
        
        # 6. 动态计算head_nf并调整线性层
        current_patch_num = enc_out.shape[-1]
        current_head_nf = self.args.d_model * current_patch_num
        
        if hasattr(self.head, 'linear') and self.head.linear.in_features != current_head_nf:
            self.head.linear = nn.Linear(current_head_nf, self.pred_len).to(enc_out.device)
        
        # 7. Prediction Head
        dec_out = self.head(enc_out)  # [B, C, pred_len]
        dec_out = dec_out.permute(0, 2, 1)  # [B, pred_len, C]
        
        # 8. De-Normalization
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        
        return dec_out
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None