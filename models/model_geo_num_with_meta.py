"""
Model_Geo_Num_With_Meta
"""

import torch
from torch import nn
import torch.nn.functional as F

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


# ======================================================
# 基础小模块
# ======================================================
class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims = dims
        self.contiguous = contiguous

    def forward(self, x):
        x = x.transpose(*self.dims)
        return x.contiguous() if self.contiguous else x


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0.0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)   # [B,C,nf]
        x = self.linear(x)    # [B,C,pred_len]
        x = self.dropout(x)
        return x


class AdaptiveImportanceMask(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.msa = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )
        self.linear_alpha = nn.Linear(d_model, 1)

    def forward(self, Ei):
        attended, _ = self.msa(Ei, Ei, Ei)
        alpha = torch.sigmoid(self.linear_alpha(attended))
        return alpha.squeeze(-1)  # [B,P]


# ======================================================
# GeoStyle:仅element embedding
# ======================================================
class ElementEmbedding(nn.Module):
    def __init__(self, d_model, num_elements, disable=False):
        super().__init__()
        self.disable = disable
        if self.disable:
            self.element_embed = None
            return

        self.element_embed = nn.Embedding(num_elements, d_model)
        nn.init.normal_(self.element_embed.weight, std=0.02)

    def sanitize_ids(self, ids):
        if ids is None:
            return None
        ids = ids.long()
        return ids.clamp(0, self.element_embed.num_embeddings - 1)

    def forward(self, element_ids):
        if self.disable:
            return None
        element_ids = self.sanitize_ids(element_ids)
        return self.element_embed(element_ids)  # [B,d_model]


# ======================================================
# TSCG:文本语义引导(可选)
# ======================================================
class SemanticGuidance(nn.Module):
    def __init__(self, d_model, llm_dim, num_heads=4):
        super().__init__()
        self.text_proj = nn.Linear(llm_dim, d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, patch_feat, caption_emb):
        t = self.text_proj(caption_emb).unsqueeze(1)  # [B,1,d]
        out, _ = self.cross_attn(
            patch_feat,  # Q
            t,           # K
            t            # V
        )
        return patch_feat + out


# ======================================================
# 主模型
# ======================================================
class Model_Geo_Num_With_Meta(nn.Module):
    def __init__(self, configs):
        super().__init__()

        # ===============基础参数==============
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.args = configs
        self.use_tscg = getattr(configs, "use_tscg", False)

        self.d_model = configs.d_model
        self.enc_in = configs.enc_in  # 通常=1
        self.patch_adaptive = getattr(configs, "patch_adaptive", 0)

        # GeoStyle element数:默认46
        self.use_element = getattr(configs, "use_element", True)
        self.num_element = getattr(configs, "num_element", 46)

        # GeoStyle group数:默认44（你数据里Unique group count=44）
        self.use_group = getattr(configs, "use_group", True)
        self.num_group = getattr(configs, "num_group", 44)

        # Group Embedding（城市/群组）
        if self.use_group:
            self.group_embed = nn.Embedding(self.num_group, configs.d_model)
            nn.init.normal_(self.group_embed.weight, std=0.02)
        else:
            self.group_embed = None

        # LLM embedding维度:默认768
        self.llm_dim = getattr(configs, "llm_dim", 768)
        

        # patch参数
        patch_len = getattr(configs, "patch_len", 16)
        stride = getattr(configs, "stride", patch_len // 2)
        padding = stride

        # ===============Patch Embedding==============
        if self.patch_adaptive == 0:
            # 单尺度patch
            self.patch_embedding = PatchEmbedding(
                configs.d_model, patch_len, stride, padding, configs.dropout
            )
            self.num_patches = int((configs.seq_len - patch_len) / stride + 2)

        else:
            # 多尺度patch
            # scale1:8/4,scale2:16/8,scale3:32/16
            self.num_patches1 = int((configs.seq_len - 8) / 4 + 2)
            self.num_patches2 = int((configs.seq_len - 16) / 8 + 2)
            self.num_patches3 = int((configs.seq_len - 32) / 16 + 2)

            total_patches = self.num_patches2  # 以16/8为主尺度

            # pred_len→ratio映射
            if self.pred_len <= 26:
                ratios = [1.0, 0.0, 0.0]
            elif self.pred_len <= 96:
                ratios = [0.75, 0.25, 0.0]
            elif self.pred_len <= 192:
                ratios = [5 / 12, 7 / 12, 0.0]
            else:
                ratios = [1 / 6, 1 / 2, 1 / 3]

            self.k1 = int(ratios[0] * total_patches)
            self.k2 = int(ratios[1] * total_patches)
            self.k3 = total_patches - self.k1 - self.k2

            # 安全裁剪
            self.k1 = min(self.k1, self.num_patches1)
            self.k2 = min(self.k2, self.num_patches2)
            self.k3 = min(self.k3, self.num_patches3)

            remain = total_patches - (self.k1 + self.k2 + self.k3)
            if remain > 0:
                self.k1 += remain

            self.patch_embedding1 = PatchEmbedding(
                configs.d_model, patch_len=8, stride=4, padding=2, dropout=configs.dropout
            )
            self.patch_embedding2 = PatchEmbedding(
                configs.d_model, patch_len=16, stride=8, padding=4, dropout=configs.dropout
            )
            self.patch_embedding3 = PatchEmbedding(
                configs.d_model, patch_len=32, stride=16, padding=8, dropout=configs.dropout
            )

            self.num_patches = self.k1 + self.k2 + self.k3

        # ===============Semantic Guidance==============
        if self.use_tscg:
            self.semantic_guidance = SemanticGuidance(
                d_model=configs.d_model,
                llm_dim=self.llm_dim,
                num_heads=configs.n_heads
            )
        else:
            self.semantic_guidance = None


        # ===============Element Embedding==============
        self.element_embed = ElementEmbedding(
            d_model=configs.d_model,
            num_elements=self.num_element,
            disable=(not self.use_element)
        )

        # ===============Feature Fusion(Geo:patch+element+group)==============
        # 只要use_element或use_group任意一个开启，就需要fusion
        if self.use_element or self.use_group:
            in_dim = configs.d_model  # patch
            if self.use_element:
                in_dim += configs.d_model
            if self.use_group:
                in_dim += configs.d_model

            self.meta_fusion = nn.Sequential(
                nn.Linear(in_dim, configs.d_model * 2),
                nn.ReLU(),
                nn.LayerNorm(configs.d_model * 2),
                nn.Linear(configs.d_model * 2, configs.d_model),
                nn.Dropout(configs.dropout),
            )

        # ===============Transformer Encoder==============
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
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
            norm_layer=nn.Sequential(
                Transpose(1, 2),
                nn.BatchNorm1d(configs.d_model),
                Transpose(1, 2),
            ),
        )

        # ===============Adaptive Importance Mask==============
        self.aim = AdaptiveImportanceMask(
            d_model=configs.d_model,
            num_heads=configs.n_heads
        )

        # ===============Head==============
        if self.patch_adaptive == 0:
            head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        else:
            head_nf = configs.d_model * self.num_patches

        
        if self.task_name in ["long_term_forecast", "short_term_forecast", "long_term_forecast_meta"]:
            self.head = FlattenHead(
                configs.enc_in,
                head_nf,
                configs.pred_len,
                head_dropout=configs.dropout
            )

    # =========================================================
    # Patch Embedding
    # =========================================================
    def apply_patch_embeddings(self, x_enc):
        if self.patch_adaptive == 0:
            enc_out, n_vars = self.patch_embedding(x_enc)
            return enc_out, n_vars

        # 多尺度拼接:对齐参考代码
        enc1, n_vars1 = self.patch_embedding1(x_enc)
        enc2, n_vars2 = self.patch_embedding2(x_enc)
        enc3, n_vars3 = self.patch_embedding3(x_enc)

        enc3_part = enc3[:, : self.k3, :]
        start_idx = (self.num_patches2 - self.k2) // 2
        enc2_part = enc2[:, start_idx: start_idx + self.k2, :]
        enc1_part = enc1[:, -self.k1:, :]

        if self.k1 == 0:
            combined = torch.cat([enc3_part, enc2_part], dim=1)
        elif self.k3 == 0:
            combined = torch.cat([enc2_part, enc1_part], dim=1)
        else:
            combined = torch.cat([enc3_part, enc2_part, enc1_part], dim=1)

        return combined, n_vars1

    # =========================================================
    # forecast
    # =========================================================
    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, element_ids=None, group_ids=None, caption_emb=None):

        B = x_enc.shape[0]

        # 1)Instance Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # 2)Patch Embedding
        x_enc = x_enc.permute(0, 2, 1)  # [B,C,L]
        enc_out, n_vars = self.apply_patch_embeddings(x_enc)  # [B*C,P,d]或类似

        # 统一成[B,P,d]
        P = enc_out.shape[1]
        enc_out = enc_out.reshape(B, n_vars, P, -1)  # [B,C,P,d]
        enc_out = enc_out.mean(dim=1)                # [B,P,d_model]

        # 2.5)TSCG语义引导
        if self.use_tscg and (caption_emb is not None):
            enc_out = self.semantic_guidance(enc_out, caption_emb)

        # 3)Metadata Embedding: Element + Group（可选）
        meta_list = [enc_out]  # patch本体: [B,P,d]

        if self.use_element and (element_ids is not None):
            elem_emb = self.element_embed(element_ids)  # [B,d]
            elem_expanded = elem_emb.unsqueeze(1).expand(-1, enc_out.shape[1], -1)  # [B,P,d]
            meta_list.append(elem_expanded)

        if self.use_group and (group_ids is not None) and (self.group_embed is not None):
            # 安全裁剪，防止group_id越界（比如出现"44"这种）
            group_ids = group_ids.long().clamp(0, self.num_group - 1)
            grp_emb = self.group_embed(group_ids)  # [B,d]
            grp_expanded = grp_emb.unsqueeze(1).expand(-1, enc_out.shape[1], -1)  # [B,P,d]
            meta_list.append(grp_expanded)

        # 只要加了任何元信息，就走fusion；否则保持原enc_out
        if len(meta_list) > 1:
            enc_out = self.meta_fusion(torch.cat(meta_list, dim=-1))


        # 4)AIM
        alpha = self.aim(enc_out)  # [B,P]
        enc_out = enc_out * alpha.unsqueeze(-1)

        # 5)Encoder
        enc_out, _ = self.encoder(enc_out)  # [B,P,d]

        # 6)Reshape到head输入[B,C,d,P]
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))  # [B,C,P,d]
        enc_out = enc_out.permute(0, 1, 3, 2)  # [B,C,d,P]

        # 7)动态计算head_nf并调整线性层
        current_patch_num = enc_out.shape[-1]
        current_head_nf = self.d_model * current_patch_num
        
        if hasattr(self.head, 'linear') and self.head.linear.in_features != current_head_nf:
            self.head.linear = nn.Linear(current_head_nf, self.pred_len).to(enc_out.device)

        # 8)Head预测
        dec_out = self.head(enc_out)       # [B,C,pred_len]
        dec_out = dec_out.permute(0, 2, 1) # [B,pred_len,C]

        # 8)De-Normalization
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out

    # =========================================================
    # forward
    # =========================================================
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, element_ids=None, group_ids=None, caption_emb=None, city_ids=None, gender_ids=None, age_ids=None, **kwargs):

        if self.task_name in ["long_term_forecast", "short_term_forecast", "long_term_forecast_meta"]:
            # 兼容：如果外部还传city_ids，就用city_ids顶上
            if group_ids is None and city_ids is not None:
                group_ids = city_ids

            dec_out = self.forecast(
                x_enc,
                x_mark_enc=x_mark_enc,
                x_dec=x_dec,
                x_mark_dec=x_mark_dec,
                element_ids=element_ids,
                group_ids=group_ids,
                caption_emb=caption_emb,
            )
            return dec_out[:, -self.pred_len:, :]
        return None