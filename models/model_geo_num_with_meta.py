"""Geo numeric forecasting model with optional element/group metadata."""

import torch
from torch import nn

from layers.Embed import PatchEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer


SUPPORTED_GEO_NUM_WITH_META_TASK = "geo_num_with_meta"


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
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class AdaptiveImportanceMask(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.msa = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.linear_alpha = nn.Linear(d_model, 1)

    def forward(self, patch_tokens):
        attended, _ = self.msa(patch_tokens, patch_tokens, patch_tokens)
        alpha = torch.sigmoid(self.linear_alpha(attended))
        return alpha.squeeze(-1)


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
        return self.element_embed(element_ids)


class SemanticGuidance(nn.Module):
    def __init__(self, d_model, llm_dim, num_heads=4):
        super().__init__()
        self.text_proj = nn.Linear(llm_dim, d_model)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)

    def forward(self, patch_feat, caption_emb):
        text_token = self.text_proj(caption_emb).unsqueeze(1)
        out, _ = self.cross_attn(patch_feat, text_token, text_token)
        return patch_feat + out


class Model_Geo_Num_With_Meta(nn.Module):
    """Geo model that fuses numeric series with element/group metadata."""

    def __init__(self, configs):
        super().__init__()

        self.task_name = configs.task_name
        if self.task_name != SUPPORTED_GEO_NUM_WITH_META_TASK:
            raise ValueError(
                "Model_Geo_Num_With_Meta only supports "
                f"task_name={SUPPORTED_GEO_NUM_WITH_META_TASK}, got {self.task_name}"
            )

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.args = configs
        self.use_tscg = getattr(configs, "use_tscg", False)

        self.d_model = configs.d_model
        self.enc_in = configs.enc_in
        self.patch_adaptive = getattr(configs, "patch_adaptive", 0)

        self.use_element = getattr(configs, "use_element", True)
        self.num_element = getattr(configs, "num_element", 46)

        self.use_group = getattr(configs, "use_group", True)
        self.num_group = getattr(configs, "num_group", 44)
        if self.use_group:
            self.group_embed = nn.Embedding(self.num_group, configs.d_model)
            nn.init.normal_(self.group_embed.weight, std=0.02)
        else:
            self.group_embed = None

        self.llm_dim = getattr(configs, "llm_dim", 768)

        patch_len = getattr(configs, "patch_len", 16)
        stride = getattr(configs, "stride", patch_len // 2)
        padding = stride

        if self.patch_adaptive == 0:
            self.patch_embedding = PatchEmbedding(configs.d_model, patch_len, stride, padding, configs.dropout)
            self.num_patches = int((configs.seq_len - patch_len) / stride + 2)
        else:
            self.num_patches1 = int((configs.seq_len - 8) / 4 + 2)
            self.num_patches2 = int((configs.seq_len - 16) / 8 + 2)
            self.num_patches3 = int((configs.seq_len - 32) / 16 + 2)
            total_patches = self.num_patches2

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

            self.k1 = min(self.k1, self.num_patches1)
            self.k2 = min(self.k2, self.num_patches2)
            self.k3 = min(self.k3, self.num_patches3)

            remain = total_patches - (self.k1 + self.k2 + self.k3)
            if remain > 0:
                self.k1 += remain

            self.patch_embedding1 = PatchEmbedding(configs.d_model, patch_len=8, stride=4, padding=2, dropout=configs.dropout)
            self.patch_embedding2 = PatchEmbedding(configs.d_model, patch_len=16, stride=8, padding=4, dropout=configs.dropout)
            self.patch_embedding3 = PatchEmbedding(configs.d_model, patch_len=32, stride=16, padding=8, dropout=configs.dropout)
            self.num_patches = self.k1 + self.k2 + self.k3

        if self.use_tscg:
            self.semantic_guidance = SemanticGuidance(d_model=configs.d_model, llm_dim=self.llm_dim, num_heads=configs.n_heads)
        else:
            self.semantic_guidance = None

        self.element_embed = ElementEmbedding(
            d_model=configs.d_model,
            num_elements=self.num_element,
            disable=(not self.use_element),
        )

        if self.use_element or self.use_group:
            in_dim = configs.d_model
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

        if self.patch_adaptive == 0:
            head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        else:
            head_nf = configs.d_model * self.num_patches

        self.head = FlattenHead(configs.enc_in, head_nf, configs.pred_len, head_dropout=configs.dropout)

    def apply_patch_embeddings(self, x_enc):
        if self.patch_adaptive == 0:
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

    @staticmethod
    def _instance_normalize(x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x_norm = x_enc - means
        stdev = torch.sqrt(torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = x_norm / stdev
        return x_norm, means, stdev

    def _patchify(self, x_enc):
        batch_size = x_enc.shape[0]
        x_enc = x_enc.permute(0, 2, 1)
        patch_tokens, n_vars = self.apply_patch_embeddings(x_enc)
        patch_count = patch_tokens.shape[1]
        patch_tokens = patch_tokens.reshape(batch_size, n_vars, patch_count, -1)
        return patch_tokens, n_vars

    def _fuse_metadata(self, sequence_tokens, element_ids, group_ids):
        meta_list = [sequence_tokens]

        if self.use_element and element_ids is not None:
            elem_emb = self.element_embed(element_ids)
            elem_expanded = elem_emb.unsqueeze(1).expand(-1, sequence_tokens.shape[1], -1)
            meta_list.append(elem_expanded)

        if self.use_group and group_ids is not None and self.group_embed is not None:
            group_ids = group_ids.long().clamp(0, self.num_group - 1)
            grp_emb = self.group_embed(group_ids)
            grp_expanded = grp_emb.unsqueeze(1).expand(-1, sequence_tokens.shape[1], -1)
            meta_list.append(grp_expanded)

        if len(meta_list) > 1:
            sequence_tokens = self.meta_fusion(torch.cat(meta_list, dim=-1))
        return sequence_tokens

    def _encode_sequence(self, patch_tokens, element_ids, group_ids, caption_emb):
        sequence_tokens = patch_tokens.mean(dim=1)

        if self.use_tscg and caption_emb is not None:
            sequence_tokens = self.semantic_guidance(sequence_tokens, caption_emb)

        sequence_tokens = self._fuse_metadata(sequence_tokens, element_ids, group_ids)
        alpha = self.aim(sequence_tokens)
        sequence_tokens = sequence_tokens * alpha.unsqueeze(-1)
        encoded_sequence, _ = self.encoder(sequence_tokens)
        return encoded_sequence

    def _decode_forecast(self, encoded_tokens, means, stdev):
        dec_tokens = encoded_tokens.permute(0, 1, 3, 2)
        current_patch_num = dec_tokens.shape[-1]
        current_head_nf = self.d_model * current_patch_num
        if self.head.linear.in_features != current_head_nf:
            self.head.linear = nn.Linear(current_head_nf, self.pred_len).to(dec_tokens.device)

        dec_out = self.head(dec_tokens)
        dec_out = dec_out.permute(0, 2, 1)
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    @staticmethod
    def _pool_summary(encoded_tokens):
        return encoded_tokens.mean(dim=(1, 2))

    def _run_backbone(self, x_enc, element_ids, group_ids, caption_emb):
        x_norm, means, stdev = self._instance_normalize(x_enc)
        patch_tokens, _ = self._patchify(x_norm)
        encoded_sequence = self._encode_sequence(patch_tokens, element_ids, group_ids, caption_emb)
        encoded_tokens = encoded_sequence.unsqueeze(1)
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
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
        element_ids=None,
        group_ids=None,
        caption_emb=None,
    ):
        del x_mark_enc, x_dec, x_mark_dec
        return self._run_backbone(x_enc, element_ids, group_ids, caption_emb)

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, element_ids=None, group_ids=None, caption_emb=None):
        return self.extract_features(
            x_enc,
            x_mark_enc=x_mark_enc,
            x_dec=x_dec,
            x_mark_dec=x_mark_dec,
            element_ids=element_ids,
            group_ids=group_ids,
            caption_emb=caption_emb,
        )["forecast"]

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
        element_ids=None,
        group_ids=None,
        caption_emb=None,
    ):
        dec_out = self.forecast(
            x_enc,
            x_mark_enc=x_mark_enc,
            x_dec=x_dec,
            x_mark_dec=x_mark_dec,
            element_ids=element_ids,
            group_ids=group_ids,
            caption_emb=caption_emb,
        )
        return dec_out[:, -self.pred_len :, :]
