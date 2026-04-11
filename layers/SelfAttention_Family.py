"""
注意力机制模块集合

本文件实现了多种注意力机制，包括：
- DSAttention: 去平稳化注意力，使用学习的去平稳化因子重新缩放注意力分数
- FullAttention: 标准全注意力机制
- ProbAttention: 概率注意力，通过采样减少计算复杂度
- AttentionLayer: 注意力层包装器，处理投影和多头注意力
- ReformerLayer: 基于LSH的高效注意力实现
- TwoStageAttentionLayer: 两阶段注意力层，处理时间和维度两个维度的注意力

这些注意力机制可用于时间序列预测、自然语言处理等任务。
"""
import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange, repeat


class DSAttention(nn.Module):
    """
    去平稳化注意力机制

    该注意力机制通过学习的去平稳化因子(tau和delta)对注意力分数进行重新缩放，
    以适应时间序列数据的非平稳特性。

    Args:
        mask_flag (bool): 是否使用掩码，默认为True
        factor (int): 概率注意力的采样因子，默认为5
        scale (float): 注意力分数缩放因子，默认为None
        attention_dropout (float): 注意力 dropout 概率，默认为0.1
        output_attention (bool): 是否输出注意力矩阵，默认为False

    Inputs:
        queries: 查询向量，形状为 [B, L, H, E]
        keys: 键向量，形状为 [B, S, H, E]
        values: 值向量，形状为 [B, S, H, D]
        attn_mask: 注意力掩码，默认为None
        tau: 去平稳化因子tau，默认为None
        delta: 去平稳化因子delta，默认为None

    Outputs:
        V: 注意力加权的值向量，形状为 [B, L, H, D]
        A: 注意力矩阵，形状为 [B, H, L, S]，仅当output_attention为True时返回
    """
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x 1
        delta = 0.0 if delta is None else delta.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x S

        # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class FullAttention(nn.Module):
    """
    标准全注意力机制

    实现了标准的Transformer注意力机制，计算所有查询和键之间的注意力分数。

    Args:
        mask_flag (bool): 是否使用掩码，默认为True
        factor (int): 概率注意力的采样因子，默认为5
        scale (float): 注意力分数缩放因子，默认为None
        attention_dropout (float): 注意力 dropout 概率，默认为0.1
        output_attention (bool): 是否输出注意力矩阵，默认为False

    Inputs:
        queries: 查询向量，形状为 [B, L, H, E]
        keys: 键向量，形状为 [B, S, H, E]
        values: 值向量，形状为 [B, S, H, D]
        attn_mask: 注意力掩码，默认为None
        tau: 未使用，保持接口一致性
        delta: 未使用，保持接口一致性

    Outputs:
        V: 注意力加权的值向量，形状为 [B, L, H, D]
        A: 注意力矩阵，形状为 [B, H, L, S]，仅当output_attention为True时返回
    """
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class ProbAttention(nn.Module):
    """
    概率注意力机制

    通过采样减少计算复杂度，适用于长序列输入。

    Args:
        mask_flag (bool): 是否使用掩码，默认为True
        factor (int): 采样因子，默认为5
        scale (float): 注意力分数缩放因子，默认为None
        attention_dropout (float): 注意力 dropout 概率，默认为0.1
        output_attention (bool): 是否输出注意力矩阵，默认为False

    Inputs:
        queries: 查询向量，形状为 [B, L, H, D]
        keys: 键向量，形状为 [B, S, H, D]
        values: 值向量，形状为 [B, S, H, D]
        attn_mask: 注意力掩码，默认为None
        tau: 未使用，保持接口一致性
        delta: 未使用，保持接口一致性

    Outputs:
        context: 注意力加权的上下文向量，形状为 [B, H, L, D]
        attn: 注意力矩阵，形状为 [B, H, L, S]，仅当output_attention为True时返回
    """
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        """
        计算概率QK注意力分数

        Args:
            Q: 查询向量，形状为 [B, H, L_Q, D]
            K: 键向量，形状为 [B, H, L_K, D]
            sample_k: 采样的键数量
            n_top: 选择的查询数量

        Returns:
            Q_K: 注意力分数，形状为 [B, H, n_top, L_K]
            M_top: 选择的查询索引，形状为 [B, H, n_top]
        """
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(
            L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        """
        获取初始上下文向量

        Args:
            V: 值向量，形状为 [B, H, L_V, D]
            L_Q: 查询序列长度

        Returns:
            contex: 初始上下文向量，形状为 [B, H, L_Q, D]
        """
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        """
        更新上下文向量

        Args:
            context_in: 初始上下文向量，形状为 [B, H, L_Q, D]
            V: 值向量，形状为 [B, H, L_V, D]
            scores: 注意力分数，形状为 [B, H, n_top, L_K]
            index: 选择的查询索引，形状为 [B, H, n_top]
            L_Q: 查询序列长度
            attn_mask: 注意力掩码

        Returns:
            context_in: 更新后的上下文向量，形状为 [B, H, L_Q, D]
            attns: 注意力矩阵，形状为 [B, H, L_V, L_V]，仅当output_attention为True时返回
        """
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                                                  None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
                 np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    """
    注意力层包装器

    处理查询、键、值的投影和多头注意力计算。

    Args:
        attention: 注意力机制实例
        d_model: 模型维度
        n_heads: 注意力头数量
        d_keys: 键维度，默认为None，此时为d_model // n_heads
        d_values: 值维度，默认为None，此时为d_model // n_heads

    Inputs:
        queries: 查询向量，形状为 [B, L, d_model]
        keys: 键向量，形状为 [B, S, d_model]
        values: 值向量，形状为 [B, S, d_model]
        attn_mask: 注意力掩码，默认为None
        tau: 去平稳化因子tau，默认为None
        delta: 去平稳化因子delta，默认为None

    Outputs:
        out: 注意力层输出，形状为 [B, L, d_model]
        attn: 注意力矩阵，形状为 [B, H, L, S]，仅当output_attention为True时返回
    """
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class ReformerLayer(nn.Module):
    """
    Reformer注意力层

    基于局部敏感哈希(LSH)的高效注意力实现，适用于长序列。

    Args:
        attention: 未使用，保持接口一致性
        d_model: 模型维度
        n_heads: 注意力头数量
        d_keys: 未使用，保持接口一致性
        d_values: 未使用，保持接口一致性
        causal: 是否使用因果掩码，默认为False
        bucket_size: 桶大小，默认为4
        n_hashes: 哈希函数数量，默认为4

    Inputs:
        queries: 查询向量，形状为 [B, N, d_model]
        keys: 未使用，保持接口一致性
        values: 未使用，保持接口一致性
        attn_mask: 未使用，保持接口一致性
        tau: 未使用，保持接口一致性
        delta: 未使用，保持接口一致性

    Outputs:
        queries: 注意力处理后的查询向量，形状为 [B, N, d_model]
        None: 始终返回None作为注意力矩阵
    """
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        """
        调整序列长度以适应Reformer的要求

        Args:
            queries: 查询向量，形状为 [B, N, C]

        Returns:
            调整长度后的查询向量
        """
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        # in Reformer: defalut queries=keys
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None


class TwoStageAttentionLayer(nn.Module):
    """
    两阶段注意力层

    处理时间和维度两个维度的注意力，适用于多维时间序列数据。

    Args:
        configs: 配置对象
        seg_num: 序列段数
        factor: 注意力因子
        d_model: 模型维度
        n_heads: 注意力头数量
        d_ff: 前馈网络隐藏层维度，默认为4 * d_model
        dropout: dropout概率，默认为0.1

    Inputs:
        x: 输入张量，形状为 [batch_size, Data_dim(D), Seg_num(L), d_model]
        attn_mask: 注意力掩码，默认为None
        tau: 未使用，保持接口一致性
        delta: 未使用，保持接口一致性

    Outputs:
        final_out: 输出张量，形状为 [batch_size, Data_dim(D), Seg_num(L), d_model]
    """
    def __init__(self, configs,
                 seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=False), d_model, n_heads)
        self.dim_sender = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                       output_attention=False), d_model, n_heads)
        self.dim_receiver = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                         output_attention=False), d_model, n_heads)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc, attn = self.time_attention(
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None)
        dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)

        return final_out