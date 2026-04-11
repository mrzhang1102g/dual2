"""
Transformer编码器-解码器模块

本文件实现了Transformer的编码器和解码器结构，包括：
- ConvLayer: 卷积层，用于特征提取和降采样
- EncoderLayer: 编码器层，包含注意力机制和前馈网络
- Encoder: 编码器，由多个编码器层组成
- DecoderLayer: 解码器层，包含自注意力和交叉注意力
- Decoder: 解码器，由多个解码器层组成

这些模块可用于时间序列预测、机器翻译等序列到序列任务。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    """
    卷积层

    用于特征提取和降采样，采用1D卷积、批归一化、激活函数和最大池化。

    Args:
        c_in: 输入通道数

    Inputs:
        x: 输入张量，形状为 [B, L, C]

    Outputs:
        x: 输出张量，形状为 [B, L//2, C]
    """
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    """
    编码器层

    包含自注意力机制和前馈网络，是编码器的基本组成单元。

    Args:
        attention: 注意力机制实例
        d_model: 模型维度
        d_ff: 前馈网络隐藏层维度，默认为4 * d_model
        dropout: dropout概率，默认为0.1
        activation: 激活函数类型，默认为"relu"

    Inputs:
        x: 输入张量，形状为 [B, L, d_model]
        attn_mask: 注意力掩码，默认为None
        tau: 去平稳化因子tau，默认为None
        delta: 去平稳化因子delta，默认为None

    Outputs:
        x: 输出张量，形状为 [B, L, d_model]
        attn: 注意力矩阵，形状为 [B, H, L, L]
    """
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    """
    编码器

    由多个编码器层组成，可选地包含卷积层用于特征提取。

    Args:
        attn_layers: 注意力层列表
        conv_layers: 卷积层列表，默认为None
        norm_layer: 归一化层，默认为None

    Inputs:
        x: 输入张量，形状为 [B, L, d_model]
        attn_mask: 注意力掩码，默认为None
        tau: 去平稳化因子tau，默认为None
        delta: 去平稳化因子delta，默认为None

    Outputs:
        x: 输出张量，形状为 [B, L, d_model]
        attns: 注意力矩阵列表
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    解码器层

    包含自注意力机制、交叉注意力机制和前馈网络，是解码器的基本组成单元。

    Args:
        self_attention: 自注意力机制实例
        cross_attention: 交叉注意力机制实例
        d_model: 模型维度
        d_ff: 前馈网络隐藏层维度，默认为4 * d_model
        dropout: dropout概率，默认为0.1
        activation: 激活函数类型，默认为"relu"

    Inputs:
        x: 解码器输入张量，形状为 [B, L, d_model]
        cross: 编码器输出张量，形状为 [B, S, d_model]
        x_mask: 解码器自注意力掩码，默认为None
        cross_mask: 交叉注意力掩码，默认为None
        tau: 去平稳化因子tau，默认为None
        delta: 去平稳化因子delta，默认为None

    Outputs:
        x: 输出张量，形状为 [B, L, d_model]
    """
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    """
    解码器

    由多个解码器层组成，可选地包含投影层用于输出预测。

    Args:
        layers: 解码器层列表
        norm_layer: 归一化层，默认为None
        projection: 投影层，默认为None

    Inputs:
        x: 解码器输入张量，形状为 [B, L, d_model]
        cross: 编码器输出张量，形状为 [B, S, d_model]
        x_mask: 解码器自注意力掩码，默认为None
        cross_mask: 交叉注意力掩码，默认为None
        tau: 去平稳化因子tau，默认为None
        delta: 去平稳化因子delta，默认为None

    Outputs:
        x: 输出张量，形状为 [B, L, d_model] 或 [B, L, output_dim]
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x