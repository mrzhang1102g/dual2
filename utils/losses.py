# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
损失函数模块

本文件实现了时间序列预测中常用的损失函数，包括：
- mape_loss: 平均绝对百分比误差损失
- smape_loss: 对称平均绝对百分比误差损失
- mase_loss: 平均绝对缩放误差损失

这些损失函数用于评估和优化时间序列预测模型的性能。
"""

import torch as t
import torch.nn as nn
import numpy as np


def divide_no_nan(a, b):
    """
    执行除法操作，将结果中的NaN或Inf替换为0。

    Args:
        a: 被除数张量
        b: 除数张量

    Returns:
        除法结果，其中NaN和Inf被替换为0
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    """
    平均绝对百分比误差损失

    定义为预测值与真实值之间的绝对百分比误差的平均值。
    参考: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

    Inputs:
        insample: 样本内数据，形状为 [batch, time_i]
        freq: 时间序列频率
        forecast: 预测值，形状为 [batch, time_o]
        target: 目标值，形状为 [batch, time_o]
        mask: 0/1掩码，形状为 [batch, time_o]

    Outputs:
        损失值
    """
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    """
    对称平均绝对百分比误差损失

    定义为预测值与真实值之间的对称绝对百分比误差的平均值。
    参考: https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

    Inputs:
        insample: 样本内数据，形状为 [batch, time_i]
        freq: 时间序列频率
        forecast: 预测值，形状为 [batch, time_o]
        target: 目标值，形状为 [batch, time_o]
        mask: 0/1掩码，形状为 [batch, time_o]

    Outputs:
        损失值（乘以200以保持与传统定义一致）
    """
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    """
    平均绝对缩放误差损失

    定义为预测值与真实值之间的绝对误差，除以样本内数据的平均绝对季节性误差。
    参考: "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

    Inputs:
        insample: 样本内数据，形状为 [batch, time_i]
        freq: 时间序列频率
        forecast: 预测值，形状为 [batch, time_o]
        target: 目标值，形状为 [batch, time_o]
        mask: 0/1掩码，形状为 [batch, time_o]

    Outputs:
        损失值
    """
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)