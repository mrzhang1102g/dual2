"""
评估指标模块

本文件实现了时间序列预测中常用的评估指标，包括：
- RSE: 相对平方误差
- CORR: 相关系数
- MAE: 平均绝对误差
- MSE: 均方误差
- RMSE: 均方根误差
- MAPE: 平均绝对百分比误差
- MSPE: 均方百分比误差
- WAPE: 加权绝对百分比误差
- metric: 综合计算所有指标的函数

这些指标用于评估时间序列预测模型的性能。
"""
import numpy as np
import torch


def RSE(pred, true):
    """
    相对平方误差

    计算预测值与真实值之间的相对平方误差，衡量预测的相对准确性。

    Args:
        pred: 预测值数组
        true: 真实值数组

    Returns:
        相对平方误差值
    """
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(
        np.sum((true - true.mean()) ** 2)
    )


def CORR(pred, true):
    """
    相关系数

    计算预测值与真实值之间的相关系数，衡量两者之间的线性关系强度。

    Args:
        pred: 预测值数组
        true: 真实值数组

    Returns:
        相关系数值
    """
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    """
    平均绝对误差

    计算预测值与真实值之间的平均绝对误差。

    Args:
        pred: 预测值数组
        true: 真实值数组

    Returns:
        平均绝对误差值
    """
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    """
    均方误差

    计算预测值与真实值之间的均方误差。

    Args:
        pred: 预测值数组
        true: 真实值数组

    Returns:
        均方误差值
    """
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    """
    均方根误差

    计算预测值与真实值之间的均方根误差。

    Args:
        pred: 预测值数组
        true: 真实值数组

    Returns:
        均方根误差值
    """
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    """
    平均绝对百分比误差

    计算预测值与真实值之间的平均绝对百分比误差，忽略真实值为0的情况。

    Args:
        pred: 预测值数组
        true: 真实值数组

    Returns:
        平均绝对百分比误差值，如果所有真实值都为0则返回nan
    """
    mask = true != 0
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((true[mask] - pred[mask]) / true[mask]))


def MSPE(pred, true):
    """
    均方百分比误差

    计算预测值与真实值之间的均方百分比误差，忽略真实值为0的情况。

    Args:
        pred: 预测值数组
        true: 真实值数组

    Returns:
        均方百分比误差值，如果所有真实值都为0则返回nan
    """
    mask = true != 0
    if not np.any(mask):
        return np.nan
    return np.mean(np.square((true[mask] - pred[mask]) / true[mask]))


def WAPE(pred, true):
    """
    加权绝对百分比误差

    计算预测值与真实值之间的加权绝对百分比误差，权重为真实值的绝对值。

    Args:
        pred: 预测值数组
        true: 真实值数组

    Returns:
        加权绝对百分比误差值，如果所有真实值的绝对值和为0则返回nan
    """
    absolute_error = np.abs(true - pred)
    denom = np.sum(np.abs(true))
    if denom == 0:
        return np.nan
    return np.sum(absolute_error) / denom


def metric(pred, true):
    """
    综合计算所有评估指标

    计算并返回预测值与真实值之间的所有评估指标。

    Args:
        pred: 预测值数组
        true: 真实值数组

    Returns:
        mae: 平均绝对误差
        mse: 均方误差
        rmse: 均方根误差
        mape: 平均绝对百分比误差
        mspe: 均方百分比误差
        wape: 加权绝对百分比误差
    """
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    wape = WAPE(pred, true)

    return mae, mse, rmse, mape, mspe, wape