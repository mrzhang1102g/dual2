"""
工具函数模块

本文件实现了各种实用工具函数，包括：
- adjust_learning_rate: 调整学习率
- EarlyStopping: 早停机制
- dotdict: 支持点号访问的字典类
- StandardScaler: 数据标准化器
- visual: 结果可视化
- adjustment: 异常检测结果调整
- cal_accuracy: 计算准确率

这些工具函数用于模型训练、评估和结果分析。
"""
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    """
    调整学习率

    根据训练轮次和指定的调整策略调整学习率。

    Args:
        optimizer: 优化器实例
        epoch: 当前训练轮次
        args: 包含学习率调整策略的参数对象

    Returns:
        None
    """
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    elif args.lradj == 'type3':
        # KERN-style: 每20个epoch衰减为0.1倍
        lr_adjust = {epoch: args.learning_rate * (0.1 ** (epoch // 20))}
    else:
        return  # 不调整学习率
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        if args.learning_rate != 0:
            scale = lr / args.learning_rate
        else:
            scale = 1.0

        updated_lrs = []
        for index, param_group in enumerate(optimizer.param_groups):
            base_lr = param_group.setdefault('base_lr', param_group['lr'])
            new_lr = base_lr * scale
            param_group['lr'] = new_lr
            group_name = param_group.get('group_name', f'group{index}')
            updated_lrs.append(f'{group_name}={new_lr}')

        if len(updated_lrs) == 1:
            print('Updating learning rate to {}'.format(lr))
        else:
            print('Updating learning rates: {}'.format(', '.join(updated_lrs)))


class EarlyStopping:
    """
    早停机制

    当验证损失不再改善时停止训练，防止过拟合。

    Args:
        patience: 容忍验证损失不改善的轮次数，默认为7
        verbose: 是否打印详细信息，默认为False
        delta: 认为验证损失改善的最小变化量，默认为0

    Attributes:
        patience: 容忍验证损失不改善的轮次数
        verbose: 是否打印详细信息
        counter: 验证损失不改善的计数器
        best_score: 最佳验证损失的负数
        early_stop: 是否触发早停
        val_loss_min: 最小验证损失
        delta: 认为验证损失改善的最小变化量
    """
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path, task = 'None'):
        """
        检查是否应该早停

        Args:
            val_loss: 当前验证损失
            model: 模型实例
            path: 模型保存路径
            task: 任务类型，默认为'None'

        Returns:
            None
        """
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, task)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, task)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, task):
        """
        保存模型检查点

        Args:
            val_loss: 当前验证损失
            model: 模型实例
            path: 模型保存路径
            task: 任务类型

        Returns:
            None
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if task == 'coarse':
            torch.save(model.state_dict(), path + '/' + 'checkpoint_coarse.pth')
        elif task == 'fine':
            torch.save(model.state_dict(), path + '/' + 'checkpoint_fine.pth')
        else:
            torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """
    支持点号访问的字典类

    允许使用点号（如obj.key）访问字典属性，而不是传统的字典访问方式（如obj['key']）。
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    """
    数据标准化器

    对数据进行标准化处理，将数据转换为均值为0、标准差为1的分布。

    Args:
        mean: 均值
        std: 标准差

    Attributes:
        mean: 均值
        std: 标准差
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        """
        对数据进行标准化

        Args:
            data: 输入数据

        Returns:
            标准化后的数据
        """
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """
        对数据进行反标准化，恢复原始尺度

        Args:
            data: 标准化后的数据

        Returns:
            反标准化后的数据
        """
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf', history_len=None):
    """
    结果可视化（历史+预测）

    绘制时间序列预测结果，包括历史数据和预测数据。

    Args:
        true: 真实值，形状为 [历史长度 + 未来长度]
        preds: 预测值，形状为 [历史长度 + 未来长度]，默认为None
        name: 保存图片的路径，默认为'./pic/test.pdf'
        history_len: 历史数据的长度，默认为None

    Returns:
        None
    """
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(10, 4))

    total_len = len(true)

    # 如果没给 history_len，就退化成原始行为（向后兼容）
    if history_len is None or preds is None:
        plt.plot(true, label='Ground Truth', linewidth=2)
        if preds is not None:
            plt.plot(preds, label='Prediction', linewidth=2)
        plt.legend()
        plt.savefig(name, bbox_inches='tight')
        plt.close()
        return

    T = history_len

    x_hist = np.arange(T)
    x_future = np.arange(T, total_len)

    # ===== 历史（只画一条真实线）=====
    plt.plot(
        x_hist,
        true[:T],
        color='gray',
        linestyle='--',
        linewidth=2,
        label='History (Observed)'
    )

    # 预测起点
    plt.axvline(
        x=T - 1,
        color='black',
        linestyle=':',
        linewidth=1,
        alpha=0.8
    )

    # ===== 未来：真实 vs 预测 =====
    plt.plot(
        x_future,
        true[T:],
        linewidth=2,
        label='Ground Truth (Future)'
    )

    plt.plot(
        x_future,
        preds[T:],
        linewidth=2,
        label='Prediction'
    )

    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(name, bbox_inches='tight')
    plt.close()


def adjustment(gt, pred):
    """
    异常检测结果调整

    对异常检测的预测结果进行调整，确保异常区间的连续性。

    Args:
        gt: 真实异常标签，形状为 [序列长度]
        pred: 预测异常标签，形状为 [序列长度]

    Returns:
        gt: 原始真实标签
        pred: 调整后的预测标签
    """
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    """
    计算准确率

    计算预测值与真实值之间的准确率。

    Args:
        y_pred: 预测值数组
        y_true: 真实值数组

    Returns:
        准确率值
    """
    return np.mean(y_pred == y_true)
