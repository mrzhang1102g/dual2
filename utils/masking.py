"""
掩码工具模块

本文件实现了注意力机制中使用的掩码类，包括：
- TriangularCausalMask: 三角形因果掩码，用于确保自注意力机制只能看到当前位置及之前的位置
- ProbMask: 概率掩码，用于概率注意力机制中

这些掩码类在Transformer-based模型中用于控制注意力的范围，特别是在时间序列预测任务中。
"""
import torch


class TriangularCausalMask():
    """
    三角形因果掩码

    用于Transformer自注意力机制，确保模型只能看到当前位置及之前的位置，
    防止未来信息泄露。

    Args:
        B: 批量大小
        L: 序列长度
        device: 设备，默认为"cpu"

    Attributes:
        mask: 掩码张量，形状为 [B, 1, L, L]，上三角部分为True（被屏蔽）
    """
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        """
        获取掩码张量

        Returns:
            掩码张量，形状为 [B, 1, L, L]
        """
        return self._mask


class ProbMask():
    """
    概率掩码

    用于概率注意力机制，根据索引和分数生成掩码。

    Args:
        B: 批量大小
        H: 注意力头数量
        L: 序列长度
        index: 索引张量
        scores: 注意力分数张量
        device: 设备，默认为"cpu"

    Attributes:
        mask: 掩码张量，形状与scores相同
    """
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        """
        获取掩码张量

        Returns:
            掩码张量，形状与scores相同
        """
        return self._mask