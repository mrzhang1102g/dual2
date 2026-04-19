"""
实验基类

负责：
- 模型注册
- 设备管理
- 构建模型
"""

import os
import torch
from utils.logger import log, log_warning, log_error

# ===== FIT 系列 =====
from models import Model_Fit_Num
from models import Model_Fit_Num_With_Meta
from models import Model_Fit_Num_With_Meta_Semantic
from models import Model_Fit_Fusion

# ===== Geo 系列 =====
from models import Model_Geo_Num
from models import Model_Geo_Num_With_Meta
from models import Model_Geo_Fusion


class Exp_Basic(object):
    """
    实验基类
    
    负责：
    - 模型注册
    - 设备管理
    - 构建模型
    """

    def __init__(self, args):
        """
        初始化实验基类
        
        参数:
            args: 命令行参数
        """
        self.args = args

        # =====================================================
        # 模型注册表
        # =====================================================
        self.model_dict = {
            # ---------- FIT ----------
            'Model_Fit_Num': Model_Fit_Num,
            'Model_Fit_Num_With_Meta': Model_Fit_Num_With_Meta,
            'Model_Fit_Num_With_Meta_Semantic': Model_Fit_Num_With_Meta_Semantic,

            # ---------- GeoStyle ----------
            'Model_Geo_Num': Model_Geo_Num,
            'Model_Geo_Num_With_Meta': Model_Geo_Num_With_Meta,
            'Model_Geo_Fusion': Model_Geo_Fusion,
        }

        if Model_Fit_Fusion is not None:
            self.model_dict['Model_Fit_Fusion'] = Model_Fit_Fusion

        # =====================================================
        # 设备
        # =====================================================
        self.device = self._acquire_device()

        # =====================================================
        # 构建模型
        # =====================================================
        assert args.model in self.model_dict, \
            f"❌ Unknown model: {args.model}, available: {list(self.model_dict.keys())}"

        self.model = self._build_model(self.args).to(self.device)

    # =========================================================
    # 子类必须实现
    # =========================================================
    def _build_model(self, args):
        """
        构建模型
        
        参数:
            args: 命令行参数
            
        返回:
            model: 构建好的模型
        """
        raise NotImplementedError

    # =========================================================
    # 设备管理
    # =========================================================
    def _acquire_device(self):
        """
        获取设备
        
        返回:
            device: 设备对象
        """
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                str(self.args.gpu)
                if not self.args.use_multi_gpu
                else self.args.devices
            )
            device = torch.device(f'cuda:{self.args.gpu}')
            self.log(f'Use GPU: cuda:{self.args.gpu}')

        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            self.log('Use GPU: mps')

        else:
            device = torch.device('cpu')
            self.log('Use CPU')

        return device

    def log(self, message, level='info'):
        """
        统一日志输出
        
        参数:
            message: 日志消息
            level: 日志级别，可选值: 'info', 'warn', 'error'
        """
        if level == 'info':
            log(message)
        elif level == 'warn':
            log_warning(message)
        elif level == 'error':
            log_error(message)
        else:
            log(message)

    def print_trainable_parameters(self):
        """
        打印可学习参数量
        """
        self.log("\n" + "=" * 80)
        self.log("Trainable Parameters")
        self.log("=" * 80)

        total_params, trainable_params = 0, 0
        uninitialized_count = 0
        for name, param in self.model.named_parameters():
            try:
                n = param.numel()
                total_params += n
                if param.requires_grad:
                    trainable_params += n
                    self.log(f"[TRAIN] {name:60s} | shape={tuple(param.shape)}")
                else:
                    self.log(f"[FROZEN] {name:60s} | shape={tuple(param.shape)}")
            except ValueError:
                # 跳过未初始化的参数
                uninitialized_count += 1
                self.log(f"[UNINIT] {name:60s} | 未初始化，跳过")

        self.log("-" * 80)
        self.log(f"Total params:     {total_params:,}")
        self.log(f"Trainable params: {trainable_params:,}")
        if uninitialized_count > 0:
            self.log(f"Uninitialized params: {uninitialized_count:,} (已在首次前向传播时初始化)")
        self.log("=" * 80 + "\n")

    # =========================================================
    # 以下接口由具体 Exp 实现
    # =========================================================
    def _get_data(self, *args, **kwargs):
        """
        获取数据
        """
        raise NotImplementedError

    def vali(self, *args, **kwargs):
        """
        验证模型
        """
        raise NotImplementedError

    def train(self, *args, **kwargs):
        """
        训练模型
        """
        raise NotImplementedError

    def test(self, *args, **kwargs):
        """
        测试模型
        """
        raise NotImplementedError
