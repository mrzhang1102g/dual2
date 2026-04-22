"""实验基类。

负责模型注册、设备选择和通用日志输出，
具体训练/验证/测试流程由各子类实现。
"""

import os

import torch

from models import Model_Fit_Fusion
from models import Model_Fit_Num
from models import Model_Fit_Num_With_Meta
from models import Model_Geo_Fusion
from models import Model_Geo_Num
from models import Model_Geo_Num_With_Meta
from utils.logger import log, log_error, log_warning


class Exp_Basic:
    """所有实验类的公共基类。"""

    def __init__(self, args):
        self.args = args

        # 统一维护模型注册表，避免各实验文件重复导入判断。
        self.model_dict = {
            "Model_Fit_Num": Model_Fit_Num,
            "Model_Fit_Num_With_Meta": Model_Fit_Num_With_Meta,
            "Model_Fit_Fusion": Model_Fit_Fusion,
            "Model_Geo_Num": Model_Geo_Num,
            "Model_Geo_Num_With_Meta": Model_Geo_Num_With_Meta,
            "Model_Geo_Fusion": Model_Geo_Fusion,
        }

        self.device = self._acquire_device()

        assert args.model in self.model_dict, (
            f"Unknown model: {args.model}, available: {list(self.model_dict.keys())}"
        )
        self.model = self._build_model(self.args).to(self.device)

    def _build_model(self, args):
        """由子类负责实例化具体模型。"""
        raise NotImplementedError

    def _acquire_device(self):
        """根据参数选择 CPU / CUDA / MPS。"""
        if self.args.use_gpu and self.args.gpu_type == "cuda":
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            )
            device = torch.device(f"cuda:{self.args.gpu}")
            self.log(f"Use GPU: cuda:{self.args.gpu}")
        elif self.args.use_gpu and self.args.gpu_type == "mps":
            device = torch.device("mps")
            self.log("Use GPU: mps")
        else:
            device = torch.device("cpu")
            self.log("Use CPU")
        return device

    def log(self, message, level="info"):
        """统一日志出口。"""
        if level == "info":
            log(message)
        elif level == "warn":
            log_warning(message)
        elif level == "error":
            log_error(message)
        else:
            log(message)

    def print_trainable_parameters(self):
        """打印参数冻结状态，便于检查融合实验是否只训练了预期模块。"""
        self.log("\n" + "=" * 80)
        self.log("Trainable Parameters")
        self.log("=" * 80)

        total_params = 0
        trainable_params = 0
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
                # LazyLinear 等模块在首个前向前可能仍未初始化。
                uninitialized_count += 1
                self.log(f"[UNINIT] {name:60s} | 尚未初始化，训练前会自动完成")

        self.log("-" * 80)
        self.log(f"Total params:     {total_params:,}")
        self.log(f"Trainable params: {trainable_params:,}")
        if uninitialized_count > 0:
            self.log(f"Uninitialized params: {uninitialized_count:,} (会在首次前向时初始化)")
        self.log("=" * 80 + "\n")

    def _get_data(self, *args, **kwargs):
        raise NotImplementedError

    def vali(self, *args, **kwargs):
        raise NotImplementedError

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def test(self, *args, **kwargs):
        raise NotImplementedError
