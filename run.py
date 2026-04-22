"""项目统一入口。

按任务类型组装参数、固定随机种子，并分发到 FIT / Geo 的训练与测试流程。
"""

import argparse
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch

from exp.exp_fit_fusion import Exp_Fit_Fusion
from exp.exp_fit_num import Exp_Fit_Num
from exp.exp_geo_fusion import Exp_Geo_Fusion
from exp.exp_geo_num import Exp_Geo_Num
from utils.logger import log
from utils.print_args import print_args


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_setting(args, with_timestamp=True):
    # 训练任务默认附带时间戳，便于区分多次实验输出目录。
    if with_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{args.model_id}_{timestamp}"
    return f"{args.model_id}"


def set_seed(seed):
    # 显式关闭 cudnn benchmark，优先保证实验可复现。
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    log(f"[Seed] fixed seed={seed}")


def add_runtime_args(parser):
    """运行时公共参数。"""
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--is_training", type=int, required=True, default=1)
    parser.add_argument("--model_id", type=str, required=True, default="test")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--itr", type=int, default=1)


def add_data_args(parser):
    """数据路径、切分和缩放相关参数。"""
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--root_path", type=str, default="./dataset/")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--features", type=str, default="S")
    parser.add_argument("--target", type=str, default="trend")
    parser.add_argument("--freq", type=str, default="w")
    parser.add_argument("--scale", type=str2bool, default=True)
    parser.add_argument("--inverse", action="store_true", default=True)

    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--fit_scaler_mode", type=str, default="train_only", choices=["train_only", "split_fit"])

    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_val_samples", type=int, default=-1)
    parser.add_argument("--max_test_samples", type=int, default=-1)


def add_sequence_args(parser):
    """历史窗口和预测窗口长度。"""
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=0)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--seasonal_patterns", type=str, default="Monthly")


def add_model_args(parser):
    """数值模型主干结构参数。"""
    parser.add_argument("--enc_in", type=int, default=1)
    parser.add_argument("--dec_in", type=int, default=1)
    parser.add_argument("--c_out", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--distil", action="store_false", default=True)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--num_kernels", type=int, default=6)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--d_conv", type=int, default=4)

    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--patch_adaptive", type=int, default=1)

    parser.add_argument("--channel_independence", type=int, default=1)
    parser.add_argument("--decomp_method", type=str, default="moving_avg")
    parser.add_argument("--use_norm", type=int, default=1)
    parser.add_argument("--down_sampling_layers", type=int, default=0)
    parser.add_argument("--down_sampling_window", type=int, default=1)
    parser.add_argument("--down_sampling_method", type=str, default=None)
    parser.add_argument("--seg_len", type=int, default=48)


def add_training_args(parser):
    """优化器、训练轮数和增强策略。"""
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--train_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=800)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--lr_num", type=float, default=1e-4)
    parser.add_argument("--lr_text", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--weight_decay_text", type=float, default=0.01)
    parser.add_argument("--loss", type=str, default="MAE")
    parser.add_argument("--lradj", type=str, default="type3")
    parser.add_argument("--adjust", type=int, default=1)
    parser.add_argument("--use_amp", action="store_true", default=False)
    parser.add_argument("--use_dtw", type=str2bool, default=False)

    parser.add_argument("--augmentation_ratio", type=int, default=0)
    parser.add_argument("--jitter", action="store_true")
    parser.add_argument("--scaling", action="store_true")
    parser.add_argument("--permutation", action="store_true")
    parser.add_argument("--randompermutation", action="store_true")
    parser.add_argument("--magwarp", action="store_true")
    parser.add_argument("--timewarp", action="store_true")
    parser.add_argument("--windowslice", action="store_true")
    parser.add_argument("--windowwarp", action="store_true")
    parser.add_argument("--rotation", action="store_true")
    parser.add_argument("--spawner", action="store_true")
    parser.add_argument("--dtwwarp", action="store_true")
    parser.add_argument("--shapedtwwarp", action="store_true")
    parser.add_argument("--wdba", action="store_true")
    parser.add_argument("--discdtw", action="store_true")
    parser.add_argument("--discsdtw", action="store_true")


def add_device_args(parser):
    """设备与多卡配置。"""
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpu_type", type=str, default="cuda")
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    parser.add_argument("--devices", type=str, default="0,1")


def add_common_fusion_args(parser):
    """FIT / Geo 融合实验共享参数。"""
    parser.add_argument("--num_model_path", type=str, default="")
    parser.add_argument("--caption_emb_path", type=str, default=None)
    parser.add_argument("--freeze_numerical", action="store_true")
    parser.add_argument("--disable_text", action="store_true")
    parser.add_argument("--text_mode", type=str, default="direct", choices=["direct", "residual"])
    parser.add_argument("--num_feat_dim", type=int, default=48)
    parser.add_argument("--fusion_dropout", type=float, default=0.1)
    parser.add_argument("--fusion_hidden", type=int, default=96)


def add_fit_fusion_args(parser):
    """FIT 融合流额外参数。"""
    add_common_fusion_args(parser)
    parser.add_argument("--fusion_optimizer_mode", type=str, default="unified", choices=["unified", "split"])
    parser.add_argument("--text_hidden", type=int, default=128)
    parser.add_argument("--residual_rank", type=int, default=8)
    parser.add_argument("--direct_gate_bias", type=float, default=-0.8)
    parser.add_argument("--direct_gate_cap", type=float, default=0.4)
    parser.add_argument("--direct_text_scale", type=float, default=0.9)
    parser.add_argument("--residual_radius_scale", type=float, default=1.1)
    parser.add_argument("--residual_delta_scale", type=float, default=1.05)


def add_geo_fusion_args(parser):
    """Geo 融合流额外参数。"""
    add_common_fusion_args(parser)
    parser.add_argument("--llm_dim", type=int, default=768)
    parser.add_argument("--delta_scale", type=float, default=1.0)
    parser.add_argument("--use_vol_prior", type=int, default=1)
    parser.add_argument("--force_gain", type=float, default=-1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--direct_w_mode", type=str, default="learned", choices=["learned", "fixed"])
    parser.add_argument("--direct_w_fixed", type=float, default=0.5)


def add_geo_args(parser):
    """Geo 数值流使用的元数据开关。"""
    parser.add_argument("--use_element", action="store_true")
    parser.add_argument("--use_group", action="store_true")


def add_output_args(parser):
    """结果输出目录与可视化开关。"""
    parser.add_argument("--visualize", type=str2bool, default=False)
    parser.add_argument("--output_dir", type=str, default="./model_outputs/")
    parser.add_argument("--checkpoint_dir", type=str, default="./model_checkpoints/")


def build_parser(task_name=None):
    # 先挂载通用参数，再按任务补充专属配置，避免不同实验互相污染参数空间。
    parser = argparse.ArgumentParser(description="DualSG FIT/Geo runner")
    add_runtime_args(parser)
    add_data_args(parser)
    add_sequence_args(parser)
    add_model_args(parser)
    add_training_args(parser)
    add_device_args(parser)

    if task_name == "fit_fusion":
        add_fit_fusion_args(parser)
    elif task_name == "geo_fusion":
        add_geo_fusion_args(parser)

    if task_name in {"geo_num", "geo_num_with_meta", "geo_fusion"}:
        add_geo_args(parser)

    add_output_args(parser)
    return parser


def infer_task_name(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    if any(flag in argv for flag in ("-h", "--help")) and "--task_name" not in argv:
        return None

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task_name", type=str, required=True)
    parsed, _ = parser.parse_known_args(argv)
    return parsed.task_name


def configure_device(args):
    log("\n" + "=" * 50)
    log("Device setup")
    log("=" * 50)

    cuda_available = torch.cuda.is_available()
    log(f"CUDA available: {cuda_available}")
    log(f"Requested GPU: {args.use_gpu}")

    if args.use_gpu and args.gpu_type == "cuda" and cuda_available:
        selected_gpu = args.gpu
        try:
            import subprocess

            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=False,
            )
            gpu_info = result.stdout.strip().split("\n")
            gpu_usage = []
            for line in gpu_info:
                if line:
                    index, used, total = line.split(",")
                    gpu_usage.append((int(index), int(used), int(total)))

            if gpu_usage:
                gpu_usage.sort(key=lambda item: item[1] / max(item[2], 1))
                selected_gpu, memory_used, memory_total = gpu_usage[0]
                args.gpu = selected_gpu
                log(
                    f"Auto-selected GPU: cuda:{selected_gpu} "
                    f"(free memory: {memory_total - memory_used}/{memory_total} MB)"
                )
        except Exception as exc:
            log(f"Auto GPU selection failed: {exc}; fallback to cuda:{args.gpu}")

        args.device = torch.device(f"cuda:{args.gpu}")
        try:
            _ = torch.tensor([1.0]).to(args.device)
            log("GPU smoke check passed")
        except Exception as exc:
            log(f"GPU initialization failed: {exc}")
            args.device = torch.device("cpu")
            args.use_gpu = False
            log("Fallback to CPU")
    elif args.use_gpu and args.gpu_type == "mps":
        args.device = torch.device("mps")
        log("Using MPS device")
    else:
        if args.use_gpu and not cuda_available:
            log("CUDA not available; fallback to CPU")
        else:
            log("Using CPU")
        args.device = torch.device("cpu")
        args.use_gpu = False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        args.device_ids = [int(device_id) for device_id in args.devices.split(",")]
        args.gpu = args.device_ids[0]

    log("=" * 50 + "\n")


def get_exp_class(task_name):
    exp_map = {
        "fit_num": Exp_Fit_Num,
        "fit_num_with_meta": Exp_Fit_Num,
        "fit_fusion": Exp_Fit_Fusion,
        "geo_num": Exp_Geo_Num,
        "geo_num_with_meta": Exp_Geo_Num,
        "geo_fusion": Exp_Geo_Fusion,
    }
    if task_name not in exp_map:
        raise ValueError(f"Unsupported task_name: {task_name}")
    return exp_map[task_name]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    task_name = infer_task_name(argv)
    parser = build_parser(task_name=task_name)
    args = parser.parse_args(argv)

    set_seed(args.seed)
    configure_device(args)

    log("Args in experiment:")
    print_args(args)

    Exp = get_exp_class(args.task_name)

    if args.is_training:
        for _ in range(args.itr):
            exp = Exp(args)
            setting = build_setting(args, with_timestamp=True)

            log(f">>>>>>> start training : {setting} >>>>>>>>>")
            start_time = time.time()
            exp.train(setting)
            log(f"Training time: {time.time() - start_time:.2f}s")

            log(f">>>>>>> testing : {setting} <<<<<<<<<")
            exp.test(setting)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        exp = Exp(args)
        setting = build_setting(args, with_timestamp=False)
        exp.test(setting, test=1)


if __name__ == "__main__":
    main()
