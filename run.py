# run.py
import argparse
import os
import time
import random
import numpy as np
import torch
from datetime import datetime

from exp.exp_fit_num import Exp_Fit_Num
from exp.exp_fit_fusion import Exp_Fit_Fusion
from exp.exp_geo_num import Exp_Geo_Num
from exp.exp_geo_fusion import Exp_Geo_Fusion

from utils.print_args import print_args
from utils.logger import log


# 关掉huggingface/tokenizers fork并行警告
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def build_setting(args, with_timestamp=True):
    """
    极简实验名：model_id + 时间戳
    """
    if with_timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{args.model_id}_{ts}"
    return f"{args.model_id}"


def _set_seed(seed: int):
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 更稳定复现（不影响你的业务逻辑，只影响底层随机/算法选择）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    log(f"[Seed] fixed seed={seed}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TimesNet')

    # =========================================================
    # 2. 基础运行配置
    # =========================================================
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, required=True, default=1)
    parser.add_argument('--model_id', type=str, required=True, default='test')
    parser.add_argument('--model', type=str, required=True, default='Autoformer')

    # =========================================================
    # 3. 数据与任务配置
    # =========================================================
    parser.add_argument('--data', type=str, required=True, default='ETTm1')
    parser.add_argument('--root_path', type=str, default='./data/ETT/')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--features', type=str, default='S')
    parser.add_argument('--target', type=str, default='trend')
    parser.add_argument('--freq', type=str, default='w')
    parser.add_argument('--scale', type=bool, default=True)
    parser.add_argument('--checkpoints', type=str, default='./model_checkpoints/')
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.2)

    # =========================================================
    # 4. 时间序列任务参数
    # =========================================================
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--seq_len_imputation', type=int, default=96)
    parser.add_argument('--seq_len_fine', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=0)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly')
    parser.add_argument('--inverse', action='store_true', default=True)

    # =========================================================
    # 5. 特殊任务参数（补全/异常/微调）
    # =========================================================
    parser.add_argument('--mask_rate', type=float, default=0.25)
    parser.add_argument('--fine_rate', type=float, default=0.75)
    parser.add_argument('--anomaly_ratio', type=float, default=0.25)

    # =========================================================
    # 6. 模型结构参数（通用）
    # =========================================================
    parser.add_argument('--expand', type=int, default=2)
    parser.add_argument('--d_conv', type=int, default=4)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--num_kernels', type=int, default=6)
    parser.add_argument('--enc_in', type=int, default=1)
    parser.add_argument('--dec_in', type=int, default=1)
    parser.add_argument('--c_out', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--d_ff', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--distil', action='store_false', default=True)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--channel_independence', type=int, default=1)
    parser.add_argument('--decomp_method', type=str, default='moving_avg')
    parser.add_argument('--use_norm', type=int, default=1)
    parser.add_argument('--down_sampling_layers', type=int, default=0)
    parser.add_argument('--down_sampling_window', type=int, default=1)
    parser.add_argument('--down_sampling_method', type=str, default=None)
    parser.add_argument('--seg_len', type=int, default=48)

    # =========================================================
    # 7. 训练与优化参数
    # =========================================================
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=800)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--weight_decay_text', type=float, default=0.01)
    parser.add_argument('--des', type=str, default='test')
    parser.add_argument('--loss', type=str, default='MAE')
    parser.add_argument('--lradj', type=str, default='type3')
    parser.add_argument('--use_amp', action='store_true', default=False)

    # =========================================================
    # 8. GPU与设备配置
    # =========================================================
    parser.add_argument('--use_gpu', type=lambda x: x.lower() in ['true', '1', 'yes'], default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gpu_type', type=str, default='cuda')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0,1')

    # =========================================================
    # 9. 投影器与度量
    # =========================================================
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128])
    parser.add_argument('--p_hidden_layers', type=int, default=2)
    parser.add_argument('--use_dtw', type=bool, default=False)

    # =========================================================
    # 10. 数据增强
    # =========================================================
    parser.add_argument('--augmentation_ratio', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--jitter', action='store_true')
    parser.add_argument('--scaling', action='store_true')
    parser.add_argument('--permutation', action='store_true')
    parser.add_argument('--randompermutation', action='store_true')
    parser.add_argument('--magwarp', action='store_true')
    parser.add_argument('--timewarp', action='store_true')
    parser.add_argument('--windowslice', action='store_true')
    parser.add_argument('--windowwarp', action='store_true')
    parser.add_argument('--rotation', action='store_true')
    parser.add_argument('--spawner', action='store_true')
    parser.add_argument('--dtwwarp', action='store_true')
    parser.add_argument('--shapedtwwarp', action='store_true')
    parser.add_argument('--wdba', action='store_true')
    parser.add_argument('--discdtw', action='store_true')
    parser.add_argument('--discsdtw', action='store_true')

    # =========================================================
    # 11. LLM/Patch/Prompt相关（原工程保留）
    # =========================================================
    parser.add_argument('--fc_dropout', type=float, default=0.1)
    parser.add_argument('--llm_layers', type=int, default=6)
    parser.add_argument('--pool_type', type=str, default='avg')
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--max_length', type=int, default=15)
    parser.add_argument('--llm_model', type=str, default='GPT2')
    parser.add_argument('--use_fullmodel', type=int, default=0)
    parser.add_argument('--text_cd', type=int, default=0)
    parser.add_argument('--patch_adaptive', type=int, default=1)
    parser.add_argument('--adjust', type=int, default=1)
    parser.add_argument('--prompt', type=int, default=1)

    # =========================================================
    # 12. GeoStyle/Meta特征
    # =========================================================
    parser.add_argument('--use_element', action='store_true')
    parser.add_argument('--use_group', action='store_true')

    # =========================================================
    # 13. 融合模型参数
    # =========================================================
    parser.add_argument('--num_model_path', type=str, default='')
    parser.add_argument('--text_model_path', type=str, default='')
    parser.add_argument('--caption_emb_path', type=str, default=None)

    parser.add_argument('--freeze_numerical', action='store_true')
    parser.add_argument('--disable_text', action='store_true')
    parser.add_argument('--text_mode', type=str, default='direct')
    parser.add_argument('--delta_scale', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta_delta', type=float, default=0.0)
    parser.add_argument('--use_vol_prior', type=int, default=1)
    parser.add_argument('--gain_init', type=float, default=None)

    # Fusion head超参（你要的3个）
    parser.add_argument('--num_feat_dim', type=int, default=48)       # pred_len=24时默认=48
    parser.add_argument('--fusion_dropout', type=float, default=0.1)  # 默认=0.1
    parser.add_argument('--fusion_hidden', type=int, default=96)      # pred_len=24时默认=96

    # =========================================================
    # 13.x 文本输入模式(emb/raw)
    # =========================================================
    parser.add_argument('--raw_text_source', type=str, default='llm_caption', choices=['annotations','llm_caption','llm_json'])
    parser.add_argument('--text_input_mode', type=str, default='emb', choices=['emb', 'raw'])
    parser.add_argument('--text_encoder_name', type=str, default='gpt2')  # raw模式用
    parser.add_argument('--text_max_length', type=int, default=128)       # raw模式token长度
    parser.add_argument('--raw_fallback_to_annotations', type=int, default=0)  # 0/1
    parser.add_argument('--text_pooling', type=str, default='mean', choices=['mean', 'attn'])
    parser.add_argument('--llm_dim', type=int, default=768)

    # B档：解冻最后N个block
    parser.add_argument('--text_unfreeze_last_n', type=int, default=0)

    # 兼容老开关：如果你Model里还保留了freeze_text_encoder逻辑
    parser.add_argument('--freeze_text_encoder', type=int, default=0)  # 建议默认0，避免把B档又冻回去

    # 学习率相关
    parser.add_argument('--lr_num', type=float, default=1e-4)
    parser.add_argument('--lr_text', type=float, default=1e-3)

    # =========================================================
    # 14. 消融实验参数
    # =========================================================
    parser.add_argument('--force_gain', type=float, default=-1.0)
    parser.add_argument('--direct_w_mode', type=str, default='learned', choices=['learned', 'fixed'])
    parser.add_argument('--direct_w_fixed', type=float, default=0.5)
    parser.add_argument('--max_test_samples', type=int, default=-1)

    # =========================================================
    # 15. Debug日志参数
    # =========================================================
    parser.add_argument('--debug_every', type=int, default=0)
    parser.add_argument('--debug_residual_stats', type=int, default=0)
    parser.add_argument('--debug_direct_stats', type=int, default=0)
    parser.add_argument('--debug_only_train', type=int, default=1)

    # =========================================================
    # 16. 可视化与输出配置
    # =========================================================
    parser.add_argument('--visualize', type=lambda x: x.lower() in ['true', '1', 'yes'], default=False, help='是否生成可视化图表')
    parser.add_argument('--output_dir', type=str, default='./model_outputs/', help='模型输出目录')
    parser.add_argument('--checkpoint_dir', type=str, default='./model_checkpoints/', help='检查点保存目录')

    args = parser.parse_args()

    # =========================================================
    # 1. 固定随机种子
    # =========================================================
    _set_seed(args.seed)

    # =========================================================
    # 15. 设备初始化
    # =========================================================
    log('\n' + '='*50)
    log('设备初始化')
    log('='*50)
    
    selected_gpu = args.gpu
    memory_free = 0
    memory_total = 0
    
    # 打印CUDA信息
    log(f'CUDA可用: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        # 自动选择空闲GPU
        import subprocess
        try:
            # 运行nvidia-smi命令获取GPU信息
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            gpu_info = result.stdout.strip().split('\n')
            
            # 解析GPU信息
            gpu_usage = []
            for line in gpu_info:
                if line:
                    index, used, total = line.split(',')
                    gpu_usage.append((int(index), int(used), int(total)))
            
            # 按显存使用率排序，选择最空闲的GPU
            gpu_usage.sort(key=lambda x: x[1]/x[2])
            selected_gpu = gpu_usage[0][0]
            memory_used = gpu_usage[0][1]
            memory_total = gpu_usage[0][2]
            memory_free = memory_total - memory_used
            args.gpu = selected_gpu
            log(f'自动选择GPU: cuda:{selected_gpu} (显存剩余: {memory_free}/{memory_total} MB)')
        except Exception as e:
            log(f'自动选择GPU失败: {e}，使用默认GPU: {args.gpu}')
    
    # 使用选择的GPU
    if torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.gpu}')
        args.use_gpu = True
        # 测试GPU是否可用
        try:
            test_tensor = torch.tensor([1.0]).to(args.device)
            log('GPU测试成功')
        except Exception as e:
            log(f'GPU使用失败: {e}')
            args.device = torch.device('cpu')
            args.use_gpu = False
            log('回退到使用CPU')
    else:
        args.device = torch.device('cpu')
        args.use_gpu = False
        log('使用CPU')
    
    log('='*50 + '\n')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(i) for i in device_ids]
        args.gpu = args.device_ids[0]

    log('Args in experiment:')
    print_args(args)

    # =========================================================
    # 16. 选择实验类
    # =========================================================
    if args.task_name == 'fit_num':
        Exp = Exp_Fit_Num
    elif args.task_name == 'fit_num_with_meta':
        Exp = Exp_Fit_Num
    elif args.task_name == 'fit_fusion':
        Exp = Exp_Fit_Fusion
    elif args.task_name == 'geo_num':
        Exp = Exp_Geo_Num
    elif args.task_name == 'geo_num_with_meta':
        Exp = Exp_Geo_Num
    elif args.task_name == 'geo_fusion':
        Exp = Exp_Geo_Fusion
    else:
        Exp = Exp_Fit_Num

    # =========================================================
    # 17. 运行训练/测试
    # =========================================================
    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)
            setting = build_setting(args, with_timestamp=True)

            log(f'>>>>>>> start training : {setting} >>>>>>>>>')
            start_time = time.time()
            exp.train(setting)
            log(f'⏱️ Training time: {time.time() - start_time:.2f}s')

            log(f'>>>>>>> testing : {setting} <<<<<<<<<')
            exp.test(setting)

            torch.cuda.empty_cache()
    else:
        exp = Exp(args)
        setting = build_setting(args, with_timestamp=False)
        exp.test(setting, test=1)
