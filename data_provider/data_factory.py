"""
数据提供者工厂

该模块负责根据配置创建对应的数据集和数据加载器。

支持的数据集：
- FIT_Meta: FIT数据集的纯数值流版本，包含city/gender/age/element元数据
- FIT_Fusion: FIT数据集的融合版本，结合数值流和文本流
- Geo_Meta: GeoStyle数据集的纯数值流版本，可选element元数据
- Geo_Fusion: GeoStyle数据集的融合版本，结合数值流和文本流

依赖：
- torch.utils.data.DataLoader
- data_provider下的各种数据加载器
"""
from torch.utils.data import DataLoader
from utils.logger import log

# ===============================================================
# FIT datasets
# ===============================================================
from data_provider.data_loader_fit_num_meta import Dataset_DualSG_Fit_Num_Meta
from data_provider.data_loader_fit_num_meta_semantic import Dataset_DualSG_Fit_Num_Meta_Semantic
from data_provider.data_loader_fit_fusion import Dataset_DualSG_Fit_Fusion

# ===============================================================
# GeoStyle datasets
# ===============================================================
from data_provider.data_loader_geo_num_meta import Dataset_DualSG_Geo_Num_Meta
from data_provider.data_loader_geo_fusion import Dataset_DualSG_Geo_Fusion


# ===============================================================
# Dataset Registry
# ===============================================================
data_dict = {
    # FIT
    'FIT_Meta': Dataset_DualSG_Fit_Num_Meta,
    'FIT_Meta_Semantic': Dataset_DualSG_Fit_Num_Meta_Semantic,
    'FIT_Fusion': Dataset_DualSG_Fit_Fusion,

    # GeoStyle
    'Geo_Meta': Dataset_DualSG_Geo_Num_Meta,
    'Geo_Fusion': Dataset_DualSG_Geo_Fusion,
}


# ===============================================================
# data_provider 主入口
# ===============================================================
def data_provider(args, flag):
    """
    数据提供者工厂函数

    参数:
        args: 配置对象，包含以下关键属性：
            - data: 数据集类型，可选值：
                - FIT_Meta: FIT数据集的纯数值流版本
                - FIT_Fusion: FIT数据集的融合版本
                - Geo_Meta: GeoStyle数据集的纯数值流版本
                - Geo_Fusion: GeoStyle数据集的融合版本
            - root_path: 数据根目录
            - data_path: 数据文件路径
            - seq_len: 输入序列长度
            - label_len: 标签序列长度
            - pred_len: 预测序列长度
            - features: 特征类型
            - target: 目标变量
            - scale: 是否进行数据缩放
            - embed: 嵌入类型
            - batch_size: 批次大小
            - num_workers: 数据加载线程数
            - caption_emb_path: 文本嵌入路径（融合模式需要）
        flag: 数据类型，可选值：train / val / test

    返回:
        tuple: (数据集对象, 数据加载器对象)

    功能:
        根据配置创建对应的数据集和数据加载器
    """


    assert args.data in data_dict, f"Unsupported dataset: {args.data}"

    timeenc = 0 if args.embed != 'timeF' else 1
    shuffle_flag = False if flag.lower() == 'test' else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq
    split_limit = getattr(args, f"max_{flag.lower()}_samples", -1)

    # ===========================================================
    # FIT Fusion（数值 + 文本语义）
    # ===========================================================
    if args.data == 'FIT_Fusion':
        data_set = Dataset_DualSG_Fit_Fusion(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            scale=args.scale,
            timeenc=timeenc,
            freq=freq,
            text_field=getattr(args, "text_field", "annotations"),
            train_ratio=getattr(args, "train_ratio", 0.7),
            val_ratio=getattr(args, "val_ratio", 0.1),
            test_ratio=getattr(args, "test_ratio", 0.2),
            max_samples=split_limit,
            fit_scaler_mode=getattr(args, "fit_scaler_mode", "train_only"),
        )

    # ===========================================================
    # FIT Meta（纯数值流，含 city / gender / age / element）
    # ===========================================================
    elif args.data == 'FIT_Meta':
        data_set = Dataset_DualSG_Fit_Num_Meta(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            scale=args.scale,
            timeenc=timeenc,
            freq=freq,
            train_ratio=getattr(args, "train_ratio", 0.7),
            val_ratio=getattr(args, "val_ratio", 0.1),
            test_ratio=getattr(args, "test_ratio", 0.2),
            max_samples=split_limit,
            fit_scaler_mode=getattr(args, "fit_scaler_mode", "train_only"),
        )

    # ===========================================================
    # FIT Meta + Semantic Supervision
    # ===========================================================
    elif args.data == 'FIT_Meta_Semantic':
        data_set = Dataset_DualSG_Fit_Num_Meta_Semantic(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            scale=args.scale,
            timeenc=timeenc,
            freq=freq,
            train_ratio=getattr(args, "train_ratio", 0.7),
            val_ratio=getattr(args, "val_ratio", 0.1),
            test_ratio=getattr(args, "test_ratio", 0.2),
            max_samples=split_limit,
            fit_scaler_mode=getattr(args, "fit_scaler_mode", "train_only"),
            text_field=getattr(args, "semantic_text_field", "annotations"),
        )

    # ===========================================================
    # GeoStyle Fusion（数值 + 文本语义）
    # ===========================================================
    elif args.data == 'Geo_Fusion':
        if not hasattr(args, "caption_emb_path"):
            raise ValueError("args.caption_emb_path must be set for Geo_Fusion")

        data_set = Dataset_DualSG_Geo_Fusion(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            scale=args.scale,
            timeenc=timeenc,
            freq=freq,
            caption_emb_path=args.caption_emb_path,
            use_element=getattr(args, "use_element", False),
            train_ratio=getattr(args, "train_ratio", 0.7),
            val_ratio=getattr(args, "val_ratio", 0.1),
            test_ratio=getattr(args, "test_ratio", 0.2),
        )

    # ===========================================================
    # GeoStyle Meta（纯数值流，默认 No Meta，可选 element）
    # ===========================================================
    elif args.data == 'Geo_Meta':
        data_set = Dataset_DualSG_Geo_Num_Meta(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            scale=args.scale,
            timeenc=timeenc,
            freq=freq,
            use_element=getattr(args, "use_element", False),
            train_ratio=getattr(args, "train_ratio", 0.7),
            val_ratio=getattr(args, "val_ratio", 0.1),
            test_ratio=getattr(args, "test_ratio", 0.2),
        )

    else:
        raise NotImplementedError(f"Unhandled dataset type: {args.data}")

    log(f"[{flag.upper()}] {len(data_set)} samples ({args.data})")

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )

    return data_set, data_loader
