"""数据集工厂。

根据 `args.data` 统一构建 FIT / Geo 的 dataset 与 dataloader，
把训练脚本中的数据分支收敛到这一处。
"""

from torch.utils.data import DataLoader

from data_provider.data_loader_fit_fusion import Dataset_DualSG_Fit_Fusion
from data_provider.data_loader_fit_num_meta import Dataset_DualSG_Fit_Num_Meta
from data_provider.data_loader_geo_fusion import Dataset_DualSG_Geo_Fusion
from data_provider.data_loader_geo_num_meta import Dataset_DualSG_Geo_Num_Meta
from utils.logger import log


data_dict = {
    "FIT_Meta": Dataset_DualSG_Fit_Num_Meta,
    "FIT_Fusion": Dataset_DualSG_Fit_Fusion,
    "Geo_Meta": Dataset_DualSG_Geo_Num_Meta,
    "Geo_Fusion": Dataset_DualSG_Geo_Fusion,
}


def data_provider(args, flag):
    """根据配置返回 `(dataset, dataloader)`。

    `flag` 只接受 `train / val / test`，并统一在这里控制
    shuffle、split 上限和 caption embedding 的加载入口。
    """

    assert args.data in data_dict, f"Unsupported dataset: {args.data}"

    timeenc = 0 if args.embed != "timeF" else 1
    shuffle_flag = flag.lower() != "test"
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq
    split_limit = getattr(args, f"max_{flag.lower()}_samples", -1)

    # FIT 融合流：数值序列与离线文本向量一起读取。
    if args.data == "FIT_Fusion":
        if not hasattr(args, "caption_emb_path"):
            raise ValueError("args.caption_emb_path must be set for FIT_Fusion")

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
            caption_emb_path=args.caption_emb_path,
            train_ratio=getattr(args, "train_ratio", 0.7),
            val_ratio=getattr(args, "val_ratio", 0.1),
            test_ratio=getattr(args, "test_ratio", 0.2),
            max_samples=split_limit,
            fit_scaler_mode=getattr(args, "fit_scaler_mode", "train_only"),
        )

    # FIT 数值流：仍然返回完整元数据，是否使用由模型决定。
    elif args.data == "FIT_Meta":
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

    # Geo 融合流：与数值流对齐，只在末尾额外追加 caption embedding。
    elif args.data == "Geo_Fusion":
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
            max_samples=split_limit,
        )

    # Geo 数值流：可选 element/group 元数据。
    elif args.data == "Geo_Meta":
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
            max_samples=split_limit,
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
