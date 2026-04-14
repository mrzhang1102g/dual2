"""
参数打印工具。

目标：
- 只打印当前实验排查真正需要的参数
- FIT / Geo 共用一份逻辑
- FIT fusion 仅展示 0414 新主线实际使用的参数
"""


def print_args(args):
    print("\033[1m" + "任务基本参数" + "\033[0m")
    print(f'  {"Task Name:":<22}{args.task_name:<24}{"Is Training:":<22}{args.is_training:<24}')
    print(f'  {"Model ID:":<22}{args.model_id:<24}{"Model:":<22}{args.model:<24}')
    print()

    print("\033[1m" + "数据参数" + "\033[0m")
    print(f'  {"Data:":<22}{args.data:<24}{"Root Path:":<22}{args.root_path:<24}')
    print(f'  {"Data Path:":<22}{args.data_path:<24}{"Features:":<22}{args.features:<24}')
    print(f'  {"Target:":<22}{args.target:<24}{"Freq:":<22}{args.freq:<24}')
    print(f'  {"Scale:":<22}{args.scale:<24}{"Inverse:":<22}{args.inverse:<24}')
    print(f'  {"Train Ratio:":<22}{args.train_ratio:<24}{"Val Ratio:":<22}{args.val_ratio:<24}')
    print(f'  {"Fit Scaler Mode:":<22}{getattr(args, "fit_scaler_mode", "n/a"):<24}{"Test Ratio:":<22}{args.test_ratio:<24}')
    print(f'  {"Max Train:":<22}{args.max_train_samples:<24}{"Max Val:":<22}{args.max_val_samples:<24}')
    print(f'  {"Max Test:":<22}{args.max_test_samples:<24}')
    print()

    print("\033[1m" + "序列参数" + "\033[0m")
    print(f'  {"Seq Len:":<22}{args.seq_len:<24}{"Label Len:":<22}{args.label_len:<24}')
    print(f'  {"Pred Len:":<22}{args.pred_len:<24}{"Seasonal Patterns:":<22}{args.seasonal_patterns:<24}')
    print()

    print("\033[1m" + "模型参数" + "\033[0m")
    print(f'  {"Enc In:":<22}{args.enc_in:<24}{"Dec In:":<22}{args.dec_in:<24}')
    print(f'  {"C Out:":<22}{args.c_out:<24}{"d_model:":<22}{args.d_model:<24}')
    print(f'  {"n_heads:":<22}{args.n_heads:<24}{"e_layers:":<22}{args.e_layers:<24}')
    print(f'  {"d_layers:":<22}{args.d_layers:<24}{"d_ff:":<22}{args.d_ff:<24}')
    print(f'  {"Factor:":<22}{args.factor:<24}{"Dropout:":<22}{args.dropout:<24}')
    print(f'  {"Patch Len:":<22}{args.patch_len:<24}{"Stride:":<22}{args.stride:<24}')
    print()

    print("\033[1m" + "训练参数" + "\033[0m")
    print(f'  {"Train Epochs:":<22}{args.train_epochs:<24}{"Batch Size:":<22}{args.batch_size:<24}')
    print(f'  {"Patience:":<22}{args.patience:<24}{"Learning Rate:":<22}{args.learning_rate:<24}')
    print(f'  {"Weight Decay:":<22}{args.weight_decay:<24}{"Loss:":<22}{args.loss:<24}')
    print(f'  {"Lradj:":<22}{args.lradj:<24}{"Adjust LR:":<22}{args.adjust:<24}')
    print(f'  {"Use Amp:":<22}{args.use_amp:<24}{"Num Workers:":<22}{args.num_workers:<24}')
    print(f'  {"Use DTW:":<22}{args.use_dtw:<24}')
    print()

    print("\033[1m" + "设备参数" + "\033[0m")
    print(f'  {"Use GPU:":<22}{args.use_gpu:<24}{"GPU Type:":<22}{args.gpu_type:<24}')
    print(f'  {"GPU:":<22}{args.gpu:<24}{"Use Multi GPU:":<22}{args.use_multi_gpu:<24}')
    print(f'  {"Devices:":<22}{args.devices:<24}')
    print()

    if args.task_name == "fit_fusion":
        print("\033[1m" + "FIT Fusion 参数" + "\033[0m")
        print(f'  {"Text Mode:":<22}{args.text_mode:<24}{"Caption Emb Path:":<22}{str(args.caption_emb_path):<24}')
        print(f'  {"Num Model Path:":<22}{args.num_model_path:<24}{"Freeze Numerical:":<22}{args.freeze_numerical:<24}')
        print(f'  {"Disable Text:":<22}{args.disable_text:<24}{"Opt Mode:":<22}{args.fusion_optimizer_mode:<24}')
        print(f'  {"LR Num:":<22}{args.lr_num:<24}{"LR Text:":<22}{args.lr_text:<24}')
        print(f'  {"WD Text:":<22}{args.weight_decay_text:<24}{"Text Hidden:":<22}{args.text_hidden:<24}')
        print(f'  {"Residual Rank:":<22}{args.residual_rank:<24}{"Num Feat Dim:":<22}{args.num_feat_dim:<24}')
        print(f'  {"Fusion Hidden:":<22}{args.fusion_hidden:<24}{"Fusion Dropout:":<22}{args.fusion_dropout:<24}')
        print()

    if args.task_name in ["geo_num", "geo_num_with_meta", "geo_fusion"]:
        print("\033[1m" + "Geo 参数" + "\033[0m")
        print(f'  {"Use Element:":<22}{args.use_element:<24}{"Use Group:":<22}{args.use_group:<24}')
        print(f'  {"Patch Adaptive:":<22}{args.patch_adaptive:<24}')
        print()

    print("\033[1m" + "输出参数" + "\033[0m")
    print(f'  {"Visualize:":<22}{args.visualize:<24}{"Output Dir:":<22}{args.output_dir:<24}')
    print(f'  {"Checkpoint Dir:":<22}{args.checkpoint_dir:<24}{"Seed:":<22}{args.seed:<24}')
    print()
