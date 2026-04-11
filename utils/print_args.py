"""
参数打印工具模块

本文件实现了一个函数，用于打印模型训练和测试的参数配置，方便查看和调试。
"""

def print_args(args):
    """
    打印参数配置

    按照不同类别打印模型训练和测试的参数配置，包括任务基本参数、数据参数、时间序列参数、模型参数、训练参数、GPU配置和其他参数。

    Args:
        args: 参数对象，包含所有配置参数

    Returns:
        None
    """
    print("\033[1m" + "任务基本参数" + "\033[0m")
    print(f'  {"Task Name:":<20}{args.task_name:<20}{"Is Training:":<20}{args.is_training:<20}')
    print(f'  {"Model ID:":<20}{args.model_id:<20}{"Model:":<20}{args.model:<20}')
    print()

    print("\033[1m" + "数据参数" + "\033[0m")
    print(f'  {"Data:":<20}{args.data:<20}{"Root Path:":<20}{args.root_path:<20}')
    print(f'  {"Data Path:":<20}{args.data_path:<20}{"Features:":<20}{args.features:<20}')
    print(f'  {"Target:":<20}{args.target:<20}{"Freq:":<20}{args.freq:<20}')
    print(f'  {"Scale:":<20}{args.scale:<20}{"Inverse:":<20}{args.inverse:<20}')
    print()

    print("\033[1m" + "时间序列参数" + "\033[0m")
    print(f'  {"Seq Len:":<20}{args.seq_len:<20}{"Label Len:":<20}{args.label_len:<20}')
    print(f'  {"Pred Len:":<20}{args.pred_len:<20}{"Seasonal Patterns:":<20}{args.seasonal_patterns:<20}')
    print()

    print("\033[1m" + "模型参数" + "\033[0m")
    print(f'  {"Top k:":<20}{args.top_k:<20}{"Num Kernels:":<20}{args.num_kernels:<20}')
    print(f'  {"Enc In:":<20}{args.enc_in:<20}{"Dec In:":<20}{args.dec_in:<20}')
    print(f'  {"C Out:":<20}{args.c_out:<20}{"d model:":<20}{args.d_model:<20}')
    print(f'  {"n heads:":<20}{args.n_heads:<20}{"e layers:":<20}{args.e_layers:<20}')
    print(f'  {"d layers:":<20}{args.d_layers:<20}{"d FF:":<20}{args.d_ff:<20}')
    print(f'  {"Moving Avg:":<20}{args.moving_avg:<20}{"Factor:":<20}{args.factor:<20}')
    print(f'  {"Distil:":<20}{args.distil:<20}{"Dropout:":<20}{args.dropout:<20}')
    print(f'  {"Embed:":<20}{args.embed:<20}{"Activation:":<20}{args.activation:<20}')
    print()

    print("\033[1m" + "训练参数" + "\033[0m")
    print(f'  {"Num Workers:":<20}{args.num_workers:<20}{"Itr:":<20}{args.itr:<20}')
    print(f'  {"Train Epochs:":<20}{args.train_epochs:<20}{"Batch Size:":<20}{args.batch_size:<20}')
    print(f'  {"Patience:":<20}{args.patience:<20}{"Learning Rate:":<20}{args.learning_rate:<20}')
    print(f'  {"Des:":<20}{args.des:<20}{"Loss:":<20}{args.loss:<20}')
    print(f'  {"Lradj:":<20}{args.lradj:<20}{"Use Amp:":<20}{args.use_amp:<20}')
    print()

    print("\033[1m" + "GPU配置" + "\033[0m")
    print(f'  {"Use GPU:":<20}{args.use_gpu:<20}{"GPU:":<20}{args.gpu:<20}')
    print(f'  {"Use Multi GPU:":<20}{args.use_multi_gpu:<20}{"Devices:":<20}{args.devices:<20}')
    print()

    # 融合模型特有参数
    if args.task_name in ['fit_fusion', 'geo_fusion']:
        print("\033[1m" + "融合模型参数" + "\033[0m")
        print(f'  {"Num Model Path:":<20}{args.num_model_path:<20}{"Text Model Path:":<20}{args.text_model_path:<20}')
        print(f'  {"Caption Emb Path:":<20}{args.caption_emb_path:<20}{"Text Mode:":<20}{args.text_mode:<20}')
        print(f'  {"Freeze Numerical:":<20}{args.freeze_numerical:<20}{"Disable Text:":<20}{args.disable_text:<20}')
        print(f'  {"Num Feat Dim:":<20}{args.num_feat_dim:<20}{"Fusion Dropout:":<20}{args.fusion_dropout:<20}')
        print(f'  {"Fusion Hidden:":<20}{args.fusion_hidden:<20}')
        print()

    # GeoStyle特有参数
    if args.task_name in ['geo_num', 'geo_num_with_meta', 'geo_fusion']:
        print("\033[1m" + "GeoStyle特有参数" + "\033[0m")
        print(f'  {"Use Element:":<20}{args.use_element:<20}{"Use Group:":<20}{args.use_group:<20}')
        print()

    print("\033[1m" + "其他参数" + "\033[0m")
    print(f'  {"Visualize:":<20}{args.visualize:<20}{"Output Dir:":<20}{args.output_dir:<20}')
    print(f'  {"Checkpoint Dir:":<20}{args.checkpoint_dir:<20}{"Seed:":<20}{args.seed:<20}')
    print()