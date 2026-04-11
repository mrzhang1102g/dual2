# 近期目标规划

## 已完成任务

### 2026-04-10
- ✅ 修改所有fit fusion脚本配置，改回原始设置（Model_Fit_Fusion, d_model=64, d_ff=256, n_heads=4, e_layers=2, d_layers=1, batch_size=800, train_epochs=100, patch_adaptive=1）
- ✅ 修改run.py参数默认值，将数值流部分的参数设置为默认值（enc_in=1, dec_in=1, c_out=1, d_model=64, d_ff=256, n_heads=4, train_epochs=100, batch_size=800）
- ✅ 优化GPU自动选择功能，移除所有sh脚本中的--gpu参数，实现自动选择显存剩余最多的GPU
- ✅ 简化GPU初始化日志输出，只显示选择的GPU编号和显存剩余量
- ✅ 统一所有脚本的task_name命名格式，使用更简洁、更直观的命名（fit_num, fit_num_with_meta, fit_fusion, geo_num, geo_num_with_meta, geo_fusion）
- ✅ 移除model_fit_fusion.py和model_geo_fusion.py中硬编码的task_name，使用传入的configs.task_name
- ✅ 统一日志输出到exp模块，创建了统一的log方法
- ✅ 修复model_fit_num.py中的硬编码52值，实现动态计算周期长度
- ✅ 分类整理run.py中的参数，按功能分组并添加注释
- ✅ 统一所有sh脚本的参数顺序和分组方式，添加清晰的注释
- ✅ 更新global_architecture.md文件，记录代码优化的工作
- ✅ 进一步修复model_fit_num.py中的硬编码52值，直接使用配置中的seq_len参数
- ✅ 改进参数表格格式，按类型分组显示参数，支持特有参数显示
- ✅ 在训练、验证、测试过程中添加进度条，提高用户体验
- ✅ 优化可视化功能，确保可通过--visualize参数开关
- ✅ 统一所有模块的日志输出，将print语句替换为self.log()方法
- ✅ 检查并修复其他可能存在的硬编码参数，确保代码风格一致
- ✅ 统一data_provider模块的日志输出，将所有print语句替换为log函数
- ✅ 提取重复代码到公共工具函数，减少代码冗余
- ✅ 创建data_provider/utils.py文件，提供公共工具函数
- ✅ 更新所有数据加载器文件，使用公共工具函数
- ✅ 将data_provider/utils.py移动到utils文件夹中并改名为data_provider_utils.py
- ✅ 更新所有引用data_provider/utils.py的地方，改为引用utils.data_provider_utils
- ✅ 修改exp/exp_basic.py文件，使用统一的日志工具
- ✅ 更新global_architecture.md文件，记录日志工具的统一使用情况和文件结构的调整
- ✅ 修改exp_fit_fusion.py和exp_geo_fusion.py，使用checkpoint_dir参数替代checkpoints
- ✅ 修改run.py中--checkpoints参数默认值为./model_checkpoints/
- ✅ 清空model_checkpoints和model_outputs文件夹
- ✅ 删除checkpoints文件夹
- ✅ 修复exp模块中硬编码的随机种子（42），改为使用self.args.seed
- ✅ 修复exp_fit_num.py、exp_fit_fusion.py、exp_geo_num.py、exp_geo_fusion.py中的随机种子
- ✅ 修改run.py文件，将所有print语句替换为log函数
- ✅ 导入utils.logger中的log函数到run.py
- ✅ 确保全局随机种子统一使用args.seed
- ✅ 在exp_basic.py中添加统一的print_trainable_parameters()方法
- ✅ 修改exp_fit_fusion.py，使用统一的print_trainable_parameters()方法
- ✅ 修改exp_geo_fusion.py，使用统一的print_trainable_parameters()方法
- ✅ 修改exp_fit_num.py，添加print_trainable_parameters()调用
- ✅ 修改exp_geo_num.py，使用统一的print_trainable_parameters()方法
- ✅ 确保所有实验都会在训练开始前打印可学习参数量

### 2026-04-09
- ✅ 创建 global_architecture.md 文件，记录项目架构和已完成工作
- ✅ 清理 model_outputs 和 model_checkpoints 文件夹，只保留6个成功实验的结果
- ✅ 修复可视化代码中的开关问题
- ✅ 完成6个数值流实验的训练和测试
- ✅ 生成 training_results_20260409_102400.md 结果汇总文件
- ✅ 修复 Model_Fit_Num 的时间戳维度问题
- ✅ 修复 GeoStyle 带元数据模型的输出一致性问题
- ✅ 清理项目临时文件，精简项目结构
- ✅ 更新 global_architecture.md 文件，记录最新完成的工作
- ✅ 更新 target_planning.md 文件，记录已完成任务
- ✅ 统一所有实验的输出结构
- ✅ 清理 model_checkpoints 目录，只保留6个对应的检查点
- ✅ 对比分析本次实验与原始实验的差异

### 2026-04-08
- ✅ 为bf文件夹中的所有6个sh脚本添加可视化参数
- ✅ 修复 result_long_term_forecast.txt 生成位置问题
- ✅ 执行6个数值流实验的训练

## 近期目标

### 1. 融合模型测试（优先级：高）

#### FIT融合模型
- [ ] 运行 fit_fusion_halfyear_direct.sh
- [ ] 运行 fit_fusion_halfyear_residual.sh
- [ ] 运行 fit_fusion_oneyear_direct.sh
- [ ] 运行 fit_fusion_oneyear_residual.sh

#### GeoStyle融合模型
- [ ] 运行 geostyle_fusion_direct.sh
- [ ] 运行 geostyle_fusion_residual.sh

### 2. 结果分析（优先级：中）
- [ ] 生成融合模型的结果汇总文件
- [ ] 比较融合模型与纯数值流模型的性能
- [ ] 分析不同融合方式（direct vs residual）的效果

### 3. 代码优化（优先级：低）
- [ ] 检查融合模型的文本输入处理
- [ ] 确保所有脚本参数配置一致
- [ ] 优化模型性能和训练速度

## 执行计划

1. **4月9日-4月10日**: 完成FIT融合模型测试
2. **4月11日-4月12日**: 完成GeoStyle融合模型测试
3. **4月13日**: 整理所有实验结果，生成最终报告

## 注意事项
- 数值流模型代码已调通，尽量不修改
- 融合模型可能需要调整文本输入处理
- 确保所有脚本使用相同的参数配置
- 记录所有实验结果到汇总文件