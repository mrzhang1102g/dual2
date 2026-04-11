# DualSG 项目全局架构

**最后更新时间**: 2026-04-10 (最新)

## 项目概述

DualSG 是一个基于时间序列的预测模型项目，主要处理两种数据集：FIT（Fashion Industry Trends）和 GeoStyle。项目结合了数值流和文本流的融合模型，旨在提高时间序列预测的准确性。

## 项目结构

```
DualSG_refined/
├── bf/                    # 实验配置脚本（备份）
├── data_provider/         # 数据加载器模块
│   ├── data_factory.py           # 数据加载器工厂
│   ├── data_loader_fit_num.py    # FIT数值流数据加载器
│   ├── data_loader_fit_num_meta.py  # FIT数值流+元数据数据加载器
│   ├── data_loader_fit_fusion.py    # FIT融合模型数据加载器
│   ├── data_loader_geo_num.py    # GeoStyle数值流数据加载器
│   ├── data_loader_geo_num_meta.py  # GeoStyle数值流+元数据数据加载器
│   └── data_loader_geo_fusion.py    # GeoStyle融合模型数据加载器
├── dataset/               # 数据集
│   ├── FIT_DualSG/               # FIT数据集
│   └── Geo_DualSG/               # GeoStyle数据集
├── exp/                   # 实验类模块
│   ├── exp_basic.py       # 基础实验类
│   ├── exp_fit_num.py     # FIT纯数值流实验
│   ├── exp_fit_num_with_meta.py  # FIT数值流+元数据实验
│   ├── exp_fit_fusion.py  # FIT融合模型实验
│   ├── exp_geo_num.py     # GeoStyle纯数值流实验
│   ├── exp_geo_num_with_meta.py  # GeoStyle数值流+元数据实验
│   └── exp_geo_fusion.py  # GeoStyle融合模型实验
├── layers/                # 模型层定义
├── models/                # 模型定义
│   ├── model_fit_num.py
│   ├── model_fit_num_with_meta.py
│   ├── model_fit_fusion.py
│   ├── model_geo_num.py
│   ├── model_geo_num_with_meta.py
│   └── model_geo_fusion.py
├── utils/                 # 工具函数
│   ├── logger.py          # 统一日志工具
│   ├── data_provider_utils.py  # 数据处理工具函数
│   ├── tools.py           # 其他工具函数
│   ├── print_args.py      # 参数打印工具
│   └── masking.py         # 掩码工具
├── weights/               # 预训练权重
├── model_outputs/         # 模型输出目录
├── model_checkpoints/     # 模型检查点目录
├── run.py                 # 主运行脚本
├── *.sh                   # 实验脚本（根目录）
├── global_architecture.md # 全局架构文档
├── target_planning.md     # 近期目标规划
└── 审稿意见.md            # 审稿意见
```

## 核心模块说明

### 1. 数据加载器模块 (data_provider/)

负责加载和预处理FIT和GeoStyle数据集，提供统一的数据接口。

#### 主要组件
- **data_factory.py**: 数据加载器工厂，根据参数选择合适的数据加载器
- **数据加载器**: 针对不同数据集和模型类型的数据加载器
  - FIT系列：处理FIT数据集
  - GeoStyle系列：处理GeoStyle数据集
  - 纯数值流：只加载数值数据
  - 数值流+元数据：加载数值数据和元数据
  - 融合模型：加载数值数据、元数据和文本数据

#### 数据切分
- **全量模式**: 使用包含"all"的文件名，内部按70%/10%/20%切分训练/验证/测试集
- **分离模式**: 使用单独的train/val/test文件

### 2. 实验类模块 (exp/)

负责模型的训练、验证和测试流程。

#### 主要组件
- **exp_basic.py**: 基础实验类，提供通用功能
  - 设备管理（GPU/CPU自动选择）
  - 模型构建
  - 统一日志输出
- **具体实验类**: 针对不同数据集和模型类型的实验类
  - 实现数据获取、训练、验证、测试等具体逻辑
  - 处理特定模型的输出和可视化

#### 训练流程
1. 获取训练、验证、测试数据
2. 构建模型和优化器
3. 训练循环，每个epoch进行验证
4. 早停机制
5. 保存最佳模型检查点
6. 加载最佳模型进行测试
7. 生成可视化结果（可选）

### 3. 模型模块 (models/)

定义时间序列预测模型的架构。

#### 模型类型
- **纯数值流模型**: 只使用数值时间序列数据
  - Model_Fit_Num / Model_Geo_Num
- **数值流+元数据模型**: 使用数值数据和元数据
  - Model_Fit_Num_With_Meta / Model_Geo_Num_With_Meta
- **融合模型**: 结合数值流和文本流
  - Model_Fit_Fusion / Model_Geo_Fusion
  - 支持direct和residual两种融合方式

### 4. 工具函数模块 (utils/)

提供项目通用的工具函数。

#### 主要工具
- **logger.py**: 统一日志工具，提供log、log_warning、log_error等函数
- **data_provider_utils.py**: 数据处理工具函数
  - is_all_mode: 判断是否是全量文件模式
  - split_data: 数据切分
  - build_metadata_maps: 构建元数据映射表
  - generate_time_features: 生成时间特征
  - parse_fit_group: 解析FIT group字符串
- **tools.py**: 其他工具函数，包括可视化
- **print_args.py**: 参数打印工具，生成格式化的参数表格
- **masking.py**: 掩码工具

### 5. 主运行脚本 (run.py)

项目的入口点，负责解析命令行参数并启动实验。

#### 参数分类
1. **任务基本参数**: task_name, is_training, model_id, model, data
2. **数据参数**: root_path, data_path, features, target, freq, scale
3. **时间序列参数**: seq_len, label_len, pred_len
4. **模型参数**: d_model, d_ff, n_heads, e_layers, d_layers, dropout等
5. **训练参数**: train_epochs, batch_size, learning_rate, loss, patience等
6. **融合模型参数**: num_model_path, text_model_path, caption_emb_path等
7. **GeoStyle特有参数**: use_element, use_group
8. **可视化与输出配置**: visualize, output_dir, checkpoint_dir
9. **设备配置**: use_gpu, gpu, gpu_type等

#### 流程
1. 解析命令行参数
2. 设置随机种子
3. 初始化设备（GPU/CPU）
4. 打印参数表格
5. 创建实验对象
6. 执行训练或测试

## 数据流

### 训练流程
```
run.py 
  ↓
Exp_* (实验类)
  ↓
data_factory (数据加载器工厂)
  ↓
data_loader_* (数据加载器)
  ↓
dataset (数据集)
  ↓
model (模型)
  ↓
optimizer (优化器)
  ↓
loss (损失函数)
  ↓
model_checkpoints/ (检查点保存)
  ↓
model_outputs/ (结果输出)
```

## 配置规范

### 参数命名规范
- 数值流参数: 使用简洁的命名
- 融合模型参数: 以fusion_为前缀
- GeoStyle参数: 保持原有的命名
- 输出配置: output_dir, checkpoint_dir

### 日志输出规范
- 所有模块使用统一的日志工具
- 日志级别: info, warning, error
- 保持日志格式一致

### 目录规范
- 模型检查点: model_checkpoints/
- 模型输出: model_outputs/
- 可视化结果: model_outputs/{setting}/
- 结果文件: result.txt

## 模型配置

### FIT数据集
- **序列长度**: 半年预测 (48-12), 一年预测 (48-24)
- **频率**: 周 (w)
- **特点**: 包含元数据（城市、性别、年龄）

### GeoStyle数据集
- **序列长度**: 季度预测 (52-26)
- **频率**: 周 (w)
- **特点**: 包含元数据（element、group）

## 关键技术点

### 1. GPU自动选择
- 自动检测可用GPU
- 选择显存剩余最多的GPU
- 支持MPS（Apple Silicon）

### 2. 统一日志系统
- 所有模块使用统一的日志工具
- 支持不同日志级别
- 保持日志格式一致

### 3. 参数管理
- run.py集中管理所有参数
- 设置合理的默认值
- 支持命令行覆盖

### 4. 可视化
- 可开关控制
- 生成预测结果图表
- 保存到model_outputs目录

## 扩展指南

### 添加新的数据加载器
1. 在data_provider/目录下创建新的数据加载器文件
2. 继承Dataset类
3. 实现__getitem__和__len__方法
4. 在data_factory.py中注册

### 添加新的模型
1. 在models/目录下创建新的模型文件
2. 继承nn.Module类
3. 实现forward方法
4. 在exp_basic.py的model_dict中注册

### 添加新的实验类
1. 在exp/目录下创建新的实验类
2. 继承Exp_Basic类
3. 实现_get_data、train、vali、test等方法
4. 在run.py中注册

## 注意事项

1. **随机种子**: 全局固定随机种子为2026，确保可复现性
2. **日志输出**: 所有日志使用统一的日志工具
3. **目录规范**: 检查点和输出分别保存到model_checkpoints和model_outputs目录
4. **参数管理**: 新参数应在run.py中添加默认值
5. **代码风格**: 保持代码风格一致，使用中文注释
