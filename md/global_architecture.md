# FIT Architecture

最后更新：2026-04-13

## 1. 文档定位

这份文档只描述当前仓库里 FIT 相关的真实实现。

配套文档：

- `fit_refactor_worklog.md`：整理过程、smoke 结果、已知问题和实验结论
- `fit_fusion_redesign.md`：当前 FIT fusion 的设计思路
- `fit_next_round_plan.md`：2026-04-13 之后的下一轮计划
- `fit_server_runbook.md`：服务器执行顺序与结果记录模板

## 2. 当前入口

统一入口：

- `run.py`

任务与实验类：

- `fit_num` -> `exp/exp_fit_num.py`
- `fit_num_with_meta` -> `exp/exp_fit_num.py`
- `fit_fusion` -> `exp/exp_fit_fusion.py`

任务与模型：

- `Model_Fit_Num` -> `models/model_fit_num.py`
- `Model_Fit_Num_With_Meta` -> `models/model_fit_num_with_meta.py`
- `Model_Fit_Fusion` -> `models/model_fit_fusion.py`

任务与数据集：

- `FIT_Meta` -> `data_provider/data_loader_fit_num_meta.py`
- `FIT_Fusion` -> `data_provider/data_loader_fit_fusion.py`

## 3. FIT 关键文件

数据层：

- `data_provider/fit_dataset_utils.py`
- `data_provider/data_loader_fit_num_meta.py`
- `data_provider/data_loader_fit_fusion.py`
- `data_provider/data_factory.py`

实验层：

- `exp/exp_fit_num.py`
- `exp/exp_fit_fusion.py`

模型层：

- `models/model_fit_num.py`
- `models/model_fit_num_with_meta.py`
- `models/model_fit_fusion.py`

训练脚本：

- `fit_halfyear_num.sh`
- `fit_halfyear_num_with_meta.sh`
- `fit_oneyear_num.sh`
- `fit_oneyear_num_with_meta.sh`
- `fit_fusion_halfyear_direct.sh`
- `fit_fusion_halfyear_residual.sh`
- `fit_fusion_oneyear_direct.sh`
- `fit_fusion_oneyear_residual.sh`
- `fit_fusion_halfyear_direct_from_ckpt.sh`
- `fit_fusion_halfyear_residual_unfreeze.sh`
- `fit_fusion_oneyear_direct_from_ckpt.sh`
- `fit_fusion_oneyear_residual_unfreeze.sh`

## 4. 数据层

### 4.1 原始 FIT 样本

当前 FIT 原始 json 主要用到这些字段：

- `series`
- `target`
- `annotations`
- `metadata.group`
- `metadata.element`
- `metadata.norm`

### 4.2 公共数据工具

`data_provider/fit_dataset_utils.py` 统一负责：

- `CITY_MAP / GENDER_MAP / AGE_MAP`
- `group` 解析
- `element_map`
- train / val / test 划分
- 标准化
- 时间特征
- `caption_emb` 对齐
- 进程内缓存

### 4.3 scaler 语义

当前支持：

- `fit_scaler_mode=train_only`
- `fit_scaler_mode=split_fit`

默认：

- `train_only`

含义：

- `train_only`
  - 只用 train split 拟合 scaler
  - val/test 复用同一个 train-fit scaler
- `split_fit`
  - 保留旧行为
  - train/val/test 各自拟合自己的 scaler

### 4.4 本地 smoke 截断

当前支持：

- `max_train_samples`
- `max_val_samples`
- `max_test_samples`

约定：

- `<= 0` 表示不截断
- 截断发生在正式 split 之后
- 默认不会影响服务器正式训练

## 5. 数值流

### 5.1 fit_num

链路：

`run.py` -> `Exp_Fit_Num` -> `Dataset_DualSG_Fit_Num_Meta` -> `Model_Fit_Num`

特点：

- 只消费数值序列
- dataset 仍会返回 meta tuple，但 `Model_Fit_Num` 自己不用这些 meta

### 5.2 fit_num_with_meta

链路：

`run.py` -> `Exp_Fit_Num` -> `Dataset_DualSG_Fit_Num_Meta` -> `Model_Fit_Num_With_Meta`

特点：

- 使用数值序列 + `city / gender / age / element`
- 这是当前最强数值 baseline
- 也是 FIT fusion 当前复用的数值 backbone

### 5.3 `Model_Fit_Num_With_Meta`

当前保持原有训练语义不变：

1. Instance Norm
2. Patch Embedding
3. 融合 group / element 元数据
4. AIM
5. Transformer Encoder
6. Head 输出未来序列
7. De-Norm

本轮只新增：

- `extract_features()`

返回：

- `forecast`
- `encoded_tokens`
- `summary_state`

这个接口只给 fusion 读取中间特征，不改变数值 baseline 的训练方式。

## 6. FIT fusion

### 6.1 当前定位

当前 FIT fusion 仍然使用预处理好的 `caption_emb`，不在训练时接 raw text encoder。

文本流当前链路是：

1. 先生成文本描述
2. 离线编码成 `caption_emb`
3. 训练时直接读 `caption_emb`

### 6.2 数值 backbone 复用方式

FIT fusion 当前复用：

- `Model_Fit_Num_With_Meta`

复用方式不是简单调用 `forward()`，而是显式调用：

- `extract_features()`

这样 fusion 能稳定拿到：

- 数值预测 `y_num`
- 中间特征 `encoded_tokens`
- 数值摘要 `summary_state`

### 6.3 当前 fusion 主体结构

当前 FIT fusion 不是“文本单独再预测一个序列”，而是“文本先调制数值中间特征”。

核心结构：

- `caption_emb -> text_adapter`
- 文本生成条件参数 `gamma / beta`
- 用 `gamma / beta` 调制 `encoded_tokens`
- 再从融合后的上下文里输出最终预测

融合上下文来自：

- 文本上下文
- 调制后的数值摘要
- 原始数值预测 `y_num` 的摘要
- 历史统计特征

### 6.4 历史统计特征

当前直接从 `x_enc` 提取：

- `last`
- `mean`
- `std`
- `slope`
- `range`

### 6.5 direct 模式

参数：

- `text_mode=direct`

当前语义：

- 文本先调制数值 latent
- 再从融合上下文直接输出最终未来序列

当前已经不再使用旧版的浅层线性融合：

- `(1 - w) * y_num + w * y_text`

### 6.6 residual 模式

参数：

- `text_mode=residual`

当前语义：

- 先得到数值主预测 `y_num`
- 文本分支只预测一个“有边界的纠偏量”

当前 residual 约束方式：

- 不再使用 `beta_delta`
- 不再使用 `force_gain`
- 直接用历史 `std + range` 给纠偏量设边界

最终形式：

- `y_final = y_num + delta`

### 6.7 freeze / joint

当前 FIT fusion 支持：

- 从头 joint 训练
- 从数值 ckpt 启动
- 冻结数值流
- 不冻结数值流联合更新

关键参数：

- `num_model_path`
- `freeze_numerical`
- `text_mode`
- `fusion_optimizer_mode`

## 7. FIT fusion 优化器

### 7.1 unified

参数：

- `fusion_optimizer_mode=unified`

语义：

- 所有可训练参数共用一个 `learning_rate`

### 7.2 split

参数：

- `fusion_optimizer_mode=split`

语义：

- 数值 backbone 参数使用 `lr_num`
- 文本 / 融合参数使用 `lr_text`
- 文本分支可单独设 `weight_decay_text`

当前实现已经修复：

- 每个 param group 都保留自己的 `base_lr`
- scheduler 按各组自己的 `base_lr` 做相对缩放
- 不会再把所有组统一覆盖成 `args.learning_rate`

## 8. 当前脚本语义

### 8.1 数值流

- `fit_halfyear_num.sh`
- `fit_halfyear_num_with_meta.sh`
- `fit_oneyear_num.sh`
- `fit_oneyear_num_with_meta.sh`

这些脚本都已显式带上：

- `fit_scaler_mode=train_only`

### 8.2 joint_direct

脚本：

- `fit_fusion_halfyear_direct.sh`
- `fit_fusion_oneyear_direct.sh`

语义：

- 从头 joint 训练
- `text_mode=direct`
- 默认不加载数值 ckpt
- 默认不冻结数值流

当前脚本显式设置：

- `train_epochs=100`
- `batch_size=200`
- `patience=30`
- `adjust=0`

### 8.3 residual_correction

脚本：

- `fit_fusion_halfyear_residual.sh`
- `fit_fusion_oneyear_residual.sh`

语义：

- 从数值 ckpt 启动
- `text_mode=residual`
- 默认冻结数值流

说明：

- 通过 shell 变量 `NUM_CKPT` 指定数值 checkpoint
- 当前脚本显式设置：
  - `train_epochs=100`
  - `batch_size=200`
  - `patience=30`
  - `adjust=0`

### 8.4 推荐补充对照脚本

为了给文本实验一个更公平的比较起点，当前额外提供：

- `fit_fusion_halfyear_direct_from_ckpt.sh`
- `fit_fusion_halfyear_residual_unfreeze.sh`
- `fit_fusion_oneyear_direct_from_ckpt.sh`
- `fit_fusion_oneyear_residual_unfreeze.sh`

它们分别覆盖：

- `direct + load num ckpt + unfreeze`
- `residual + load num ckpt + unfreeze`

这些脚本当前统一支持环境变量覆盖：

- `TRAIN_EPOCHS`
- `BATCH_SIZE`
- `PATIENCE`
- `ADJUST`
- `LEARNING_RATE`
- `LR_NUM`
- `LR_TEXT`
- `CAPTION_EMB_PATH`
- `DATA_PATH`

## 9. FIT 已经不再使用的旧 fusion 项

FIT 当前实现已经不再依赖：

- `beta_delta`
- `force_gain`
- 写死输入维度的 `llm_dim`

注意：

- `run.py` 里仍保留 `llm_dim / delta_scale / use_vol_prior / force_gain`
- 这是为了 Geo 兼容
- 它们当前不再参与 FIT fusion

## 10. 本地验证状态

### 10.1 已装依赖

当前 base Python 可用：

- `torch` CPU
- `numpy`
- `scipy`
- `scikit-learn`
- `pandas`
- `matplotlib`
- `tqdm`
- `einops`
- `reformer_pytorch`

### 10.2 已通过的检查

静态检查：

- `python -m compileall -q run.py exp models data_provider utils layers`

真实 smoke：

- `fit_num`
- `fit_num_with_meta`
- `fit_num_with_meta` 在增加 `extract_features()` 之后再次跑通
- 新版 `fit_fusion direct + split`
- 新版 `fit_fusion residual + split`
- 新版 `fit_fusion direct + unified`

说明：

- 正式 `fit_caption_emb_all.pt` 当前不在本地
- 之前为了验证 fusion 代码路径，使用过临时小子集和临时 `caption_emb`
- 临时资产已经删除，不参与正式实验

## 11. 当前已知限制

### 11.1 正式效果仍要靠服务器实验确认

当前已经确认：

- 代码路径通了
- `direct / residual` 都能跑
- `split / unified` 都能跑

当前还未确认：

- 新结构在正式 `caption_emb` 和全量训练上是否一定优于旧结构

### 11.2 文本上限仍由描述质量决定

即使 fusion 结构更合理，文本分支上限仍然取决于：

- 文本描述是否真的捕捉了趋势模式
- 文本描述是否与数值走势强相关
- 预处理 `caption_emb` 是否足够表达这些描述

### 11.3 当前官方 FIT fusion 脚本默认关闭 scheduler

当前 FIT fusion 官方脚本默认显式加了：

- `--adjust 0`

原因：

- 当前这一轮先不想把 scheduler 重新混进实验判断
- 默认保持更长训练 + split lr + 无 scheduler
- 这样更利于判断结构本身和初始化策略本身有没有帮助

如果后续要重新启用 scheduler，需要明确：

- 是否增加训练轮数
- 或者换一个更适合短程训练的调度策略

## 12. 当前建议实验顺序

优先先做半年任务：

1. `fit_num_with_meta`
2. `fit_fusion_halfyear_direct.sh`
3. `fit_fusion_halfyear_direct_from_ckpt.sh`
4. `fit_fusion_halfyear_residual.sh`
5. `fit_fusion_halfyear_residual_unfreeze.sh`

如果半年任务里出现明确提升，再复制到一年任务。
