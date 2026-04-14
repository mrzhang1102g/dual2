# FIT 架构文档

最后更新：2026-04-14

## 文档定位

这份文档只描述当前仓库里 FIT 相关代码的真实实现。

配套文档：

- [fit_server_runbook.md](/D:/zhangjing/project/Dualsg_refined/md/fit_server_runbook.md)
  当前实验结果、服务器执行记录、对照实验结论
- [fit_refactor_worklog.md](/D:/zhangjing/project/Dualsg_refined/md/fit_refactor_worklog.md)
  本轮整理、重构、当前阶段结论
- [项目文件结构.md](/D:/zhangjing/project/Dualsg_refined/md/项目文件结构.md)
  当前项目文件树

## 当前入口

统一入口：

- `run.py`

任务到实验类：

- `fit_num` -> `exp/exp_fit_num.py`
- `fit_num_with_meta` -> `exp/exp_fit_num.py`
- `fit_fusion` -> `exp/exp_fit_fusion.py`

任务到模型：

- `Model_Fit_Num` -> `models/model_fit_num.py`
- `Model_Fit_Num_With_Meta` -> `models/model_fit_num_with_meta.py`
- `Model_Fit_Fusion` -> `models/model_fit_fusion.py`

任务到数据集：

- `FIT_Meta` -> `data_provider/data_loader_fit_num_meta.py`
- `FIT_Fusion` -> `data_provider/data_loader_fit_fusion.py`

## FIT 关键文件

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

脚本层：

- `fit_halfyear_num.sh`
- `fit_halfyear_num_with_meta.sh`
- `fit_oneyear_num.sh`
- `fit_oneyear_num_with_meta.sh`
- `fit_fusion_halfyear_direct.sh`
- `fit_fusion_halfyear_direct_freeze.sh`
- `fit_fusion_halfyear_direct_from_ckpt.sh`
- `fit_fusion_halfyear_residual.sh`
- `fit_fusion_halfyear_residual_unfreeze.sh`
- `fit_fusion_halfyear_direct_from_ckpt_disable_text.sh`
- `fit_fusion_halfyear_residual_unfreeze_disable_text.sh`

## 数据层

当前 FIT 数值数据仍然读取：

- `dataset/FIT_DualSG/fit_dualsg_all.json`

当前文本输入仍然是离线 embedding：

- `caption_emb_path -> *.pt`

`fit_dataset_utils.py` 统一负责：

- `CITY_MAP / GENDER_MAP / AGE_MAP`
- `group` 解析
- `element_map`
- train / val / test 切分
- 标准化
- 时间特征
- `caption_emb` 对齐
- 进程内缓存

当前默认 scaler 语义：

- `fit_scaler_mode=train_only`

含义：

- 只用 train split 拟合 scaler
- val / test 复用同一个 train-fit scaler

## 数值流

### `fit_num`

链路：

`run.py` -> `Exp_Fit_Num` -> `Dataset_DualSG_Fit_Num_Meta` -> `Model_Fit_Num`

### `fit_num_with_meta`

链路：

`run.py` -> `Exp_Fit_Num` -> `Dataset_DualSG_Fit_Num_Meta` -> `Model_Fit_Num_With_Meta`

这是当前 FIT 最强的数值 baseline，也是 fusion 当前复用的数值 backbone。

### `Model_Fit_Num_With_Meta`

当前保持原有训练语义不变，只额外提供：

- `extract_features()`

返回：

- `forecast`
- `encoded_tokens`
- `summary_state`

这个接口只给 fusion 读取中间特征，不改变数值 baseline 的训练方式。

## FIT fusion

### 当前定位

当前 FIT fusion 仍然使用预处理好的 `caption_emb`，不在训练时接 raw text encoder。

### 数值 backbone 复用方式

FIT fusion 当前复用：

- `Model_Fit_Num_With_Meta`

并通过：

- `extract_features()`

拿到：

- 数值预测 `y_num`
- 中间特征 `encoded_tokens`
- 数值摘要 `summary_state`

### 当前主体结构

当前新版 FIT fusion 不是“文本单独预测一个序列，再和数值结果浅加权”，而是“文本先调制数值中间特征，再输出预测”。

核心链路：

- `caption_emb -> text_adapter`
- 文本生成条件参数 `gamma / beta`
- `gamma / beta` 调制 `encoded_tokens`
- 再从融合上下文里输出最终预测

融合上下文来自：

- 文本上下文
- 调制后的数值摘要
- 原始数值预测 `y_num` 的摘要
- 历史统计特征

### `direct`

参数：

- `text_mode=direct`

语义：

- 文本先调制数值 latent
- 再从融合上下文直接输出最终未来序列

注意：

- 当前新版 `direct` 没有旧版那种显式的 `y_num` 硬 skip
- 所以它更灵活，但也更容易让模型不稳定

### `residual`

参数：

- `text_mode=residual`

语义：

- 先得到数值主预测 `y_num`
- 再输出一个有边界的纠偏量

最终形式：

- `y_final = y_num + delta`

因此 `residual` 天然比 `direct` 更稳。

### `disable_text`

参数：

- `--disable_text`

语义：

- fusion loader 仍然会读入 `caption_emb`
- 但模型会直接返回数值预测
- `text_mode=direct` 和 `text_mode=residual` 不再参与实际 forward

所以：

- `direct + disable_text`
- `residual + disable_text`

在当前实现里本质上是同一个实验。

### 当前已移除的旧思路

FIT 当前实现已经不再依赖：

- `beta_delta`
- `force_gain`
- 写死输入维度的 `llm_dim`

## 当前脚本语义

### half-year

- `fit_fusion_halfyear_direct.sh`
  - `direct`
  - 从头训练
- `fit_fusion_halfyear_direct_freeze.sh`
  - `direct`
  - 加载数值 ckpt
  - 冻结数值流
- `fit_fusion_halfyear_direct_from_ckpt.sh`
  - `direct`
  - 加载数值 ckpt
  - 不冻结数值流
- `fit_fusion_halfyear_residual.sh`
  - `residual`
  - 加载数值 ckpt
  - 冻结数值流
- `fit_fusion_halfyear_residual_unfreeze.sh`
  - `residual`
  - 加载数值 ckpt
  - 不冻结数值流
- `fit_fusion_halfyear_direct_from_ckpt_disable_text.sh`
  - `disable_text`
  - 当前用于验证文本是否真正起作用
- `fit_fusion_halfyear_residual_unfreeze_disable_text.sh`
  - `disable_text`
  - 当前用于验证文本是否真正起作用

### one-year

语义和 half-year 对应，只是 `pred_len=24`。

## 当前默认训练配置

当前 FIT fusion 脚本默认：

- `train_epochs=20`
- `batch_size=200`
- `patience=100`
- `fusion_optimizer_mode=split`
- `learning_rate=0.001`
- `lr_num=0.0001`
- `lr_text=0.0005`
- `adjust=0`
- `fit_scaler_mode=train_only`

当前脚本都是显式常量写法，直接改 `.sh` 即可。

## 当前实验结论对应到架构的解释

当前最关键的实验结论是：

- 真实长文本
- random text
- filler text
- `disable_text`

这几类结果都非常接近。

这说明当前新版 fusion 的主要问题不是“文本 prompt 不够好”，而是：

- 当前结构允许模型几乎完全绕开文本
- 模型可以只依赖数值侧特征，也拿到几乎一样的结果

因此当前结构上的真实问题是：

- 文本并没有被强制真正参与预测

## 当前阶段建议

当前这个版本的 half-year FIT 实验已经基本收束。

再继续换更多文本版本，价值已经不高。

更合理的下一步只有两条：

1. 回退旧版 fusion 头，在当前整理后的训练体系下做严格 A/B
2. 继续重写新版 fusion，让文本必须真正参与预测
