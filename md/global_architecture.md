# FIT Architecture

最后更新：2026-04-13

## 1. 文档定位

这份文档只描述当前仓库里 FIT 相关的真实实现。

配套文档：

- `fit_refactor_worklog.md`：整理过程、smoke 结果、已知问题和实验结论
- `fit_server_runbook.md`：服务器执行顺序与结果记录模板
- `项目文件结构.md`：当前项目文件树

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

脚本层：

- `fit_halfyear_num.sh`
- `fit_halfyear_num_with_meta.sh`
- `fit_oneyear_num.sh`
- `fit_oneyear_num_with_meta.sh`
- `fit_fusion_halfyear_direct.sh`
- `fit_fusion_halfyear_direct_from_ckpt.sh`
- `fit_fusion_halfyear_residual.sh`
- `fit_fusion_halfyear_residual_unfreeze.sh`
- `fit_fusion_oneyear_direct.sh`
- `fit_fusion_oneyear_direct_from_ckpt.sh`
- `fit_fusion_oneyear_residual.sh`
- `fit_fusion_oneyear_residual_unfreeze.sh`

## 4. 数据层

当前 FIT 原始数据仍然读取：

- `dataset/FIT_DualSG/fit_dualsg_all.json`

当前 FIT fusion 默认读取的文本 embedding 是：

- `dataset/FIT_DualSG/pt/fit_dualsg_structured.pt`

`fit_dataset_utils.py` 统一负责：

- `CITY_MAP / GENDER_MAP / AGE_MAP`
- `group` 解析
- `element_map`
- train / val / test 划分
- 标准化
- 时间特征
- `caption_emb` 对齐
- 进程内缓存

当前默认 scaler 语义：

- `fit_scaler_mode=train_only`

含义：

- 只用 train split 拟合 scaler
- val/test 复用同一个 train-fit scaler

## 5. 数值流

### 5.1 fit_num

链路：

`run.py` -> `Exp_Fit_Num` -> `Dataset_DualSG_Fit_Num_Meta` -> `Model_Fit_Num`

### 5.2 fit_num_with_meta

链路：

`run.py` -> `Exp_Fit_Num` -> `Dataset_DualSG_Fit_Num_Meta` -> `Model_Fit_Num_With_Meta`

这是当前最强数值 baseline，也是 FIT fusion 当前复用的数值 backbone。

### 5.3 `Model_Fit_Num_With_Meta`

当前保持原有训练语义不变，只额外提供：

- `extract_features()`

返回：

- `forecast`
- `encoded_tokens`
- `summary_state`

这个接口只给 fusion 读取中间特征，不改变数值 baseline 的训练方式。

## 6. FIT fusion

### 6.1 当前定位

当前 FIT fusion 使用预处理好的 `caption_emb`，不在训练时接 raw text encoder。

### 6.2 数值 backbone 复用方式

FIT fusion 当前复用：

- `Model_Fit_Num_With_Meta`

并通过：

- `extract_features()`

拿到：

- 数值预测 `y_num`
- 中间特征 `encoded_tokens`
- 数值摘要 `summary_state`

### 6.3 当前主体结构

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

### 6.4 direct 模式

参数：

- `text_mode=direct`

语义：

- 文本先调制数值 latent
- 再从融合上下文直接输出最终未来序列

### 6.5 residual 模式

参数：

- `text_mode=residual`

语义：

- 先得到数值主预测 `y_num`
- 文本分支只预测一个有边界的纠偏量

最终形式：

- `y_final = y_num + delta`

### 6.6 当前已经删除的旧思路

FIT 当前实现已经不再依赖：

- `beta_delta`
- `force_gain`
- 写死输入维度的 `llm_dim`

## 7. 当前推荐脚本语义

### 7.1 半年任务

- `fit_fusion_halfyear_direct.sh`
  - direct
  - 从头训练
- `fit_fusion_halfyear_direct_from_ckpt.sh`
  - direct
  - 加载数值 ckpt
  - 不冻结数值流
- `fit_fusion_halfyear_residual.sh`
  - residual
  - 加载数值 ckpt
  - 冻结数值流
- `fit_fusion_halfyear_residual_unfreeze.sh`
  - residual
  - 加载数值 ckpt
  - 不冻结数值流

### 7.2 一年任务

语义与半年任务对应，只是 `pred_len=24`。

## 8. 当前脚本默认超参

当前 FIT fusion 默认：

- `train_epochs=20`
- `batch_size=200`
- `patience=100`
- `fusion_optimizer_mode=split`
- `learning_rate=0.001`
- `lr_num=0.0001`
- `lr_text=0.0005`
- `adjust=0`
- `fit_scaler_mode=train_only`

当前脚本已恢复成显式常量写法。

如果需要改参数，直接打开对应 `.sh` 修改即可。

## 9. 当前实验判断

基于半年的 20 epoch 正式结果：

- `direct + scratch` 明显弱
- `direct + num_ckpt + unfreeze` 已优于 baseline
- `residual + num_ckpt + freeze` 略优于 baseline
- `residual + num_ckpt + unfreeze` 当前最好

因此当前更合理的策略是：

1. 先把半年任务的 100 epoch 跑完
2. 再决定是否继续改结构
3. 再考虑是否把同样改法复制到一年任务

## 10. 当前建议阅读顺序

1. `fit_server_runbook.md`
2. `fit_refactor_worklog.md`
3. `global_architecture.md`
