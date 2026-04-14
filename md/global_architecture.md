# FIT 架构文档

最后更新：2026-04-14

## 文档定位

这份文档只描述当前 `0414` 分支里 FIT 的活跃实现。

`0411` 阶段的诊断脚本、旧结构和结果快照已经归档，不再作为当前主线的一部分：

- [fit_halfyear_0411_manifest.md](/D:/zhangjing/project/Dualsg_refined/md/fit_halfyear_0411_manifest.md)
- [scripts_archive/fit_halfyear_0411/README.md](/D:/zhangjing/project/Dualsg_refined/scripts_archive/fit_halfyear_0411/README.md)

配套文档：

- [fit_server_runbook.md](/D:/zhangjing/project/Dualsg_refined/md/fit_server_runbook.md)
- [fit_refactor_worklog.md](/D:/zhangjing/project/Dualsg_refined/md/fit_refactor_worklog.md)
- [项目文件结构.md](/D:/zhangjing/project/Dualsg_refined/md/项目文件结构.md)

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

## FIT 数据契约

当前 FIT 数值数据仍然读取：

- `dataset/FIT_DualSG/fit_dualsg_all.json`

当前文本输入仍然是离线 embedding：

- `caption_emb_path -> *.pt`

也就是说：

- `json` 决定样本顺序、数值序列和 meta
- `pt` 只在这个顺序上提供每条样本对应的文本 embedding

因此切换长文本、结构化文本、random text、filler text 时：

- 不需要换 `json`
- 只需要换 `caption_emb_path`

前提是：

- `pt` 的样本数必须和 `fit_dualsg_all.json` 一致
- 顺序必须严格一致

## 标准化

当前 FIT 默认使用：

- `fit_scaler_mode=train_only`

含义：

- 只用 train split 拟合 scaler
- val / test 复用这个 train-fit scaler

这是当前推荐的正式语义。

## 数值流

### `fit_num`

链路：

`run.py` -> `Exp_Fit_Num` -> `Dataset_DualSG_Fit_Num_Meta` -> `Model_Fit_Num`

### `fit_num_with_meta`

链路：

`run.py` -> `Exp_Fit_Num` -> `Dataset_DualSG_Fit_Num_Meta` -> `Model_Fit_Num_With_Meta`

这是当前 FIT 最强的数值 baseline，也是 fusion 当前复用的 backbone。

### `Model_Fit_Num_With_Meta`

当前保持原有训练语义不变，只额外提供：

- `extract_features()`

返回：

- `forecast`
- `encoded_tokens`
- `summary_state`

这个接口只给 fusion 读取中间特征，不改变数值 baseline 的训练方式。

## 0414 FIT Fusion 主线

### 总体原则

`0414` 活跃代码只保留一套新主线，不再在运行时保留 `modern / legacy` 切换。

当前 `Model_Fit_Fusion` 只有两个模式：

- `text_mode=direct`
- `text_mode=residual`

两者共用：

- 同一个数值 backbone：`Model_Fit_Num_With_Meta`
- 同一个共享文本适配器：`caption_emb -> text_proj -> text_ctx`

### 当前关键参数

当前 0414 主线真正使用的 FIT fusion 参数：

- `text_mode`
- `num_model_path`
- `caption_emb_path`
- `freeze_numerical`
- `disable_text`
- `fusion_optimizer_mode`
- `lr_num`
- `lr_text`
- `weight_decay_text`
- `text_hidden`
- `residual_rank`
- `num_feat_dim`
- `fusion_hidden`
- `fusion_dropout`

## Direct v2

### 设计目标

`direct` 是一条可解释的双流基线。

它不追求一定强于 residual，但必须满足两点：

1. 文本流单独生成预测
2. 数值主预测 `y_num` 显式进入最终输出

### 结构

数值流：

- 数值 backbone 输出 `y_num`

文本流：

- `caption_emb -> text_proj -> text_ctx`
- `text_ctx -> direct_text_head -> y_text`

融合门控：

- `gate_input = [text_ctx, summary_ctx, y_num_ctx, hist_ctx]`
- `gate = sigmoid(direct_gate_head(gate_input))`

最终输出：

- `y_final = (1 - gate) * y_num + gate * y_text`

其中：

- `gate` 是逐步预测的 `[B, pred_len, C]`
- 不是全局标量

### 当前含义

这条路保留了强显式数值 skip。

因此：

- 即使文本流还不够强
- `direct` 也不会像之前那版 “把 `y_num` 埋进大 MLP” 一样容易失稳

## Residual v2

### 设计目标

`residual` 是当前主方法。

它的目标不是“再做一个纯数值纠偏器”，而是：

- 数值侧只生成可供纠偏的 basis
- 文本侧必须提供 basis 系数和纠偏幅度控制

这样能在结构上避免“只靠数值侧就把 delta 直接算出来”的旁路。

### 结构

主预测：

- 数值 backbone 输出 `y_num`

数值侧上下文：

- `encoded_tokens`
- `summary_state`
- `y_num`
- `hist_stats`

数值侧 basis：

- `basis_input = [encoded_ctx, summary_ctx, y_num_ctx, hist_ctx]`
- `residual_basis = residual_basis_head(basis_input)`
- reshape 为 `[B, pred_len, C, R]`

文本侧控制：

- `text_ctx = text_proj(caption_emb)`
- `text_coeff = residual_text_coeff_head(text_ctx)`
- reshape 为 `[B, pred_len, R]`

纠偏幅度：

- `radius_input = [text_ctx, hist_ctx]`
- `radius = sigmoid(residual_radius_head(radius_input))`

最终纠偏：

- `delta_raw = sum_k residual_basis[..., k] * text_coeff[..., k]`
- `delta = radius * tanh(delta_raw)`
- `y_final = y_num + delta`

### 当前约束

当前 residual 里不存在：

- “纯数值 MLP 直接输出 `delta`” 的路径

也就是说：

- 如果没有文本系数
- residual basis 不能自己变成最终纠偏量

## Disable Text

参数：

- `--disable_text`

当前语义是：

- 直接返回数值预测 `y_num`
- 冻结所有非数值参数

因此它是一个严格的纯数值对照。

## 当前脚本入口

当前 0414 活跃的 FIT half-year fusion 脚本只有两个：

- [fit_fusion_halfyear_direct_v2.sh](/D:/zhangjing/project/Dualsg_refined/fit_fusion_halfyear_direct_v2.sh)
- [fit_fusion_halfyear_residual_v2.sh](/D:/zhangjing/project/Dualsg_refined/fit_fusion_halfyear_residual_v2.sh)

首轮固定配置：

- `half-year`
- `real long text = ./dataset/FIT_DualSG/pt/fit_dualsg_all.pt`
- `num_model_path = ./model_checkpoints/fit_halfyear_num_with_meta_20260413_083428/checkpoint.pth`
- `unfreeze numerical`
- `train_epochs = 20`
- `patience = 100`
- `adjust = 0`
- `fusion_optimizer_mode = split`
- `fit_scaler_mode = train_only`

## 0411 归档和 0414 主线的关系

`0411` 已经完成的事情：

- modern / legacy 结构诊断
- random / filler / disable_text 控制实验
- structured / multi-view 文本诊断
- 旧脚本归档

`0414` 的定位是：

- 不再继续堆旧结构实验
- 直接进入一套更干净的新主线
- 重点不是“换更多 prompt”
- 而是“从结构上减少文本被绕开的可能性”
