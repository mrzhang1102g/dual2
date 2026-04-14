﻿# FIT 重构工作记录

最后更新：2026-04-14

## 本轮代码整理

### 数据层

- 抽出 FIT 公共数据逻辑到 `data_provider/fit_dataset_utils.py`
- 统一处理：
  - 城市、性别、年龄映射
  - `group` 解析
  - train / val / test 切分
  - `element_map`
  - 标准化
  - 时间特征
  - `caption_emb` 对齐
- 新增 `fit_scaler_mode`
  - `train_only`
  - `split_fit`
- 当前默认：
  - `fit_scaler_mode=train_only`

### 数值流

- 保持 `fit_num` 和 `fit_num_with_meta` 的训练逻辑不变
- 只给 `Model_Fit_Num_With_Meta` 增加 `extract_features()`，供 fusion 读取：
  - `forecast`
  - `encoded_tokens`
  - `summary_state`

### 新版 FIT fusion

- 保留离线 `caption_emb`
- 仍然分为 `direct` / `residual`
- 当前主干仍然基于 `Model_Fit_Num_With_Meta`
- 当前实验矩阵已经补齐到：
  - `direct + scratch`
  - `direct + ckpt + freeze`
  - `direct + ckpt + unfreeze`
  - `residual + ckpt + freeze`
  - `residual + ckpt + unfreeze`

### 脚本与训练设置

- 当前 half-year fusion 脚本统一使用：
  - `train_epochs = 20`
  - `batch_size = 200`
  - `patience = 100`
  - `adjust = 0`
- `split lr` 被 scheduler 覆盖的问题已修复

## 当前结果总结

### baseline

- `fit_num_with_meta`
  - `MAE = 0.085858`
  - `MSE = 0.013315`
  - `RMSE = 0.115391`
  - `MAPE = 30.23%`
  - `WAPE = 18.52%`

### 长文本

- 20 epoch 最好：
  - `residual + ckpt + unfreeze`
  - `MAE = 0.079863`
  - `MAPE = 28.83%`
- 100 epoch 最好：
  - `residual + ckpt + unfreeze`
  - `MAE = 0.075699`
  - `MAPE = 27.30%`

### 结构化与多视角文本

目前已经完成：

- 结构化完整文本
- Global Structural View
- Dynamic Behavior View
- Event-centric View
- Semantic Caption View

这些 20 epoch 结果排序都很稳定：

1. `residual + ckpt + unfreeze`
2. `direct + ckpt + unfreeze`
3. `residual + ckpt + freeze`
4. `direct + ckpt + freeze`
5. `direct + scratch`

这些文本版本之间只在小数点后三位附近波动，没有带来跨档提升。

## 新增控制实验

### random text

结果：

- `direct + ckpt + freeze`
  - `MAE = 0.087233`
  - `MAPE = 33.08%`
- `direct + scratch`
  - `MAE = 0.105321`
  - `MAPE = 40.39%`
- `direct + ckpt + unfreeze`
  - `MAE = 0.081796`
  - `MAPE = 30.96%`
- `residual + ckpt + freeze`
  - `MAE = 0.085365`
  - `MAPE = 30.54%`
- `residual + ckpt + unfreeze`
  - `MAE = 0.079736`
  - `MAPE = 28.72%`

### filler text

结果：

- `direct + ckpt + freeze`
  - `MAE = 0.087151`
  - `MAPE = 33.05%`
- `direct + scratch`
  - `MAE = 0.105571`
  - `MAPE = 40.74%`
- `direct + ckpt + unfreeze`
  - `MAE = 0.081904`
  - `MAPE = 30.96%`
- `residual + ckpt + freeze`
  - `MAE = 0.085345`
  - `MAPE = 30.51%`
- `residual + ckpt + unfreeze`
  - `MAE = 0.079818`
  - `MAPE = 28.81%`

### disable_text

结果：

- `direct + ckpt + unfreeze + disable_text`
  - `MAE = 0.080148`
  - `MSE = 0.011753`
  - `RMSE = 0.108411`
  - `MAPE = 28.70%`
  - `WAPE = 17.29%`
- `residual + ckpt + unfreeze + disable_text`
  - `MAE = 0.080148`
  - `MSE = 0.011753`
  - `RMSE = 0.108411`
  - `MAPE = 28.70%`
  - `WAPE = 17.29%`

## 当前最关键结论

### 结论 1

random / filler / 真实长文本 结果几乎一样。

说明：

- 当前模型几乎没有利用文本语义本身

### 结论 2

`disable_text` 与当前最好路线几乎一样。

说明：

- 当前新版 fusion 的主要增益不是来自文本语义
- 更像来自：
  - `ckpt + unfreeze`
  - `residual` 的硬数值 skip
  - 数值侧辅助特征本身

### 结论 3

`disable_text=True` 时，`direct` 和 `residual` 会得到完全一样的结果，这是当前实现的正常行为。

原因：

- 一旦 `disable_text=True`
- 模型直接返回数值预测
- `text_mode` 不再参与 forward

### 结论 4

当前最准确的问题表述是：

- 当前新版 fusion 结构允许模型完全绕开文本
- 因此即使换成随机文本或无语义模板文本，结果也几乎不变

### 结论 5

当前不适合继续堆更多文本版本。

更合理的下一步只有两条：

1. 回退旧版 fusion 头，在当前训练体系下做严格 A/B
2. 继续重写新版 fusion，让文本必须真正参与预测

## 旧版最好结果

用户此前最好结果：

- `MAE = 0.071344`
- `MSE = 0.009260`
- `RMSE = 0.096228`
- `MAPE = 25.23%`
- `WAPE = 15.39%`

当前新版 fusion 还明显没有追上这组结果。

## 当前进入 legacy 恢复阶段

当前分支不再继续堆更多文本版本，而是开始做：

- old heads, new framework

也就是：

- 保留当前清理过的数据层
- 保留 train-only scaler
- 保留当前 `Exp_Fit_Fusion`
- 保留当前脚本/输出结构
- 只把旧版 `direct` / `residual` fusion 头恢复回来

本轮已经落地：

- `run.py` 新增 `fusion_version = modern / legacy`
- `Model_Fit_Fusion` 内部同时支持：
  - `modern + direct/residual`
  - `legacy + direct/residual`
- `disable_text` 在两种 version 下都统一为：
  - 直接返回 `y_num`
  - 冻结所有非数值参数

新增诊断脚本：

- `fit_fusion_halfyear_legacy_direct_from_ckpt.sh`
- `fit_fusion_halfyear_legacy_residual_unfreeze.sh`

当前这轮 legacy 恢复的目标不是直接写结论，而是先回答：

- 旧版 fusion 头在现在这套干净训练框架下，还能不能重新追回明显优势

如果 legacy 依然明显强于 current modern：

- 下一轮再补 `random / filler` 对照

如果 legacy 也起不来：

- 说明旧版最好结果未必只来自旧头
- 下一轮应直接进入“强制使用文本”的结构重写，而不是继续换文本 prompt

## 0411 快照归档

为了避免后续 0414 分支继续重写结构时污染这一轮诊断结果，当前已经把本阶段 half-year 脚本快照单独归档到：

- `scripts_archive/fit_halfyear_0411/`

归档内容包括：

- baseline 脚本
- modern 长文本 20 / 100 epoch
- modern random / filler 对照
- modern 结构化与多视角文本
- disable_text 对照
- legacy long text 诊断

对应总清单文档：

- [fit_halfyear_0411_manifest.md](/D:/zhangjing/project/Dualsg_refined/md/fit_halfyear_0411_manifest.md)

这样后续即使根目录活跃脚本继续演化，这一轮 0411 阶段的 `.sh` 配置也还能完整追溯和复现。

## 第一轮 legacy 诊断结果

首轮只跑了：

- `legacy direct + ckpt + unfreeze + real long text`
- `legacy residual + ckpt + unfreeze + real long text`

结果：

- `legacy direct`
  - `MAE = 0.080265`
  - `MSE = 0.011629`
  - `RMSE = 0.107840`
  - `MAPE = 29.75%`
  - `WAPE = 17.32%`
- `legacy residual`
  - `MAE = 0.080123`
  - `MSE = 0.011747`
  - `RMSE = 0.108384`
  - `MAPE = 28.68%`
  - `WAPE = 17.28%`

当前解读：

- `legacy direct` 明显优于 `modern direct` 的同 recipe 结果，说明旧版显式 `y_num` skip 对 direct 确实更稳
- `legacy residual` 与 `modern residual` 基本打平，说明只恢复旧 residual 头本身，并没有把性能拉回用户很久之前的旧最好结果
- 两个 legacy 结果都明显落后于那组旧版最好结果：
  - `MAE = 0.071344`
  - `MAPE = 25.23%`

这意味着：

- “恢复旧头”本身不够
- 旧版最好结果未必只来自旧 fusion 头
- 还可能混有当时旧的数据预处理语义、训练设定，甚至旧流程里的其他非当前因素

另外一个重要结论：

- `legacy residual` 与 `disable_text` 几乎一样

这说明：

- 即使换回 legacy residual 头，在当前清理后的训练框架里，也还没有坐实“文本语义真正贡献了提升”

## 第二轮 legacy 诊断结果：100 epoch

用户随后补跑了：

- `legacy direct + ckpt + unfreeze`
- `legacy residual + ckpt + unfreeze`

100 epoch 结果：

- `legacy direct`
  - `MAE = 0.075869`
  - `MSE = 0.010493`
  - `RMSE = 0.102437`
  - `MAPE = 27.84%`
  - `WAPE = 16.37%`
- `legacy residual`
  - `MAE = 0.076114`
  - `MSE = 0.010665`
  - `RMSE = 0.103272`
  - `MAPE = 27.12%`
  - `WAPE = 16.42%`

当前解读更新为：

- `legacy 100 epoch` 明显强于 `legacy 20 epoch`
- `legacy direct` 继续证明旧版显式数值 skip 对 direct 更稳
- `legacy residual` 在 `MAPE` 上优于 `legacy direct`，这和“residual 更适合作为主方法”的直觉是一致的
- 但两者和用户最早那组最好结果相比，仍有明显差距

因此：

- 0411 分支已经足够完成“诊断归档”的职责
- 它说明旧结构里存在有效设计，但也说明历史最好结果不是只靠恢复旧头就能直接找回
- 后续 0414 分支继续重写新结构是合理且必要的
