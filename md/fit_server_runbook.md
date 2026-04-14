﻿# FIT 服务器执行手册

最后更新：2026-04-14

配套文档：

- [fit_halfyear_0411_manifest.md](/D:/zhangjing/project/Dualsg_refined/md/fit_halfyear_0411_manifest.md)
  0411 阶段脚本归档、PT 路径、实验分组总清单
- [fit_refactor_worklog.md](/D:/zhangjing/project/Dualsg_refined/md/fit_refactor_worklog.md)
  当前诊断结论与 legacy 恢复工作记录
- [global_architecture.md](/D:/zhangjing/project/Dualsg_refined/md/global_architecture.md)
  当前 FIT 代码结构与 `modern/legacy` 融合实现

## 当前重点

当前只聚焦 FIT half-year。

当前看结果时优先级如下：

1. `MAE`
2. `MAPE`
3. `WAPE`
4. `MSE`

## 当前阶段

已经完成：

- 数值 baseline
- 长文本 20 epoch
- 长文本 100 epoch
- 长文本 random text 20 epoch
- 长文本 filler text 20 epoch
- `disable_text` 关键对照 20 epoch
- 结构化完整文本 20 epoch
- Global Structural View 20 epoch
- Dynamic Behavior View 20 epoch
- Event-centric View 20 epoch
- Semantic Caption View 20 epoch

当前最关键的新增结论来自：

- random text
- filler text
- `disable_text`

## 当前使用的关键文件

### 数值数据

- `./dataset/FIT_DualSG/fit_dualsg_all.json`

### half-year 数值 checkpoint

- `./model_checkpoints/fit_halfyear_num_with_meta_20260413_083428/checkpoint.pth`

### 控制文本 PT

- `./dataset/FIT_DualSG/pt/fit_dualsg_random_text.pt`
- `./dataset/FIT_DualSG/pt/fit_dualsg_filler_text.pt`

### 当前 half-year 核心脚本

- [fit_fusion_halfyear_direct.sh](/D:/zhangjing/project/Dualsg_refined/fit_fusion_halfyear_direct.sh)
- [fit_fusion_halfyear_direct_freeze.sh](/D:/zhangjing/project/Dualsg_refined/fit_fusion_halfyear_direct_freeze.sh)
- [fit_fusion_halfyear_direct_from_ckpt.sh](/D:/zhangjing/project/Dualsg_refined/fit_fusion_halfyear_direct_from_ckpt.sh)
- [fit_fusion_halfyear_legacy_direct_from_ckpt.sh](/D:/zhangjing/project/Dualsg_refined/fit_fusion_halfyear_legacy_direct_from_ckpt.sh)
- [fit_fusion_halfyear_residual.sh](/D:/zhangjing/project/Dualsg_refined/fit_fusion_halfyear_residual.sh)
- [fit_fusion_halfyear_residual_unfreeze.sh](/D:/zhangjing/project/Dualsg_refined/fit_fusion_halfyear_residual_unfreeze.sh)
- [fit_fusion_halfyear_legacy_residual_unfreeze.sh](/D:/zhangjing/project/Dualsg_refined/fit_fusion_halfyear_legacy_residual_unfreeze.sh)
- [fit_fusion_halfyear_direct_from_ckpt_disable_text.sh](/D:/zhangjing/project/Dualsg_refined/fit_fusion_halfyear_direct_from_ckpt_disable_text.sh)
- [fit_fusion_halfyear_residual_unfreeze_disable_text.sh](/D:/zhangjing/project/Dualsg_refined/fit_fusion_halfyear_residual_unfreeze_disable_text.sh)

## baseline

| setting | MAE | MSE | RMSE | MAPE | WAPE |
|---|---:|---:|---:|---:|---:|
| `fit_num_with_meta` | 0.085858 | 0.013315 | 0.115391 | 30.23% | 18.52% |

## 长文本：20 epoch

| setting | MAE | MSE | RMSE | MAPE | WAPE |
|---|---:|---:|---:|---:|---:|
| `direct + ckpt + freeze` | 0.087144 | 0.013424 | 0.115862 | 33.10% | 18.80% |
| `direct + scratch` | 0.105100 | 0.018546 | 0.136183 | 40.06% | 22.67% |
| `direct + ckpt + unfreeze` | 0.081830 | 0.011850 | 0.108857 | 31.00% | 17.65% |
| `residual + ckpt + freeze` | 0.085395 | 0.013151 | 0.114679 | 30.38% | 18.42% |
| `residual + ckpt + unfreeze` | 0.079863 | 0.011620 | 0.107797 | 28.83% | 17.23% |

## 长文本：100 epoch

| setting | MAE | MSE | RMSE | MAPE | WAPE |
|---|---:|---:|---:|---:|---:|
| `direct + ckpt + freeze` | 0.087139 | 0.013465 | 0.116037 | 32.16% | 18.80% |
| `direct + scratch` | 0.088332 | 0.013615 | 0.116683 | 33.22% | 19.06% |
| `direct + ckpt + unfreeze` | 0.077575 | 0.010672 | 0.103305 | 29.91% | 16.74% |
| `residual + ckpt + freeze` | 0.085395 | 0.013151 | 0.114679 | 30.38% | 18.42% |
| `residual + ckpt + unfreeze` | 0.075699 | 0.010511 | 0.102523 | 27.30% | 16.33% |

## 融合流关闭文本：20 epoch

这两组在当前实现里本质上是同一个实验，因为 `disable_text=True` 后，模型会直接返回数值预测，`text_mode` 不再参与。

| setting | MAE | MSE | RMSE | MAPE | WAPE |
|---|---:|---:|---:|---:|---:|
| `direct + ckpt + unfreeze + disable_text` | 0.080148 | 0.011753 | 0.108411 | 28.70% | 17.29% |
| `residual + ckpt + unfreeze + disable_text` | 0.080148 | 0.011753 | 0.108411 | 28.70% | 17.29% |

## 长文本 random text：20 epoch

| setting | MAE | MSE | RMSE | MAPE | WAPE |
|---|---:|---:|---:|---:|---:|
| `direct + ckpt + freeze` | 0.087233 | 0.013407 | 0.115790 | 33.08% | 18.82% |
| `direct + scratch` | 0.105321 | 0.018585 | 0.136328 | 40.39% | 22.72% |
| `direct + ckpt + unfreeze` | 0.081796 | 0.011842 | 0.108823 | 30.96% | 17.65% |
| `residual + ckpt + freeze` | 0.085365 | 0.013130 | 0.114586 | 30.54% | 18.42% |
| `residual + ckpt + unfreeze` | 0.079736 | 0.011595 | 0.107682 | 28.72% | 17.20% |

## 长文本 filler text：20 epoch

| setting | MAE | MSE | RMSE | MAPE | WAPE |
|---|---:|---:|---:|---:|---:|
| `direct + ckpt + freeze` | 0.087151 | 0.013434 | 0.115905 | 33.05% | 18.80% |
| `direct + scratch` | 0.105571 | 0.018621 | 0.136458 | 40.74% | 22.77% |
| `direct + ckpt + unfreeze` | 0.081904 | 0.011866 | 0.108930 | 30.96% | 17.67% |
| `residual + ckpt + freeze` | 0.085345 | 0.013124 | 0.114561 | 30.51% | 18.41% |
| `residual + ckpt + unfreeze` | 0.079818 | 0.011610 | 0.107749 | 28.81% | 17.22% |

## 结构化完整文本：20 epoch

| setting | MAE | MSE | RMSE | MAPE | WAPE |
|---|---:|---:|---:|---:|---:|
| `direct + ckpt + freeze` | 0.087139 | 0.013426 | 0.115870 | 33.10% | 18.80% |
| `direct + scratch` | 0.105514 | 0.018614 | 0.136433 | 40.66% | 22.76% |
| `direct + ckpt + unfreeze` | 0.081992 | 0.011889 | 0.109036 | 30.98% | 17.69% |
| `residual + ckpt + freeze` | 0.085368 | 0.013135 | 0.114607 | 30.39% | 18.42% |
| `residual + ckpt + unfreeze` | 0.079866 | 0.011611 | 0.107754 | 28.95% | 17.23% |

## Global Structural View：20 epoch

| setting | MAE | MSE | RMSE | MAPE | WAPE |
|---|---:|---:|---:|---:|---:|
| `direct + ckpt + freeze` | 0.087178 | 0.013395 | 0.115736 | 33.00% | 18.81% |
| `direct + scratch` | 0.105468 | 0.018596 | 0.136366 | 40.78% | 22.75% |
| `direct + ckpt + unfreeze` | 0.081979 | 0.011886 | 0.109022 | 30.98% | 17.69% |
| `residual + ckpt + freeze` | 0.085385 | 0.013118 | 0.114535 | 30.76% | 18.42% |
| `residual + ckpt + unfreeze` | 0.079822 | 0.011609 | 0.107744 | 28.82% | 17.22% |

## Dynamic Behavior View：20 epoch

| setting | MAE | MSE | RMSE | MAPE | WAPE |
|---|---:|---:|---:|---:|---:|
| `direct + ckpt + freeze` | 0.087238 | 0.013422 | 0.115852 | 33.46% | 18.82% |
| `direct + scratch` | 0.105344 | 0.018596 | 0.136367 | 40.46% | 22.73% |
| `direct + ckpt + unfreeze` | 0.081835 | 0.011848 | 0.108851 | 31.09% | 17.65% |
| `residual + ckpt + freeze` | 0.085409 | 0.013160 | 0.114715 | 30.31% | 18.43% |
| `residual + ckpt + unfreeze` | 0.079980 | 0.011674 | 0.108046 | 28.72% | 17.25% |

## Event-centric View：20 epoch

| setting | MAE | MSE | RMSE | MAPE | WAPE |
|---|---:|---:|---:|---:|---:|
| `direct + ckpt + freeze` | 0.087092 | 0.013399 | 0.115755 | 33.24% | 18.79% |
| `direct + scratch` | 0.105500 | 0.018616 | 0.136440 | 40.61% | 22.76% |
| `direct + ckpt + unfreeze` | 0.081763 | 0.011828 | 0.108757 | 31.14% | 17.64% |
| `residual + ckpt + freeze` | 0.085376 | 0.013124 | 0.114560 | 30.64% | 18.42% |
| `residual + ckpt + unfreeze` | 0.079809 | 0.011600 | 0.107703 | 28.90% | 17.22% |

## Semantic Caption View：20 epoch

| setting | MAE | MSE | RMSE | MAPE | WAPE |
|---|---:|---:|---:|---:|---:|
| `direct + ckpt + freeze` | 0.087139 | 0.013426 | 0.115870 | 33.10% | 18.80% |
| `direct + scratch` | 0.105514 | 0.018614 | 0.136433 | 40.66% | 22.76% |
| `direct + ckpt + unfreeze` | 0.081992 | 0.011889 | 0.109036 | 30.98% | 17.69% |
| `residual + ckpt + freeze` | 0.085368 | 0.013135 | 0.114607 | 30.39% | 18.42% |
| `residual + ckpt + unfreeze` | 0.079866 | 0.011611 | 0.107754 | 28.95% | 17.23% |

## 旧版最好结果

用户此前最好结果：

- `MAE = 0.071344`
- `MSE = 0.009260`
- `RMSE = 0.096228`
- `MAPE = 25.23%`
- `WAPE = 15.39%`

当前新版 fusion 还没有追上这一组。

## 当前最重要的结论

### 结论 1

当前最强路线仍然是：

- `residual + ckpt + unfreeze`

第二强路线是：

- `direct + ckpt + unfreeze`

### 结论 2

`random text`、`filler text` 和真实长文本的结果几乎一样。

这说明当前模型几乎没有利用文本语义本身。

### 结论 3

`disable_text` 结果与当前最好路线非常接近：

- `disable_text`
  - `MAE = 0.080148`
  - `MAPE = 28.70%`
- 真实长文本 `residual + ckpt + unfreeze`
  - `MAE = 0.079863`
  - `MAPE = 28.83%`

这进一步说明：

- 当前新版 fusion 的主要增益不是来自文本语义
- 更像来自：
  - `ckpt + unfreeze` 这个训练 recipe
  - `residual` 的硬数值 skip
  - 数值侧辅助特征本身

### 结论 4

`disable_text=True` 时，`direct` 和 `residual` 两个脚本会得到完全相同的结果，这是当前实现的正常行为，不是 bug。

原因是：

- 模型在 `disable_text=True` 时会直接返回数值预测
- `text_mode` 不再进入实际计算

### 结论 5

两个 freeze 版本：

- `direct + ckpt + freeze`
- `residual + ckpt + freeze`

现在都更像对照/消融，不是主路线。

## 当前问题如何表述

当前最准确的问题表述不是：

- “文本没设计好，所以效果不涨”

而是：

- “当前新版 fusion 结构允许模型几乎完全忽略文本，因此即便换成随机文本、无语义模板文本，结果也基本不变”

换句话说：

- 现在的瓶颈已经不是“再换一种文本 prompt”
- 而是“模型结构没有逼迫文本真正参与预测”

## 下一步建议

当前不建议继续堆更多文本版本。

更合理的方向只有两个：

1. 回退并恢复旧版 fusion 头，在当前清理后的训练体系下做严格 A/B
2. 继续重写新版 fusion，让文本必须真正参与预测，而不是可以被数值侧完全绕开

## Legacy 恢复诊断

当前进入新阶段：

- 不回滚整个项目
- 只把旧版 fusion 头恢复到当前清理后的训练框架里
- 用同一套 data loader、同一套 scaler、同一套训练脚本做严格 A/B

首轮只跑 half-year + 真实长文本：

- `legacy direct + ckpt + unfreeze`
- `legacy residual + ckpt + unfreeze`

对应脚本：

- [fit_fusion_halfyear_legacy_direct_from_ckpt.sh](/D:/zhangjing/project/Dualsg_refined/fit_fusion_halfyear_legacy_direct_from_ckpt.sh)
- [fit_fusion_halfyear_legacy_residual_unfreeze.sh](/D:/zhangjing/project/Dualsg_refined/fit_fusion_halfyear_legacy_residual_unfreeze.sh)

当前固定配置：

- `fusion_version=legacy`
- `caption_emb_path=./dataset/FIT_DualSG/pt/fit_dualsg_all.pt`
- `train_epochs=20`
- `patience=100`
- `adjust=0`
- `fusion_optimizer_mode=split`
- `fit_scaler_mode=train_only`

结果占位：

| setting | MAE | MSE | RMSE | MAPE | WAPE |
|---|---:|---:|---:|---:|---:|
| `legacy direct + ckpt + unfreeze` | 0.080265 | 0.011629 | 0.107840 | 29.75% | 17.32% |
| `legacy residual + ckpt + unfreeze` | 0.080123 | 0.011747 | 0.108384 | 28.68% | 17.28% |

当前第一轮 legacy 诊断结论：

- `legacy direct` 明显优于当前 `modern direct` 的同 recipe 结果
- `legacy residual` 与当前 `modern residual` 基本打平
- 两个 legacy 结果都明显没有追回用户很久之前那组 `MAE=0.071344 / MAPE=25.23%` 的旧版最好结果
- `legacy residual` 与 `disable_text` 也几乎一样，说明仅恢复旧 residual 头还不能证明文本语义真的重新参与了预测
