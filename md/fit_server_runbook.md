﻿﻿# FIT 服务器执行手册

最后更新：2026-04-14

## 当前重点

当前只聚焦 FIT half-year。

当前排序优先看：

1. `MAE`
2. `MAPE`
3. `WAPE`
4. `MSE`

## 当前实验阶段

已经完成：

- 数值 baseline
- 长文本 20 epoch
- 长文本 100 epoch
- 结构化完整文本 20 epoch
- Global Structural View 20 epoch
- Dynamic Behavior View 20 epoch
- Event-centric View 20 epoch
- Semantic Caption View 20 epoch

当前正在跑：

- 暂无

下一轮准备跑：

- 建议先做 `disable_text` 对照

## 当前使用的关键文件

### 数值数据

- `./dataset/FIT_DualSG/fit_dualsg_all.json`

### 当前正在跑的文本 PT

- `./dataset/FIT_DualSG/pt/fit_dualsg_random_text.pt`

### 下一轮 filler 文本 PT

- `./dataset/FIT_DualSG/pt/fit_dualsg_filler_text.pt`

### half-year 数值 checkpoint

- `./model_checkpoints/fit_halfyear_num_with_meta_20260413_083428/checkpoint.pth`

### 当前 5 个 half-year 脚本

- [fit_fusion_halfyear_direct.sh](/D:/zhangjing/project/Dualsg_refined/fit_fusion_halfyear_direct.sh)
- [fit_fusion_halfyear_direct_freeze.sh](/D:/zhangjing/project/Dualsg_refined/fit_fusion_halfyear_direct_freeze.sh)
- [fit_fusion_halfyear_direct_from_ckpt.sh](/D:/zhangjing/project/Dualsg_refined/fit_fusion_halfyear_direct_from_ckpt.sh)
- [fit_fusion_halfyear_residual.sh](/D:/zhangjing/project/Dualsg_refined/fit_fusion_halfyear_residual.sh)
- [fit_fusion_halfyear_residual_unfreeze.sh](/D:/zhangjing/project/Dualsg_refined/fit_fusion_halfyear_residual_unfreeze.sh)

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
| `residual + ckpt + unfreeze` | 0.079282 | 0.011609 | 0.107744 | 28.82% | 17.22% |

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

## 当前已确认结论

### 结论 1

排序非常稳定，几乎所有文本版本都是：

1. `residual + ckpt + unfreeze`
2. `direct + ckpt + unfreeze`
3. `residual + ckpt + freeze`
4. `direct + ckpt + freeze`
5. `direct + scratch`

### 结论 2

当前真正值得继续看的主路线是：

- `residual + ckpt + unfreeze`
- `direct + ckpt + unfreeze`

### 结论 3

两个 freeze 版本更像对照组：

- `residual + ckpt + freeze`
- `direct + ckpt + freeze`

### 结论 4

结构化、多视角文本目前都没有带来跨档提升，数值只在小数点后三位附近波动。

### 结论 5

`structured full text` 和 `Semantic Caption View` 的结果完全一样，后面要检查它们是否实际使用了不同的 `pt`。

### 结论 6

当前新版 fusion 仍然明显没有追上旧版最好结果：

- 旧版最好：
  - `MAE = 0.071344`
  - `MSE = 0.009260`
  - `RMSE = 0.096228`
  - `MAPE = 25.23%`
  - `WAPE = 15.39%`

### 结论 7

`random text` 和 `filler text` 与真实长文本结果几乎一致。

这说明当前模型的提升基本不能归因于文本语义本身。

### 结论 8

当前结果更像是：

- `ckpt + unfreeze` 这一训练 recipe 本身带来了提升
- `residual` 的硬数值 skip 比 `direct` 更稳
- 文本内容是否真实、随机、无语义，占比都非常小

## 当前对照实验说明

已经完成两类控制文本：

- random text
- filler text

生成脚本：

- [generate_random_text_pt.py](/D:/zhangjing/project/Dualsg_refined/dataset/FIT_DualSG/scripts/generate_random_text_pt.py)
- [generate_filler_text_pt.py](/D:/zhangjing/project/Dualsg_refined/dataset/FIT_DualSG/scripts/generate_filler_text_pt.py)

说明：

- 两个脚本都基于 `fit_dualsg_all.json`
- 不需要额外新 json
- 只生成新的 `pt`
- 样本数和顺序都与 `fit_dualsg_all.json` 保持一致

## 下一步

当前最高价值、且不需要改模型代码的实验是：

1. `direct + ckpt + unfreeze + disable_text`
2. `residual + ckpt + unfreeze + disable_text`

对应脚本：

- [fit_fusion_halfyear_direct_from_ckpt_disable_text.sh](/D:/zhangjing/project/Dualsg_refined/fit_fusion_halfyear_direct_from_ckpt_disable_text.sh)
- [fit_fusion_halfyear_residual_unfreeze_disable_text.sh](/D:/zhangjing/project/Dualsg_refined/fit_fusion_halfyear_residual_unfreeze_disable_text.sh)

如果这两组结果仍然接近当前最好结果，就可以基本确认：

- 当前新版 fusion 的主要增益来自数值流继续训练和数值辅助纠偏
- 不是来自文本语义

不建议把 5 组都跑成 `disable_text`，因为：

- `direct/residual + ckpt + freeze + disable_text` 基本只会退化成数值 checkpoint 自身
- `direct + scratch + disable_text` 更像另一种数值流重训，对定位“文本到底有没有作用”价值不高
