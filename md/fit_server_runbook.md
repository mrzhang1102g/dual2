# FIT 服务器执行手册

最后更新：2026-04-14

## 文档定位

这份文档同时承担两件事：

1. 记录 `0411` 阶段 FIT half-year 的关键结果
2. 记录 `0414` 新主线的当前执行入口

配套文档：

- [fit_halfyear_0411_manifest.md](/D:/zhangjing/project/Dualsg_refined/md/fit_halfyear_0411_manifest.md)
- [fit_refactor_worklog.md](/D:/zhangjing/project/Dualsg_refined/md/fit_refactor_worklog.md)
- [global_architecture.md](/D:/zhangjing/project/Dualsg_refined/md/global_architecture.md)

## 当前优先级

当前只聚焦 FIT half-year。

结果解读优先级：

1. `MAE`
2. `MAPE`
3. `WAPE`
4. `MSE`

## 0411 诊断阶段：已完成结果

### baseline

| setting | MAE | MSE | RMSE | MAPE | WAPE |
|---|---:|---:|---:|---:|---:|
| `fit_num_with_meta` | 0.085858 | 0.013315 | 0.115391 | 30.23% | 18.52% |

### 长文本：20 epoch

| setting | MAE | MSE | RMSE | MAPE | WAPE |
|---|---:|---:|---:|---:|---:|
| `direct + ckpt + freeze` | 0.087144 | 0.013424 | 0.115862 | 33.10% | 18.80% |
| `direct + scratch` | 0.105100 | 0.018546 | 0.136183 | 40.06% | 22.67% |
| `direct + ckpt + unfreeze` | 0.081830 | 0.011850 | 0.108857 | 31.00% | 17.65% |
| `residual + ckpt + freeze` | 0.085395 | 0.013151 | 0.114679 | 30.38% | 18.42% |
| `residual + ckpt + unfreeze` | 0.079863 | 0.011620 | 0.107797 | 28.83% | 17.23% |

### 长文本：100 epoch

| setting | MAE | MSE | RMSE | MAPE | WAPE |
|---|---:|---:|---:|---:|---:|
| `direct + ckpt + freeze` | 0.087139 | 0.013465 | 0.116037 | 32.16% | 18.80% |
| `direct + scratch` | 0.088332 | 0.013615 | 0.116683 | 33.22% | 19.06% |
| `direct + ckpt + unfreeze` | 0.077575 | 0.010672 | 0.103305 | 29.91% | 16.74% |
| `residual + ckpt + freeze` | 0.085395 | 0.013151 | 0.114679 | 30.38% | 18.42% |
| `residual + ckpt + unfreeze` | 0.075699 | 0.010511 | 0.102523 | 27.30% | 16.33% |

### random / filler / disable_text 关键控制：20 epoch

#### random text

| setting | MAE | MSE | RMSE | MAPE | WAPE |
|---|---:|---:|---:|---:|---:|
| `direct + ckpt + freeze` | 0.087233 | 0.013407 | 0.115790 | 33.08% | 18.82% |
| `direct + scratch` | 0.105321 | 0.018585 | 0.136328 | 40.39% | 22.72% |
| `direct + ckpt + unfreeze` | 0.081796 | 0.011842 | 0.108823 | 30.96% | 17.65% |
| `residual + ckpt + freeze` | 0.085365 | 0.013130 | 0.114586 | 30.54% | 18.42% |
| `residual + ckpt + unfreeze` | 0.079736 | 0.011595 | 0.107682 | 28.72% | 17.20% |

#### filler text

| setting | MAE | MSE | RMSE | MAPE | WAPE |
|---|---:|---:|---:|---:|---:|
| `direct + ckpt + freeze` | 0.087151 | 0.013434 | 0.115905 | 33.05% | 18.80% |
| `direct + scratch` | 0.105571 | 0.018621 | 0.136458 | 40.74% | 22.77% |
| `direct + ckpt + unfreeze` | 0.081904 | 0.011866 | 0.108930 | 30.96% | 17.67% |
| `residual + ckpt + freeze` | 0.085345 | 0.013124 | 0.114561 | 30.51% | 18.41% |
| `residual + ckpt + unfreeze` | 0.079818 | 0.011610 | 0.107749 | 28.81% | 17.22% |

#### disable_text

| setting | MAE | MSE | RMSE | MAPE | WAPE |
|---|---:|---:|---:|---:|---:|
| `direct + ckpt + unfreeze + disable_text` | 0.080148 | 0.011753 | 0.108411 | 28.70% | 17.29% |
| `residual + ckpt + unfreeze + disable_text` | 0.080148 | 0.011753 | 0.108411 | 28.70% | 17.29% |

### 结构化和多视角文本：20 epoch

#### 结构化完整文本

| setting | MAE | MSE | RMSE | MAPE | WAPE |
|---|---:|---:|---:|---:|---:|
| `direct + ckpt + freeze` | 0.087139 | 0.013426 | 0.115870 | 33.10% | 18.80% |
| `direct + scratch` | 0.105514 | 0.018614 | 0.136433 | 40.66% | 22.76% |
| `direct + ckpt + unfreeze` | 0.081992 | 0.011889 | 0.109036 | 30.98% | 17.69% |
| `residual + ckpt + freeze` | 0.085368 | 0.013135 | 0.114607 | 30.39% | 18.42% |
| `residual + ckpt + unfreeze` | 0.079866 | 0.011611 | 0.107754 | 28.95% | 17.23% |

#### Global Structural View

| setting | MAE | MSE | RMSE | MAPE | WAPE |
|---|---:|---:|---:|---:|---:|
| `direct + ckpt + freeze` | 0.087178 | 0.013395 | 0.115736 | 33.00% | 18.81% |
| `direct + scratch` | 0.105468 | 0.018596 | 0.136366 | 40.78% | 22.75% |
| `direct + ckpt + unfreeze` | 0.081979 | 0.011886 | 0.109022 | 30.98% | 17.69% |
| `residual + ckpt + freeze` | 0.085385 | 0.013118 | 0.114535 | 30.76% | 18.42% |
| `residual + ckpt + unfreeze` | 0.079822 | 0.011609 | 0.107744 | 28.82% | 17.22% |

#### Dynamic Behavior View

| setting | MAE | MSE | RMSE | MAPE | WAPE |
|---|---:|---:|---:|---:|---:|
| `direct + ckpt + freeze` | 0.087238 | 0.013422 | 0.115852 | 33.46% | 18.82% |
| `direct + scratch` | 0.105344 | 0.018596 | 0.136367 | 40.46% | 22.73% |
| `direct + ckpt + unfreeze` | 0.081835 | 0.011848 | 0.108851 | 31.09% | 17.65% |
| `residual + ckpt + freeze` | 0.085409 | 0.013160 | 0.114715 | 30.31% | 18.43% |
| `residual + ckpt + unfreeze` | 0.079980 | 0.011674 | 0.108046 | 28.72% | 17.25% |

#### Event-centric View

| setting | MAE | MSE | RMSE | MAPE | WAPE |
|---|---:|---:|---:|---:|---:|
| `direct + ckpt + freeze` | 0.087092 | 0.013399 | 0.115755 | 33.24% | 18.79% |
| `direct + scratch` | 0.105500 | 0.018616 | 0.136440 | 40.61% | 22.76% |
| `direct + ckpt + unfreeze` | 0.081763 | 0.011828 | 0.108757 | 31.14% | 17.64% |
| `residual + ckpt + freeze` | 0.085376 | 0.013124 | 0.114560 | 30.64% | 18.42% |
| `residual + ckpt + unfreeze` | 0.079809 | 0.011600 | 0.107703 | 28.90% | 17.22% |

#### Semantic Caption View

| setting | MAE | MSE | RMSE | MAPE | WAPE |
|---|---:|---:|---:|---:|---:|
| `direct + ckpt + freeze` | 0.087139 | 0.013426 | 0.115870 | 33.10% | 18.80% |
| `direct + scratch` | 0.105514 | 0.018614 | 0.136433 | 40.66% | 22.76% |
| `direct + ckpt + unfreeze` | 0.081992 | 0.011889 | 0.109036 | 30.98% | 17.69% |
| `residual + ckpt + freeze` | 0.085368 | 0.013135 | 0.114607 | 30.39% | 18.42% |
| `residual + ckpt + unfreeze` | 0.079866 | 0.011611 | 0.107754 | 28.95% | 17.23% |

### legacy 恢复诊断：20 epoch

用户很久之前的最好结果：

- `MAE = 0.071344`
- `MSE = 0.009260`
- `RMSE = 0.096228`
- `MAPE = 25.23%`
- `WAPE = 15.39%`

在当前清理后的训练框架下恢复旧头得到：

| setting | MAE | MSE | RMSE | MAPE | WAPE |
|---|---:|---:|---:|---:|---:|
| `legacy direct + ckpt + unfreeze` | 0.080265 | 0.011629 | 0.107840 | 29.75% | 17.32% |
| `legacy residual + ckpt + unfreeze` | 0.080123 | 0.011747 | 0.108384 | 28.68% | 17.28% |

## 0411 阶段的已知结论

最关键结论已经比较明确：

1. `residual + ckpt + unfreeze` 是 0411 阶段的最强 recipe。
2. `random text`、`filler text`、真实长文本、`disable_text` 结果几乎一样。
3. 因此 0411 的 `modern/legacy` 诊断都没有坐实“文本语义真正贡献了提升”。
4. 主要增益更像来自：
   - `num_ckpt + unfreeze`
   - residual 的硬数值 skip
   - 数值侧辅助特征

## 0414 v2 首轮实验

当前 `0414` 根目录只保留两个 active half-year fusion 脚本：

- [fit_fusion_halfyear_direct_v2.sh](/D:/zhangjing/project/Dualsg_refined/fit_fusion_halfyear_direct_v2.sh)
- [fit_fusion_halfyear_residual_v2.sh](/D:/zhangjing/project/Dualsg_refined/fit_fusion_halfyear_residual_v2.sh)

固定配置：

- `real long text = ./dataset/FIT_DualSG/pt/fit_dualsg_all.pt`
- `num_model_path = ./model_checkpoints/fit_halfyear_num_with_meta_20260413_083428/checkpoint.pth`
- `unfreeze numerical`
- `train_epochs = 20`
- `patience = 100`
- `adjust = 0`
- `fusion_optimizer_mode = split`
- `fit_scaler_mode = train_only`

结果占位：

| setting | MAE | MSE | RMSE | MAPE | WAPE |
|---|---:|---:|---:|---:|---:|
| `direct_v2 + ckpt + unfreeze` | pending | pending | pending | pending | pending |
| `residual_v2 + ckpt + unfreeze` | pending | pending | pending | pending | pending |

## 0414 首轮之后的判据

第一轮不急着证明文本语义，只先看两件事：

1. `residual_v2` 是否稳定优于 `direct_v2`
2. 结构上是否已经具备“文本必须参与预测”的前提

如果 `residual_v2` 表现更稳：

- 第二轮再补
  - `disable_text`
  - `random`
  - `filler`

如果 `residual_v2` 仍然和纯数值对照差不多：

- 说明还需要继续做结构层重写
- 不再优先堆更多文本视角
