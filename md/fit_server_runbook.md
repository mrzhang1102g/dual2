# FIT 服务器执行手册

最后更新：2026-04-16

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

## 0414 v2 诊断结果

`0414 v2` 已经完成首轮和控制实验，当前不再继续把它作为 active 结构往前推。

它承担的作用是：

- 验证第三代结构比 0411 第二代更稳
- 验证 direct / residual 在 `disable_text / filler / random` 下的行为
- 判断文本语义是否已经开始真正起作用

对应结果已经在上面完整记录，这里只保留当前结论：

- `direct_v2` 已经是当前最强的 direct 版本
- `residual_v2` 在 `MAPE` 上更好，但 `MAE / WAPE` 没有压过 `direct_v2`
- `real / filler / random / disable_text` 的差距仍然太小
- 因此下一步不是继续拉长训练，而是继续改结构

## 0414 v3 结果回顾

`v3` 阶段 active half-year fusion 脚本是：

- [fit_fusion_halfyear_direct_v3.sh](/D:/zhangjing/project/Dualsg_refined/fit_fusion_halfyear_direct_v3.sh)
- [fit_fusion_halfyear_residual_v3.sh](/D:/zhangjing/project/Dualsg_refined/fit_fusion_halfyear_residual_v3.sh)

上一轮 `v2` 脚本仍保留在根目录，仅用于结果回看：

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
- `text_hidden = 128`
- `num_experts = 4`
- `fit_fusion_halfyear_direct_v3.sh` 使用 `text_mode=direct`
- `fit_fusion_halfyear_residual_v3.sh` 使用 `text_mode=residual`

`v3` 设计目标：

- `direct_v3`
  - 继续保留显式数值 skip
  - 文本侧不再只出一个预测头，而是输出 4 个候选文本预测，再由文本路由混合
- `residual_v3`
  - 数值侧输出 4 个候选纠偏 expert
  - 文本侧输出逐步路由权重和纠偏幅度
  - 最终纠偏必须经过文本路由，不允许纯数值直接给出最终 `delta`

第一轮 `v3` 已完成：

1. `fit_fusion_halfyear_direct_v3.sh`
2. `fit_fusion_halfyear_residual_v3.sh`

当前结果：

| setting | MAE | MSE | RMSE | MAPE | WAPE |
|---|---:|---:|---:|---:|---:|
| `direct_v3 + ckpt + unfreeze` | 0.079424 | 0.011435 | 0.106934 | 29.03% | 17.13% |
| `residual_v3 + ckpt + unfreeze` | 0.079695 | 0.011593 | 0.107672 | 28.79% | 17.19% |

当前解读：

- `direct_v3` 和 `v2 direct` 基本持平，`MAE / WAPE` 仍然是当前 direct 路线里最稳的一档
- `residual_v3` 相比 `v2 residual`，在 `MAE / WAPE` 上有小幅改善，但 `MAPE` 没有压过 `v2 residual`
- 也就是说，4 个候选 expert 的 routing 让 residual 路线有一点进步，但进步还不够大
- 当前仍然不能直接说 `residual_v3` 已经赢过 `direct_v3`
- `v3` 的信息增量已经足够，不再继续补完整控制实验

## 0414 v4 第一轮入口

当前 `0414` 根目录 active half-year fusion 脚本更新为：

- [fit_fusion_halfyear_direct_v3.sh](/D:/zhangjing/project/Dualsg_refined/fit_fusion_halfyear_direct_v3.sh)
- [fit_fusion_halfyear_residual_v4.sh](/D:/zhangjing/project/Dualsg_refined/fit_fusion_halfyear_residual_v4.sh)

固定配置：

- `real long text = ./dataset/FIT_DualSG/pt/fit_dualsg_all.pt`
- `num_model_path = ./model_checkpoints/fit_halfyear_num_with_meta_20260413_083428/checkpoint.pth`
- `unfreeze numerical`
- `train_epochs = 20`
- `patience = 100`
- `adjust = 0`
- `fusion_optimizer_mode = split`
- `fit_scaler_mode = train_only`
- `text_hidden = 128`
- `num_experts = 4`
- `trend_segments = 4`

`v4` 设计目标：

- `direct_v3`
  - 继续作为稳定 direct 基线
- `residual_v4`
  - 借鉴 DualSG，不再做高频 expert routing residual
  - 改成趋势级、分段常数的 forecast-space correction
  - 文本先读取数值趋势上下文，再输出低频纠偏

当前结果先留空，待服务器首轮返回后再补：

| setting | MAE | MSE | RMSE | MAPE | WAPE |
|---|---:|---:|---:|---:|---:|
| `direct_v3 + ckpt + unfreeze` | pending | pending | pending | pending | pending |
| `residual_v4 + ckpt + unfreeze` | pending | pending | pending | pending | pending |

