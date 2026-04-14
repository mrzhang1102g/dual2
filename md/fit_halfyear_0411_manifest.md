# FIT Half-Year 0411 Manifest

最后更新：2026-04-14

## 文档定位

这份文档只服务于 `0411` 阶段的 FIT half-year 诊断快照。

它回答两件事：

1. 这一阶段到底跑了哪些实验。
2. 这些实验对应的 `.sh` 脚本现在保存在哪里。

对应脚本归档目录：

- `scripts_archive/fit_halfyear_0411/`

## 统一约定

统一使用的数据：

- `./dataset/FIT_DualSG/fit_dualsg_all.json`

统一使用的 half-year 数值 checkpoint：

- `./model_checkpoints/fit_halfyear_num_with_meta_20260413_083428/checkpoint.pth`

统一训练约定：

- `batch_size = 200`
- `patience = 100`
- `adjust = 0`
- `fusion_optimizer_mode = split`
- `fit_scaler_mode = train_only`

## 实验分组与脚本目录

| 分组 | 文本 PT | epoch | 脚本目录 |
|---|---|---:|---|
| baseline | 无文本 | 20 | `scripts_archive/fit_halfyear_0411/01_baseline/` |
| modern long text | `./dataset/FIT_DualSG/pt/fit_dualsg_all.pt` | 20 | `scripts_archive/fit_halfyear_0411/02_modern_long_text_20/` |
| modern long text | `./dataset/FIT_DualSG/pt/fit_dualsg_all.pt` | 100 | `scripts_archive/fit_halfyear_0411/03_modern_long_text_100/` |
| modern random text | `./dataset/FIT_DualSG/pt/fit_dualsg_random_text.pt` | 20 | `scripts_archive/fit_halfyear_0411/04_modern_random_text_20/` |
| modern filler text | `./dataset/FIT_DualSG/pt/fit_dualsg_filler_text.pt` | 20 | `scripts_archive/fit_halfyear_0411/05_modern_filler_text_20/` |
| modern structured full | `./dataset/FIT_DualSG/pt/fit_dualsg_structured.pt` | 20 | `scripts_archive/fit_halfyear_0411/06_modern_structured_full_20/` |
| modern global structural | `./dataset/FIT_DualSG/pt/fit_dualsg_global_structural.pt` | 20 | `scripts_archive/fit_halfyear_0411/07_modern_global_structural_20/` |
| modern dynamic behavior | `./dataset/FIT_DualSG/pt/fit_dualsg_dynamic_behavior.pt` | 20 | `scripts_archive/fit_halfyear_0411/08_modern_dynamic_behavior_20/` |
| modern event centric | `./dataset/FIT_DualSG/pt/fit_dualsg_event_centric.pt` | 20 | `scripts_archive/fit_halfyear_0411/09_modern_event_centric_20/` |
| modern semantic caption | `./dataset/FIT_DualSG/pt/fit_dualsg_semantic_caption.pt` | 20 | `scripts_archive/fit_halfyear_0411/10_modern_semantic_caption_20/` |
| disable_text controls | `./dataset/FIT_DualSG/pt/fit_dualsg_random_text.pt` | 20 | `scripts_archive/fit_halfyear_0411/11_controls_disable_text_20/` |
| legacy long text | `./dataset/FIT_DualSG/pt/fit_dualsg_all.pt` | 20 | `scripts_archive/fit_halfyear_0411/12_legacy_long_text_20/` |

## 每组默认脚本矩阵

除 baseline、disable_text、legacy 外，其余 modern 分组都归档了这 5 个脚本：

- `fit_fusion_halfyear_direct.sh`
- `fit_fusion_halfyear_direct_freeze.sh`
- `fit_fusion_halfyear_direct_from_ckpt.sh`
- `fit_fusion_halfyear_residual.sh`
- `fit_fusion_halfyear_residual_unfreeze.sh`

disable_text 只保留两条关键对照：

- `fit_fusion_halfyear_direct_from_ckpt_disable_text.sh`
- `fit_fusion_halfyear_residual_unfreeze_disable_text.sh`

legacy 只保留两条诊断主线：

- `fit_fusion_halfyear_legacy_direct_from_ckpt.sh`
- `fit_fusion_halfyear_legacy_residual_unfreeze.sh`

## 当前阶段结论

这一阶段的主要作用不是产出最终模型，而是完成诊断：

- current modern 结构已经被 random / filler / disable_text 基本坐实为“可绕开文本”
- legacy direct 比 modern direct 更稳
- legacy residual 没有把结果拉回用户最早那组最好结果
- 因此当前 `0411` 更适合作为完整诊断快照保存

下一步新的结构改造，应该在新的分支继续推进，而不是继续污染这份快照。
