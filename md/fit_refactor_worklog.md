# FIT 重构工作记录

最后更新：2026-04-14

## 0411 阶段回顾

`0411` 已经完成的工作：

- FIT 数据层公共逻辑抽取
- `train_only` scaler 修正
- `fit_num_with_meta` 主干保留不动
- fusion 训练脚本整理
- random / filler / disable_text 控制实验
- structured / multi-view 文本实验
- legacy 头恢复诊断
- 脚本与结果快照归档到 `scripts_archive/fit_halfyear_0411/`

这一步的最大价值不是“得到最终模型”，而是把问题定位清楚：

- 0411 那套 active fusion 结构允许模型几乎绕开文本
- 因此真实文本、random、filler、disable_text 几乎没有本质差异

## 为什么 0411 不够

0411 已经坐实了三件事：

1. `residual + ckpt + unfreeze` 是最强 recipe
2. 但当前 strongest recipe 的增益并不能归因为文本语义
3. 恢复 legacy 头也没有直接把旧版最好结果拿回来

具体表现：

- `random text ≈ filler text ≈ real text ≈ disable_text`
- `legacy direct` 比 `modern direct` 稳
- `legacy residual` 与 `modern residual` 基本打平

因此问题不再是：

- “要不要再换一种 prompt”

而是：

- “如何从结构上减少文本被绕开的路径”

## 0414 主线改动

### 目标

`0414` 不再保留 `legacy/modern` 的运行时兼容。

当前活跃主线只保留：

- `direct`
- `residual`

目标分工：

- `direct`
  - 做成可解释的双流显式融合基线
- `residual`
  - 做成真正的文本纠偏器
  - 保证纠偏数学上必须经过文本

### 代码层变化

#### `run.py`

- 保留 active FIT fusion 需要的参数：
  - `text_mode`
  - `num_model_path`
  - `caption_emb_path`
  - `freeze_numerical`
  - `disable_text`
  - `fusion_optimizer_mode`
  - `num_feat_dim`
  - `fusion_hidden`
  - `fusion_dropout`
- 新增：
  - `text_hidden`
  - `residual_rank`
- 不再保留 `fusion_version` 作为 0414 活跃入口的一部分

#### `utils/print_args.py`

- 改成只打印 0414 主线真正用到的 FIT fusion 参数
- 不再把 legacy 专属字段当成当前活跃配置展示

#### `models/model_fit_fusion.py`

重写成单一路径：

- 没有 `modern / legacy` 运行时分支
- 数值 backbone 仍然复用 `Model_Fit_Num_With_Meta`
- 文本侧统一先走：
  - `caption_emb -> text_proj -> text_ctx`

`direct` 结构：

- `text_ctx -> direct_text_head -> y_text`
- `gate_input = [text_ctx, summary_ctx, y_num_ctx, hist_ctx]`
- `gate = sigmoid(direct_gate_head(...))`
- `y_final = (1 - gate) * y_num + gate * y_text`

`residual` 结构：

- 数值侧只生成 residual basis：
  - `residual_basis -> [B, pred_len, C, R]`
- 文本侧生成：
  - basis 系数 `text_coeff`
  - 幅度门控 `radius`
- 最终：
  - `delta_raw = sum_k basis_k * coeff_k`
  - `delta = radius * tanh(delta_raw)`
  - `y_final = y_num + delta`

关键约束：

- residual 里没有“纯数值分支直接输出 delta”的路径

#### `exp/exp_fit_fusion.py`

- 去掉了 `fusion_version` 的运行时日志
- 保持训练流程不变
- 继续保留 split / unified 优化器逻辑

## 0414 active 脚本

0414 根目录只保留两个 half-year fusion 入口：

- `fit_fusion_halfyear_direct_v2.sh`
- `fit_fusion_halfyear_residual_v2.sh`

旧的 half-year fusion 根目录脚本已经移出 active 区域。

完整 0411 快照仍在：

- `scripts_archive/fit_halfyear_0411/`

## 当前阶段判断

0414 首轮还没有开始看最终指标，当前阶段先看结构是否满足设计目标：

1. `direct` 仍有显式数值 skip
2. `residual` 的纠偏必须经过文本系数
3. `disable_text=True` 仍能退化成纯数值预测

后续第一轮只跑：

- `fit_fusion_halfyear_direct_v2.sh`
- `fit_fusion_halfyear_residual_v2.sh`

如果 `residual_v2` 更稳，再进入第二轮控制实验：

- `disable_text`
- `random`
- `filler`
