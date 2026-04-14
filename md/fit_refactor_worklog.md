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

### v2 目标

`0414` 不再保留 `legacy/modern` 的运行时兼容。

当时活跃主线只保留：

- `direct`
- `residual`

目标分工：

- `direct`
  - 做成可解释的双流显式融合基线
- `residual`
  - 做成真正的文本纠偏器
  - 保证纠偏数学上必须经过文本

### v2 代码层变化

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

## 0414 v2 active 脚本

0414 v2 阶段根目录只保留两个 half-year fusion 入口：

- `fit_fusion_halfyear_direct_v2.sh`
- `fit_fusion_halfyear_residual_v2.sh`

旧的 half-year fusion 根目录脚本已经移出 active 区域。

完整 0411 快照仍在：

- `scripts_archive/fit_halfyear_0411/`

## 0414 v2 阶段判断

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

## 0414 v2 首轮结果

用户已经完成：

- `fit_fusion_halfyear_direct_v2.sh`
- `fit_fusion_halfyear_residual_v2.sh`

结果：

- `direct_v2 + ckpt + unfreeze`
  - `MAE = 0.079404`
  - `MSE = 0.011418`
  - `RMSE = 0.106854`
  - `MAPE = 29.18%`
  - `WAPE = 17.13%`
- `residual_v2 + ckpt + unfreeze`
  - `MAE = 0.079863`
  - `MSE = 0.011669`
  - `RMSE = 0.108021`
  - `MAPE = 28.60%`
  - `WAPE = 17.23%`

当前解读：

- `direct_v2` 已经是目前最强的 direct 版本
- 说明把显式数值 skip 加回来是有效的
- `residual_v2` 在 `MAPE` 上更好，说明它在比例误差层面更有潜力
- 但 `residual_v2` 目前还没有在 `MAE / WAPE` 上压过 `direct_v2`

因此当前最合理的下一步不是立刻拉 `100 epoch`，而是先补文本控制实验。

## 0414 v2 控制实验结果

用户随后补完了三类控制：

- `disable_text`
- `random text`
- `filler text`

结果如下。

### `disable_text`

- `direct_v2 + ckpt + unfreeze + disable_text`
  - `MAE = 0.080189`
  - `MSE = 0.011770`
  - `RMSE = 0.108491`
  - `MAPE = 28.75%`
  - `WAPE = 17.30%`
- `residual_v2 + ckpt + unfreeze + disable_text`
  - `MAE = 0.080189`
  - `MSE = 0.011770`
  - `RMSE = 0.108491`
  - `MAPE = 28.75%`
  - `WAPE = 17.30%`

这两组完全一致，符合当前实现语义：

- 关闭文本后，两种模式都严格退化成纯数值预测

### `filler text`

- `direct_v2 + ckpt + unfreeze + filler`
  - `MAE = 0.079494`
  - `MSE = 0.011439`
  - `RMSE = 0.106952`
  - `MAPE = 29.26%`
  - `WAPE = 17.15%`
- `residual_v2 + ckpt + unfreeze + filler`
  - `MAE = 0.079827`
  - `MSE = 0.011651`
  - `RMSE = 0.107941`
  - `MAPE = 28.72%`
  - `WAPE = 17.22%`

### `random text`

- `direct_v2 + ckpt + unfreeze + random`
  - `MAE = 0.079525`
  - `MSE = 0.011442`
  - `RMSE = 0.106967`
  - `MAPE = 29.33%`
  - `WAPE = 17.16%`
- `residual_v2 + ckpt + unfreeze + random`
  - `MAE = 0.079860`
  - `MSE = 0.011664`
  - `RMSE = 0.108002`
  - `MAPE = 28.65%`
  - `WAPE = 17.23%`

## 当前阶段结论

0414 v2 到这里已经可以得出比较明确的判断。

### 结构层面

- 第三代结构比第二代更稳
- 尤其是 `direct_v2`，已经明显强于 0411 的 direct 系列
- 这说明“保留显式数值 skip，再做动态融合”是正确方向

### 语义层面

- `direct_v2` 的真实文本、filler、random 都非常接近，而且都只比 `disable_text` 略好一点
- `residual_v2` 的真实文本、filler、random、disable_text 几乎完全重合

这意味着：

- 第三代结构虽然比第二代干净、稳定
- 但文本语义贡献仍然没有被坐实

也就是说，当前还不能支持这样的叙事：

- `real text > filler text > random text > disable_text`

### 对两条主线的判断

- `direct_v2`
  - 当前最强的 direct 版本
  - 可能已经从“有文本分支”这件事本身获得了轻微辅助增益
  - 但还没有形成清晰的语义排序
- `residual_v2`
  - 结构上更接近我们真正想要的“文本纠偏器”
  - 但文本系数对 real / filler / random 的区分仍然太弱
  - 因而还没有把结构约束兑现成语义增益

## 为什么还要继续改到 v3

到 `disable_text / random / filler` 跑完为止，v2 已经给出足够清晰的信号：

- `direct_v2` 变稳了
- `residual_v2` 结构上更合理了
- 但文本语义仍然没有真正被坐实

因此当前最合理的方向不再是：

- 立刻把 v2 拉到更长训练轮数
- 继续堆更多 prompt / 更多文本版本

而是进入 `v3`：

- 继续做结构重写
- 重点继续改 `residual`
- 同时顺手把 `direct` 升级成轻量多候选文本预测
- 让“文本内容不同”必须导致不同的纠偏策略

更具体地说，下一轮要重点解决：

1. 为什么 `text_coeff` 对 real / filler / random 的区分仍然这么弱
2. 为什么 `residual_v2` 在结构上要求文本参与，但结果上仍然接近 `disable_text`
3. 是否需要把 `direct_v2` 的稳定性优点进一步借给新的 residual 设计

## 0414 v3 设计

### 命名约定

从现在开始，按当前分支内的实验版本命名：

- `0411` 阶段 active 结构记为 `v1`
- `0414` 的第一轮重写记为 `v2`
- 当前即将运行的 expert-routing 重写记为 `v3`

用户更早的历史最好结果仍然保留为“历史旧版最好结果”，不并入这个 `v1 / v2 / v3` 序列。

### v3 direct

`direct_v3` 继续保留：

- 显式数值 skip
- `y_final = (1 - gate) * y_num + gate * y_text`

但文本流不再只输出一个 `y_text`，而是：

- 先输出 4 个候选文本预测
- 再由文本路由权重把这 4 个候选混合成最终 `y_text`

目标是：

- 在不破坏稳定性的前提下
- 让文本侧至少具备几种不同的预测模式

### v3 residual

`residual_v3` 仍保持：

- `y_final = y_num + delta`

但 `delta` 的产生方式进一步收紧为：

- 数值侧只输出 4 个候选纠偏 expert
- 文本侧只负责：
  - 给这 4 个 expert 做逐步路由
  - 给纠偏幅度做逐步 gate

最终：

- `delta_mix = sum_k route_k * delta_k`
- `delta = radius * tanh(delta_mix)`

这里最关键的是：

- 文本不再只是给一个连续系数向量
- 而是更明确地“选择纠偏模式”
- 这样理论上更容易把 real / filler / random 拉开

### v3 的额外工程目标

除了结构本身，v3 还顺手解决一个实验可解释性问题：

- 不再打印模型总参数量
- 改成只打印“本次 optimizer 里真正会更新的参数量”

这样 direct / residual 两条路的日志会更符合真实训练状态。

### 当前 v3 第一轮

当前已经新增：

- `fit_fusion_halfyear_direct_v3.sh`
- `fit_fusion_halfyear_residual_v3.sh`

固定设置：

- `half-year`
- `real long text`
- `ckpt + unfreeze`
- `train_epochs = 20`
- `num_experts = 4`

这一轮的目的不是立刻追平历史最好结果，而是先回答：

1. `residual_v3` 能不能重新成为明显强于 `direct_v3` 的主方法
2. 4 个候选 expert 的路由机制，能不能让文本控制实验开始出现差异
