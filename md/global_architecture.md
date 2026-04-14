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

## 三代 Fusion 演进

这里把我们当前实际讨论过的三代 FIT fusion 结构统一整理一下。

为了避免概念混乱，下面的“三代”按结构思路划分：

1. 第一代：历史旧版浅融合
   当前在 `0411` 归档里对应 `legacy`
2. 第二代：0411 shared-context / latent modulation
   当前在 `0411` 归档里对应 `modern`
3. 第三代：0414 v2
   当前 `0414` 活跃主线

### 第一代：历史旧版浅融合

#### direct

结构：

- `caption_emb -> y_text`
- `y_final = (1 - w) * y_num + w * y_text`

特点：

- 文本分支单独预测一条未来序列
- 数值主预测 `y_num` 直接进入最终输出
- `w` 是显式融合权重，可以是固定值，也可以是可学习参数

优点：

- 数值流有强硬保底路径
- 训练更稳
- 文本确实有一条独立输出通路，不容易被完全埋掉

缺点：

- 文本参与方式偏浅
- 文本分支和数值分支只在最后一步相加
- 交互能力弱，更多像“外挂一个文本头”

#### residual

结构：

- `fusion_in = [caption_emb, phi(y_num.detach()), vol]`
- 预测 `delta_y` 和 `gain`
- `y_final = y_num + gain * delta_y`

特点：

- 数值主预测是核心
- 文本承担纠偏角色
- `phi(y_num)` 走 `detach`，也就是纠偏头看的数值摘要是固定的

优点：

- 语义清楚，适合讲“文本做纠偏”
- 相比 direct 更稳
- 不容易把强数值 baseline 完全带偏

缺点：

- 文本使用仍偏浅
- 纠偏依赖手工设计的输入拼接
- `gain / delta` 的控制更像经验型补丁

### 第二代：0411 shared-context / latent modulation

#### direct

结构：

- `caption_emb -> text_adapter -> text_ctx`
- `text_ctx -> gamma / beta`
- `gamma / beta` 调制 `encoded_tokens`
- 再与：
  - 文本上下文
  - 调制后的数值摘要
  - `y_num` 摘要
  - 历史统计
  拼成共享上下文
- `shared_context -> direct_head -> y_final`

特点：

- 文本更早介入数值 backbone 的中间特征
- final output 不再是显式 `y_num + something`
- 而是直接从共享上下文重新生成整条预测

优点：

- 表达能力更强
- 文本和数值的交互更深
- 理论上更容易捕捉复杂条件依赖

缺点：

- `y_num` 没有显式硬 skip
- 文本可以被共享上下文里的数值特征淹没
- 实验上已经被 `random / filler / disable_text` 坐实为“可以绕开文本”

#### residual

结构：

- 同样先构造 shared context
- `shared_context -> residual_head -> raw_delta`
- `shared_context -> radius_head -> radius`
- `delta = radius * tanh(raw_delta)`
- `y_final = y_num + delta`

特点：

- 形式上还是 residual
- 但 `delta` 已经主要由 shared context 决定
- shared context 里包含大量数值侧信息

优点：

- 比 direct 稳
- 比第一代 residual 有更强的交互表达能力
- 可以利用 latent、summary、history 等更多上下文

缺点：

- residual 里仍然存在“数值信息过强，文本变弱信号”的问题
- `delta` 仍可能主要由数值侧决定
- 这也是为什么第二代 residual 虽然比 direct 好，但仍然没有坐实文本语义贡献

### 第三代：0414 v2

#### direct

结构：

- 数值流：`y_num`
- 文本流：`caption_emb -> text_proj -> text_ctx -> y_text`
- `gate_input = [text_ctx, summary_ctx, y_num_ctx, hist_ctx]`
- `gate = sigmoid(gate_head(gate_input))`
- `y_final = (1 - gate) * y_num + gate * y_text`

和前两代的区别：

- 保留了第一代 direct 的显式数值 skip
- 但 gate 不是旧版的单一固定权重，而是逐步预测的动态 gate
- 文本侧不再只是一个很浅的线性头，也会结合数值摘要来决定融合比例

当前信号：

- `direct_v2` 已经是目前最强的 direct 版本
- 说明“恢复显式数值 skip”这个方向是对的

#### residual

结构：

- 数值侧只生成 `residual_basis`
- 文本侧生成：
  - `text_coeff`
  - `radius`
- `delta_raw = sum_k residual_basis[..., k] * text_coeff[..., k]`
- `delta = radius * tanh(delta_raw)`
- `y_final = y_num + delta`

和前两代的区别：

- 比第一代 residual 更系统，不再只是把几个特征手工拼起来做一个小头
- 比第二代 residual 更严格，因为它不允许“纯数值分支直接输出 delta”
- 文本必须通过系数去选择、组合数值侧 basis

当前信号：

- `residual_v2` 目前在 `MAPE` 上更有优势
- 但还没有在 `MAE/WAPE` 上明显压过 `direct_v2`
- 也就是说，这版 residual 的结构约束更合理，但训练效果还没有完全兑现

### 第三代当前实验信号

到目前为止，第三代已经完成了以下 half-year 控制：

- `real long text`
- `disable_text`
- `filler text`
- `random text`

当前最重要的观察不是“哪组数值更低一点”，而是这些控制之间的相对关系。

#### direct_v2

目前已知结果：

- real:
  - `MAE = 0.079404`
  - `MAPE = 29.18%`
  - `WAPE = 17.13%`
- disable:
  - `MAE = 0.080189`
  - `MAPE = 28.75%`
  - `WAPE = 17.30%`
- filler:
  - `MAE = 0.079494`
  - `MAPE = 29.26%`
  - `WAPE = 17.15%`
- random:
  - `MAE = 0.079525`
  - `MAPE = 29.33%`
  - `WAPE = 17.16%`

解读：

- `direct_v2` 已经明显比第二代 direct 稳
- 但 real / filler / random 非常接近
- 因而目前还不能说 `direct_v2` 已经有效利用了文本语义
- 更准确地说，它更像是“带文本分支的稳定双流融合”，而不是“文本语义被明确用起来了”

#### residual_v2

目前已知结果：

- real:
  - `MAE = 0.079863`
  - `MAPE = 28.60%`
  - `WAPE = 17.23%`
- disable:
  - `MAE = 0.080189`
  - `MAPE = 28.75%`
  - `WAPE = 17.30%`
- filler:
  - `MAE = 0.079827`
  - `MAPE = 28.72%`
  - `WAPE = 17.22%`
- random:
  - `MAE = 0.079860`
  - `MAPE = 28.65%`
  - `WAPE = 17.23%`

解读：

- `residual_v2` 在结构上已经比第二代更接近真正的文本纠偏器
- 但结果上 real / filler / random / disable_text 仍然几乎重合
- 这意味着“文本必须出现在公式里”还不等于“文本内容差异被模型真正利用”

因此第三代当前最准确的架构结论是：

- 它比第二代更稳、更干净
- 但仍然没有形成我们希望看到的语义排序：
  - `real > filler > random > disable`
- 所以下一步重点不应该是继续堆 prompt，而应该是继续改 residual 结构，让文本内容真正决定纠偏策略

## 三代 direct / residual 的本质区别

一句话概括：

- 第一代：文本在输出端浅融合，稳定，但浅
- 第二代：文本更早进入 latent，表达更强，但容易被绕开
- 第三代：尝试把第一代的稳定性和第二代的结构表达结合起来

如果拆成 direct / residual 两条线看：

### direct 的三代变化

第一代：

- 强显式 skip
- 文本单独预测
- 浅融合

第二代：

- 去掉显式 skip
- 改成 shared-context 重建整条序列
- 更灵活，但最容易失稳和绕开文本

第三代：

- 把显式 skip 加回来
- 同时把融合权重升级成 step-wise dynamic gate

### residual 的三代变化

第一代：

- 文本做一个小纠偏头
- 稳定、直观，但表达偏浅

第二代：

- 纠偏建立在 shared context 上
- 表达变强，但仍然可能主要依赖数值侧

第三代：

- 数值只提供 basis
- 文本必须提供组合系数和幅度控制
- 从结构上更强调“文本参与纠偏”

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
