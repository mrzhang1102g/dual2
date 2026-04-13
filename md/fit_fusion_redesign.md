# FIT Fusion Redesign

最后更新：2026-04-12

## 1. 文档定位

这份文档记录 FIT fusion 这次重写时采用的设计原则。

它回答三个问题：
- 为什么旧版浅融合不够好
- 为什么这次要删掉 `beta_delta / force_gain / llm_dim`
- 新版 direct / residual 的语义分别是什么

## 2. 旧版问题

旧版 FIT fusion 的主要问题不是“有没有文本分支”，而是“文本分支太浅”。

典型问题：
- 数值 backbone 先独立给出 `y_num`
- 文本侧再接一个很小的 head
- 最后做很浅的加权或纠偏

这样容易退化成：
- direct 只是另一个很弱的小预测器
- residual 只是一个很弱的小修补器

文本并没有真正参与数值语义建模。

## 3. 这次重写的核心原则

### 3.1 文本继续保留预处理 embedding

本轮不接 raw text encoder。

原因：
- 训练更快
- 工程更稳
- 更适合服务器批量实验

### 3.2 文本必须进入数值中间特征

文本不应该只在输出端出现。

所以这次改成：
- 先从数值 backbone 取中间特征
- 再用文本去调制这些中间特征
- 最后再进入 direct / residual

### 3.3 不靠补丁式超参约束 residual

旧版做法是：
- 用 `beta_delta` 正则去压残差
- 或者用 `force_gain` 人工固定增益

这些都不是很好的主结构设计。

新版思路是：
- 从结构上直接把 residual 做成“历史波动有界”

## 4. 数值 backbone 的新接口

这次没有改数值流 baseline 的训练逻辑。

只新增了：
- `Model_Fit_Num_With_Meta.extract_features()`

返回：
- `forecast`
- `encoded_tokens`
- `summary_state`

这样做的目的：
- `fit_num_with_meta` 的原训练方式不变
- fusion 能拿到真正有用的数值中间表示

## 5. 新版文本侧

### 5.1 不再写死 `llm_dim=768`

旧版写死 `llm_dim=768`，本质上是把当前 GPT2 embedding 维度写进模型结构。

这不合理，因为：
- 以后换 embedding 维度，模型代码不该跟着改

所以新版：
- 文本第一层改为 `LazyLinear`
- 首次 forward 自动适配 `caption_emb` 维度

### 5.2 文本不是直接预测序列

新版文本先做两件事：
- 生成 `gamma`
- 生成 `beta`

然后对数值 latent 做条件调制：
- `latent_fused = latent * (1 + gamma) + beta`

这是一种条件化调制思路，而不是文本自己单独预测一个未来序列。

## 6. 新版 direct

### 6.1 语义

新版 `direct` 表示：
- 文本和数值中间特征共同决定最终预测
- 不是简单的末端线性混合

### 6.2 实现方式

流程：
1. 数值 backbone 提供 `encoded_tokens`
2. 文本调制 `encoded_tokens`
3. 再结合：
   - 文本上下文
   - 调制后的数值摘要
   - `y_num` 摘要
   - 历史统计
4. 最后由 `direct_head` 直接输出最终未来序列

当前不再使用旧版的：
- `(1-w) * y_num + w * y_text`

## 7. 新版 residual

### 7.1 语义

新版 `residual` 表示：
- 数值 backbone 先给出主预测 `y_num`
- 文本只负责输出一个“有边界的纠偏量”

### 7.2 为什么删除 `beta_delta`

`beta_delta` 的问题是：
- 它是额外正则超参
- 调起来很不直观
- 更像在补救结构设计，而不是结构本身合理

所以新版不再使用它。

### 7.3 为什么删除 `force_gain`

`force_gain` 的问题是：
- 过于人工
- 不适合作为主训练逻辑

所以新版也不再用它。

### 7.4 新的 residual 约束

新版 residual 的做法是：
- 先输出 `raw_delta`
- 再通过 `tanh(raw_delta)` 把方向和相对幅度限制住
- 再乘以一个基于历史波动的半径

当前半径来自：
- `history_std`
- `history_range`

所以 residual 的幅度天然受历史波动边界控制。

最终：
- `y_final = y_num + delta`

## 8. 这次重写之后，FIT 应保留的开关

建议保留：
- `text_mode`
- `num_model_path`
- `freeze_numerical`
- `fusion_optimizer_mode`
- `lr_num`
- `lr_text`
- `weight_decay_text`
- `num_feat_dim`
- `fusion_hidden`
- `fusion_dropout`

## 9. 这次重写之后，FIT 已删除或停用的旧项

FIT 当前已删除或停用：
- `beta_delta`
- `force_gain`
- FIT 侧硬编码 `llm_dim`

说明：
- parser 中部分旧参数仍保留，仅用于 Geo 兼容
- 它们当前不再参与 FIT fusion

## 10. 当前实现状态

这份设计已经落地实现到当前代码里。

已实现并 smoke 通过：
- 新版 `direct`
- 新版 `residual`
- `split` 优化器
- `unified` 优化器

## 11. 后续实验建议

接下来最重要的不是再改结构，而是回服务器做正式实验比较：
- baseline 数值流
- 新 `direct + unified`
- 新 `direct + split`
- 新 `residual + freeze_numerical`
- 新 `residual + 不冻结数值流`

如果这些组合仍然涨不动，再优先回头检查：
- 文本描述生成策略
- `caption_emb` 质量
- 文本与数值趋势的真实相关性
