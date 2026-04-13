# FIT 下一轮计划

最后更新：2026-04-13

## 1. 当前问题判断

基于 2026-04-13 的首轮服务器结果，当前结论是：

- `fit_num_with_meta` 依然是稳定且强的 baseline。
- 新版 `fit_fusion_halfyear_joint_direct` 首轮正式结果弱于 baseline。
- 现在优先要排查训练 recipe 和实现细节，而不是继续盲目增加更复杂的文本结构。

首轮正式结果对比：

- `fit_num_with_meta`
  - `MAE  = 0.085858`
  - `MSE  = 0.013315`
  - `RMSE = 0.115391`
  - `MAPE = 30.23%`
  - `WAPE = 18.52%`
- `fit_fusion_halfyear_joint_direct`
  - `MAE  = 0.091474`
  - `MSE  = 0.014394`
  - `RMSE = 0.119975`
  - `MAPE = 35.42%`
  - `WAPE = 19.73%`

## 2. 已确认问题

### 2.1 split 学习率被 scheduler 覆盖

当前 `fusion_optimizer_mode=split` 虽然会把参数分成：

- numerical group -> `lr_num`
- text/fusion group -> `lr_text`

但 `utils/tools.py` 里的 `adjust_learning_rate()` 会在每个 epoch 结束后，把所有 param group 的学习率统一覆盖成 `args.learning_rate`。

这会导致：

- `split` 只在 optimizer 初始化那一刻是分开的
- 真正训练时很快又退化回统一学习率

### 2.2 20 epoch + type3 调度几乎没有实际效果

当前 FIT fusion 官方脚本默认：

- `train_epochs = 20`
- `lradj = type3`

而 `type3` 的当前定义是：

- `lr = learning_rate * (0.1 ** (epoch // 20))`

这意味着前 19 个 epoch 基本都不会发生有效衰减，20 epoch 这个设置下调度几乎没有实际意义。

### 2.3 当前 direct recipe 不是公平对比

当前 `fit_fusion_*_direct.sh` 的默认语义是：

- 从头训练
- 不加载数值 ckpt
- 不冻结数值流

它更像“一个新的 joint 模型”，而不是“在强数值主干上加文本增强”。

## 3. 本轮修改目标

本轮只做三件事：

1. 修复 `split lr` 被 scheduler 覆盖的问题。
2. 把 FIT fusion 官方脚本默认 recipe 改成更合理的版本。
3. 补充下一轮推荐实验脚本，并同步文档。

## 4. 具体实施顺序

### 4.1 先修学习率调度逻辑

目标：

- 让 `lr_num` 和 `lr_text` 在整个训练过程中都能保留 param-group 语义。

方案：

- 给每个 param group 保留自己的 `base_lr`。
- scheduler 只计算一个相对缩放因子。
- 每个 group 按自己的 `base_lr * scale` 更新，而不是统一覆盖成 `args.learning_rate`。

这一步既适用于 FIT fusion，也不会破坏 unified 模式。

### 4.2 FIT fusion 官方脚本默认先改成 `--adjust 0`

目标：

- 先去掉当前 20 epoch + `type3` 这个几乎无效的干扰项。

做法：

- `fit_fusion_halfyear_direct.sh`
- `fit_fusion_halfyear_residual.sh`
- `fit_fusion_oneyear_direct.sh`
- `fit_fusion_oneyear_residual.sh`

统一显式加上：

- `--adjust 0`

这样下一轮实验能更干净地验证：

- split 学习率是否有效
- 结构本身是否有帮助

### 4.3 增加更公平的推荐实验脚本

新增推荐组合：

- `direct + load num ckpt + unfreeze`
- `residual + load num ckpt + unfreeze`

目的：

- 给 direct 一个更公平的强初始化
- 验证 residual 是否需要和数值流联合微调

## 5. 修改后建议实验顺序

半年任务优先，建议按下面顺序跑：

1. `fit_num_with_meta` baseline
2. `direct + from scratch + split + adjust=0`
3. `direct + load num ckpt + unfreeze + split + adjust=0`
4. `residual + load num ckpt + freeze + split + adjust=0`
5. `residual + load num ckpt + unfreeze + split + adjust=0`

如果半年任务里出现明确提升，再复制到一年任务。

## 6. 这一轮不做的事

本轮不做：

- 改数值主干建模逻辑
- 引入 raw text encoder
- 重新生成 `caption_emb`
- 大改 loss 设计

这些都留到确认训练 recipe 没问题之后再考虑。
