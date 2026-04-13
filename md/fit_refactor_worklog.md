# FIT Refactor Worklog

最后更新：2026-04-13

## 1. 文档用途

这份文档记录 FIT 侧整理与重构的当前状态、真实 smoke 结果和后续待办。

建议阅读顺序：
1. `fit_refactor_worklog.md`
2. `global_architecture.md`
3. `fit_fusion_redesign.md`

## 2. 当前阶段结论

FIT 侧已经完成两轮整理：
- 第一轮：梳理数据层、实验层、脚本和基础文档。
- 第二轮：修正 scaler 语义、重写 FIT fusion、补齐本地 CPU 验证。

当前结论：
- 数值流训练逻辑保持原样。
- `fit_num_with_meta` 只新增了给 fusion 读取中间特征的接口。
- FIT fusion 已经从旧的浅融合，切换成“文本条件化数值中间特征”的版本。

## 3. 已完成事项

### 3.1 数据层

已完成：
- 抽出 FIT 公共数据逻辑到 `data_provider/fit_dataset_utils.py`
- 统一管理：
  - `CITY_MAP / GENDER_MAP / AGE_MAP`
  - `group` 解析
  - train / val / test split
  - `element_map`
  - 标准化
  - 时间特征
  - `caption_emb` 对齐
- 新增 `fit_scaler_mode`
  - `train_only`
  - `split_fit`
- 当前默认：
  - `fit_scaler_mode=train_only`

### 3.2 数值流

已完成：
- `Exp_Fit_Num` 收敛为只负责 FIT 数值流。
- 数值流本身训练逻辑不改。
- `Model_Fit_Num_With_Meta` 新增 `extract_features()`：
  - 返回 `forecast`
  - 返回 `encoded_tokens`
  - 返回 `summary_state`
- 这个接口只给 fusion 使用，不改变原 baseline 训练语义。

### 3.3 FIT fusion

已完成：
- 保留预处理 `caption_emb` 路线，不接 raw text encoder。
- 重写 `Model_Fit_Fusion` 为文本条件化版本：
  - 文本适配器 `text_adapter`
  - 文本生成 `gamma / beta`
  - 文本调制数值 `encoded_tokens`
  - 再进入 `direct / residual`

新的 `direct`：
- 不再做 `(1-w) * y_num + w * y_text`
- 直接从“文本调制后的数值特征”输出最终预测

新的 `residual`：
- 不再使用 `beta_delta`
- 不再使用 `force_gain`
- 输出“历史波动有界”的纠偏量

### 3.4 CLI 与脚本

已完成：
- `run.py` 已按参数组整理。
- FIT 侧展示层已经去掉旧 fusion 超参的打印。
- FIT fusion 脚本已清掉：
  - `--beta_delta`
  - `--llm_dim`
- FIT fusion 官方脚本已显式补上：
  - `--train_epochs 20`
  - `--batch_size 200`

注意：
- `run.py` 中 `llm_dim / delta_scale / use_vol_prior / force_gain` 仍保留在 parser 里
- 这是为了不破坏当前 Geo 路径
- FIT 当前实现已经不再使用这些参数

## 4. 本地环境状态

当前 base Python 环境已装好：
- `torch` CPU
- `numpy`
- `scipy`
- `scikit-learn`
- `pandas`
- `matplotlib`
- `tqdm`
- `einops`
- `reformer_pytorch`

静态检查已通过：
- `python run.py --help`
- `python -m compileall -q run.py exp models data_provider utils layers`

## 5. 真实 smoke 结果

### 5.1 数值流

已真实跑通：

1. `fit_num`
- CPU
- 小样本
- 训练 / 验证 / 测试 / checkpoint / 结果文件全部正常

2. `fit_num_with_meta`
- CPU
- 小样本
- 第一轮整理后跑通过

3. `fit_num_with_meta` 复测
- 在新增 `extract_features()` 之后重新跑过
- 训练 / 验证 / 测试仍正常
- 说明数值流原语义没有被改坏

### 5.2 scaler 语义

已真实验证：
- `fit_scaler_mode=train_only` 下
  - train / val / test 共享同一个 train-fit scaler
  - 不再各自 fit 各自 split

### 5.3 新版 FIT fusion

由于正式 `fit_caption_emb_all.pt` 不在本地，本轮使用临时小子集和临时 `caption_emb` 做代码路径 smoke。临时资产只用于测试，后面已删除。

已真实跑通：

1. 新版 `direct + split`
- 语义：`joint_direct`
- 结果：训练 / 验证 / 测试正常

2. 新版 `residual + split`
- 语义：`residual_correction`
- 加载本地数值 smoke checkpoint
- `freeze_numerical=True`
- 结果：训练 / 验证 / 测试正常

3. 新版 `direct + unified`
- 验证统一学习率模式
- 结果：训练 / 验证 / 测试正常

## 6. 本轮关键设计变化

### 6.1 删除的旧 FIT fusion 思路

FIT 当前已经不再依赖：
- `beta_delta`
- `force_gain`
- 文本侧写死输入维度 `llm_dim=768`

### 6.2 新的文本作用方式

旧版问题：
- 文本只在输出端做一个浅 head
- 很容易退化成“数值预测 + 小修小补”

新版改成：
- 文本先调制数值 backbone 的中间 `encoded_tokens`
- 再进入 direct / residual

### 6.3 新 residual 的约束方式

旧版：
- 靠额外正则限制纠偏幅度

新版：
- 直接用历史 `std + range` 给 residual 设上界
- 纠偏幅度天然有边界
- 语义更清楚，也更容易解释

## 7. 当前剩余风险

### 7.1 本地还没用正式 `fit_caption_emb_all.pt`

现在能确认的是：
- 代码路径通了
- 新 direct / residual 通了
- split / unified 都通了

现在还不能确认的是：
- 在正式 `caption_emb` 和服务器全量训练上，新结构一定优于旧结构

### 7.2 文本瓶颈仍在描述策略本身

即使训练框架理顺后，文本侧上限仍然取决于：
- 文本描述是否真的有信息量
- 文本描述是否和数值趋势有关
- 预处理 embedding 是否足够表达这些信息

## 8. 下一步建议

下一步建议顺序：
1. 在服务器上用正式 `fit_caption_emb_all.pt` 跑新版 `joint_direct`
2. 再跑新版 `residual_correction`
3. 比较：
   - baseline 数值流
   - `direct + unified`
   - `direct + split`
   - `residual + freeze_numerical`
   - `residual + 不冻结数值流`
4. 如果新版 fusion 仍然涨不动，再回头处理“文本描述如何生成”这个更上游的问题

## 9. 下一版 fusion 设计文档

如果后面还要继续扩展 FIT fusion，请先看：
- `fit_fusion_redesign.md`

这份文档记录的是：
- 这次重写采用的设计原则
- 为什么删掉 `beta_delta / force_gain / llm_dim`
- direct / residual 的新语义

## 10. 2026-04-13 服务器首轮结果

### 10.1 半年 `fit_num_with_meta` baseline

当前用户反馈的正式结果：
- `MAE  = 0.085858`
- `MSE  = 0.013315`
- `RMSE = 0.115391`
- `MAPE = 30.23%`
- `WAPE = 18.52%`

### 10.2 半年新版 `fit_fusion_halfyear_joint_direct`

当前用户反馈的正式结果：
- `MAE  = 0.091474`
- `MSE  = 0.014394`
- `RMSE = 0.119975`
- `MAPE = 35.42%`
- `WAPE = 19.73%`

结论：
- 当前新版 `direct` 首轮正式实验弱于数值主干 baseline
- 因此现阶段不能判断“文本流设计已经有效”

## 11. 当前已确认问题

### 11.1 学习率分组被 scheduler 覆盖

当前 `fusion_optimizer_mode=split` 虽然初始化时会分成：
- num group = `lr_num`
- text group = `lr_text`

但 `adjust_learning_rate()` 会在每个 epoch 结束后把所有 param group 统一覆盖成 `args.learning_rate`。

这意味着：
- `split` 现在不是全程分组学习率
- 只是初始化时短暂分组，后面就被抹平

### 11.2 `type3` 在 20 epoch 训练里几乎不起作用

当前 `type3` 的写法是：
- `lr = learning_rate * (0.1 ** (epoch // 20))`

而脚本默认：
- `train_epochs = 20`

再加上 lr 调整发生在 epoch 结束后，所以：
- 前 19 个 epoch 都是 `0.001`
- 到第 20 个 epoch 结束才会准备降 lr
- 但训练其实已经结束

也就是说：
- 当前 20 epoch 配置下，`type3` 基本没有实际调度效果

### 11.3 当前 `joint_direct` 不是公平的“文本增强 baseline”

现在官方 `direct` 脚本语义是：
- 从头 joint 训练
- 不加载数值 ckpt
- 不冻结数值流

这意味着它不是“在强数值主干基础上加文本”，而是：
- 一个带文本条件化的新模型，从头重训

所以它首轮弱于已有强 baseline，并不意外。

## 12. 当前判断

现阶段更像是：
- 数值主干本身是稳定且强的
- 文本实验的首要问题，未必是文本信息本身无用
- 更可能是训练 recipe 和实验设计还没有站在一个公平起点上

优先级判断：
1. 先修学习率与调度问题
2. 再重新跑 direct / residual
3. 再判断文本设计本身是否有效

## 13. 下一轮实验建议

建议按下面顺序做，而不是继续直接堆新结构：

1. 先修 `split lr` 被覆盖的问题
- 至少保证 `lr_num` 和 `lr_text` 能全程独立

2. 先把 fusion 脚本改成不使用当前无效的 `type3`
- 一个简单可行方案是 `--adjust 0`

3. 半年任务优先重跑下面 4 组
- `direct + from scratch + split + adjust=0`
- `direct + load num ckpt + unfreeze + split + adjust=0`
- `residual + load num ckpt + freeze + split + adjust=0`
- `residual + load num ckpt + unfreeze + split + adjust=0`

4. 如果上述仍无提升，再看结构层
- 是否给 direct 保留更强的 `y_num` skip
- 是否先对 `caption_emb` 做更稳的归一化
- 是否需要重新检查文本描述本身的质量

## 14. 2026-04-13 下一轮修改计划

当前判断已经比较明确：
- 数值主干 `fit_num_with_meta` 仍然稳定且强
- 新版文本流首轮正式结果弱于 baseline
- 现在优先要排查训练 recipe 与实现细节，而不是继续盲目改更复杂的文本结构

下一轮修改按这个顺序执行：

1. 先修 `split lr` 被 scheduler 覆盖的问题
- 目标：让 `lr_num` 和 `lr_text` 在整个训练过程中都能保持分组语义
- 做法：修改学习率调度逻辑，让每个 param group 基于自己的 `base_lr` 衰减，而不是统一覆盖成 `args.learning_rate`

2. 同时修正 FIT fusion 官方脚本的默认训练策略
- 当前 20 epoch + `type3` 几乎没有有效调度
- 下一轮默认先改成：
  - `--adjust 0`
- 这样先把 scheduler 干扰拿掉，保证实验结论更干净

3. 补一组更公平的官方实验入口
- 不是只保留：
  - `joint_direct`
  - `residual_correction`
- 还要补上更关键的组合：
  - `direct + load num ckpt + unfreeze`
  - `residual + load num ckpt + unfreeze`

4. 完成后做本地静态检查
- `compileall`
- `run.py --help`
- 核对脚本参数与文档一致

5. 文档持续同步
- `fit_refactor_worklog.md`
- `global_architecture.md`
- 如有需要，补充 `fit_fusion_redesign.md`

## 15. 2026-04-13 学习率与脚本修正

本轮已完成：

1. 修复 `split lr` 被 scheduler 覆盖的问题
- `exp/exp_fit_fusion.py` 里的 optimizer param group 现在会显式保存自己的 `base_lr`
- `utils/tools.py` 里的 `adjust_learning_rate()` 不再把所有 group 一起覆盖成 `args.learning_rate`
- 现在 scheduler 会按每个 group 自己的 `base_lr` 做相对缩放

2. 修正 FIT fusion 官方脚本默认 recipe
- 4 个原有 FIT fusion 脚本现在都显式加上了 `--adjust 0`
- 这样可以先去掉当前 `20 epoch + type3` 的无效调度干扰

3. 补充更公平的推荐实验脚本
- `fit_fusion_halfyear_direct_from_ckpt.sh`
- `fit_fusion_halfyear_residual_unfreeze.sh`
- `fit_fusion_oneyear_direct_from_ckpt.sh`
- `fit_fusion_oneyear_residual_unfreeze.sh`

4. 本轮静态验证
- `python -m compileall -q run.py exp models data_provider utils layers` 已通过
- 单独做了一个 scheduler 小测试，确认 `type3` 在 split 模式下会把：
  - numerical group: `1e-4 -> 1e-5`
  - text_fusion group: `5e-4 -> 5e-5`

当前判断：
- 当前最值得重新跑的还是 4 组半年实验：
  - `direct + from scratch + split + adjust=0`
  - `direct + load num ckpt + unfreeze + split + adjust=0`
  - `residual + load num ckpt + freeze + split + adjust=0`
  - `residual + load num ckpt + unfreeze + split + adjust=0`

补充：
- 服务器执行顺序与结果记录模板已单独写入 `fit_server_runbook.md`

## 16. 2026-04-13 半年 20 epoch 正式结果

用户已完成半年任务 5 组结果：

- `fit_num_with_meta`
  - `MAE  = 0.085858`
  - `MSE  = 0.013315`
  - `RMSE = 0.115391`
  - `MAPE = 30.23%`
  - `WAPE = 18.52%`
- `direct + scratch`
  - `MAE  = 0.105100`
  - `MSE  = 0.018546`
  - `RMSE = 0.136183`
  - `MAPE = 40.06%`
  - `WAPE = 22.67%`
- `direct + num_ckpt + unfreeze`
  - `MAE  = 0.081830`
  - `MSE  = 0.011850`
  - `RMSE = 0.108857`
  - `MAPE = 31.00%`
  - `WAPE = 17.65%`
- `residual + num_ckpt + freeze`
  - `MAE  = 0.085395`
  - `MSE  = 0.013151`
  - `RMSE = 0.114679`
  - `MAPE = 30.38%`
  - `WAPE = 18.42%`
- `residual + num_ckpt + unfreeze`
  - `MAE  = 0.079863`
  - `MSE  = 0.011620`
  - `RMSE = 0.107797`
  - `MAPE = 28.83%`
  - `WAPE = 17.23%`

当前结论：

- `direct + scratch` 可以视为当前无效对照组。
- `direct + num_ckpt + unfreeze` 已经明显优于 baseline。
- `residual + num_ckpt + freeze` 略优于 baseline，但提升有限。
- `residual + num_ckpt + unfreeze` 当前是半年任务里最强的一组。

## 17. 2026-04-13 下一步：先跑 100 epoch

当前不继续改 fusion 结构，先把训练 recipe 拉长到 100 epoch。

已完成：

- 8 个 FIT fusion 脚本改为默认 `train_epochs=100`
- 默认 `patience=30`
- 保留 `adjust=0`
- 脚本支持环境变量覆盖，不需要以后反复手改

当前建议：

- 先继续跑半年这 4 个 fusion 组合的 100 epoch 版本
- 跑完再决定是否继续改结构，还是直接复制到一年任务
