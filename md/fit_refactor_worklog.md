# FIT 重构工作记录

最后更新：2026-04-14

## 文档用途

这份文档只记录 FIT 侧这轮整理、重构、实验推进的关键结论。
完整结果表和当前执行说明看：

- [fit_server_runbook.md](/D:/zhangjing/project/Dualsg_refined/md/fit_server_runbook.md)
- [global_architecture.md](/D:/zhangjing/project/Dualsg_refined/md/global_architecture.md)

## 已完成的代码整理

### 数据层

- 抽出 FIT 公共数据逻辑到 `data_provider/fit_dataset_utils.py`
- 统一处理：
  - 城市、性别、年龄映射
  - `group` 解析
  - train / val / test 切分
  - `element_map`
  - 标准化
  - 时间特征
  - `caption_emb` 对齐
- 新增 `fit_scaler_mode`
  - `train_only`
  - `split_fit`
- 当前默认：
  - `fit_scaler_mode=train_only`

### 数值流

- 保持 `fit_num` 和 `fit_num_with_meta` 的训练逻辑不变
- 只给 `Model_Fit_Num_With_Meta` 增加了 `extract_features()` 接口，给 fusion 读取：
  - `forecast`
  - `encoded_tokens`
  - `summary_state`

### FIT fusion

- 保留离线 `caption_emb` 路线，不接 raw text encoder
- `direct` 和 `residual` 仍然都依赖 `Model_Fit_Num_With_Meta`
- 当前新版 fusion 是“文本条件化数值中间特征”的结构
- 当前实验矩阵已经补齐：
  - `direct + scratch`
  - `direct + ckpt + freeze`
  - `direct + ckpt + unfreeze`
  - `residual + ckpt + freeze`
  - `residual + ckpt + unfreeze`

### 训练与脚本

- `split lr` 被 scheduler 覆盖的问题已经修掉
- 当前 `fusion_optimizer_mode=split` 时：
  - `lr_num` 控制数值参数组
  - `lr_text` 控制文本/融合参数组
- 当前 half-year FIT fusion 脚本统一使用：
  - `train_epochs = 20`
  - `batch_size = 200`
  - `patience = 100`
  - `adjust = 0`

## 本地与服务器实验结论

### baseline

- `fit_num_with_meta`
  - `MAE = 0.085858`
  - `MSE = 0.013315`
  - `RMSE = 0.115391`
  - `MAPE = 30.23%`
  - `WAPE = 18.52%`

### 长文本结果

- half-year 20 epoch
  - 最好：`residual + ckpt + unfreeze`
  - `MAE = 0.079863`
  - `MAPE = 28.83%`
- half-year 100 epoch
  - 最好：`residual + ckpt + unfreeze`
  - `MAE = 0.075699`
  - `MAPE = 27.30%`

### 结构化与多视角文本结果

目前已经跑完的 20 epoch 文本版本：

- 结构化完整文本
- Global Structural View
- Dynamic Behavior View
- Event-centric View
- Semantic Caption View

这些版本的排序非常稳定：

1. `residual + ckpt + unfreeze`
2. `direct + ckpt + unfreeze`
3. `residual + ckpt + freeze`
4. `direct + ckpt + freeze`
5. `direct + scratch`

代表性最优结果：

- 结构化完整文本
  - 最好：`residual + ckpt + unfreeze`
  - `MAE = 0.079866`
  - `MAPE = 28.95%`
- Global Structural View
  - 最好：`residual + ckpt + unfreeze`
  - `MAE = 0.079282`
  - `MAPE = 28.82%`
- Dynamic Behavior View
  - 最好：`residual + ckpt + unfreeze`
  - `MAE = 0.079980`
  - `MAPE = 28.72%`
- Event-centric View
  - 最好：`residual + ckpt + unfreeze`
  - `MAE = 0.079809`
  - `MAPE = 28.90%`
- Semantic Caption View
  - 最好：`residual + ckpt + unfreeze`
  - `MAE = 0.079866`
  - `MAPE = 28.95%`

### 当前判断

- 目前真正值得继续看的仍然是：
  - `residual + ckpt + unfreeze`
  - `direct + ckpt + unfreeze`
- 两个 freeze 版本更像对照/消融，不是主路线
- `direct + scratch` 基本可以视为弱对照
- 结构化和多视角文本目前只带来很小波动，没有带来跨档提升

## 和旧版最好结果的对比

用户给出的旧版最好结果：

- `MAE = 0.071344`
- `MSE = 0.009260`
- `RMSE = 0.096228`
- `MAPE = 25.23%`
- `WAPE = 15.39%`

当前新版 fusion 还没有追上这一组结果。

但要注意，这不是完全严格的一对一对比，因为这期间同时变了：

- fusion 结构
- scaler 语义
- 学习率调度逻辑
- 脚本默认设置
- 文本 `pt` 版本

## 新增的对照实验

2026-04-14 已经加入两类控制文本：

- random text
- filler text

对应生成脚本：

- [generate_random_text_pt.py](/D:/zhangjing/project/Dualsg_refined/dataset/FIT_DualSG/scripts/generate_random_text_pt.py)
- [generate_filler_text_pt.py](/D:/zhangjing/project/Dualsg_refined/dataset/FIT_DualSG/scripts/generate_filler_text_pt.py)

它们都基于：

- `./dataset/FIT_DualSG/fit_dualsg_all.json`

并且保持：

- 样本数一致
- 样本顺序一致

当前状态：

- 长文本 random text 的 5 组实验已经开始跑
- 下一轮准备跑长文本 filler text 的 5 组实验

## 当前最重要的待验证问题

### 问题 1

如果把真实长文本换成 random text，结果会不会明显下降？

如果会明显下降，说明：

- 当前模型确实在利用文本和数值序列之间的语义对应关系

如果不会明显下降，说明：

- 当前文本分支可能主要只利用了分布层面的弱信号

### 问题 2

如果 filler text 和 random text 结果也差不多，说明：

- 当前 fusion 很可能还没有真正把文本语义用起来

### 问题 3

如果 random / filler 都显著差于真实长文本，再去比较：

- 长文本
- 结构化文本
- 多视角文本

这个对比才更有意义

## 当前阶段建议

先不要继续改 fusion 结构。

当前最合理的顺序是：

1. 跑完 long-text random text 的 5 组
2. 再跑 long-text filler text 的 5 组
3. 对比：
   - 真实长文本
   - random text
   - filler text
4. 只有在这一步结论清楚后，再决定是否回头继续改模型结构
