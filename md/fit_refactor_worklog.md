﻿﻿# FIT 重构工作记录

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

- 长文本 random text 的 5 组实验已完成
- 长文本 filler text 的 5 组实验已完成

random text 结果：

- `direct + ckpt + freeze`
  - `MAE = 0.087233`
  - `MAPE = 33.08%`
- `direct + scratch`
  - `MAE = 0.105321`
  - `MAPE = 40.39%`
- `direct + ckpt + unfreeze`
  - `MAE = 0.081796`
  - `MAPE = 30.96%`
- `residual + ckpt + freeze`
  - `MAE = 0.085365`
  - `MAPE = 30.54%`
- `residual + ckpt + unfreeze`
  - `MAE = 0.079736`
  - `MAPE = 28.72%`

filler text 结果：

- `direct + ckpt + freeze`
  - `MAE = 0.087151`
  - `MAPE = 33.05%`
- `direct + scratch`
  - `MAE = 0.105571`
  - `MAPE = 40.74%`
- `direct + ckpt + unfreeze`
  - `MAE = 0.081904`
  - `MAPE = 30.96%`
- `residual + ckpt + freeze`
  - `MAE = 0.085345`
  - `MAPE = 30.51%`
- `residual + ckpt + unfreeze`
  - `MAE = 0.079818`
  - `MAPE = 28.81%`

## 当前最重要的待验证问题

### 问题 1

如果把真实长文本换成 random text，结果会不会明显下降？

现在答案已经比较明确：

- 不会明显下降

说明：

- 当前文本分支没有显著利用文本和数值序列之间的语义对应关系

### 问题 2

如果 filler text 和 random text 结果也差不多，说明：

现在答案也比较明确：

- 是的，二者和真实文本都差不多

说明：

- 当前 fusion 很可能还没有真正把文本语义用起来

### 问题 3

既然 random / filler 和真实长文本几乎一样，当前阶段更合理的判断是：

- 现在这套模型里，文本内容本身不是主导因素
- 当前主导因素更像是：
  - `ckpt + unfreeze`
  - `residual` 的硬数值 skip
  - 融合头里对数值侧辅助特征的利用

## 当前阶段建议

先不要继续改 fusion 结构。

当前最合理的下一步是：

1. 先跑两个 `disable_text` 对照：
   - `direct + ckpt + unfreeze + disable_text`
   - `residual + ckpt + unfreeze + disable_text`
2. 如果它们仍然接近当前最好结果，就可以基本坐实：
   - 当前新版 fusion 的增益不是来自文本语义
3. 然后再决定：
   - 是回退旧版 fusion 头做严格 A/B
   - 还是继续重写新版 fusion，让文本必须真正参与预测

2026-04-14 已补好两个对应脚本：

- `fit_fusion_halfyear_direct_from_ckpt_disable_text.sh`
- `fit_fusion_halfyear_residual_unfreeze_disable_text.sh`

当前不建议把 5 组都补成 `disable_text`，因为最有信息量的就是这两组主路线。
