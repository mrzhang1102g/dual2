# FIT 服务器执行手册

最后更新：2026-04-13

## 当前重点

当前先只看 FIT half-year。

你更关注：

- `MAE`
- `MAPE`

当前已经知道的结论：

- `residual + num_ckpt + unfreeze` 是目前最强候选
- `direct + num_ckpt + unfreeze` 是第二候选
- `residual + freeze` 更像纯纠偏对照组
- `direct + scratch` 不是主路线

## 已有正式结果

### 数值 baseline

- `fit_num_with_meta`
  - `MAE = 0.085858`
  - `MSE = 0.013315`
  - `RMSE = 0.115391`
  - `MAPE = 30.23%`
  - `WAPE = 18.52%`

### half-year 20 epoch，完整长文本 pt

- `direct + scratch`
  - `MAE = 0.105100`
  - `MSE = 0.018546`
  - `RMSE = 0.136183`
  - `MAPE = 40.06%`
  - `WAPE = 22.67%`
- `direct + num_ckpt + unfreeze`
  - `MAE = 0.081830`
  - `MSE = 0.011850`
  - `RMSE = 0.108857`
  - `MAPE = 31.00%`
  - `WAPE = 17.65%`
- `residual + num_ckpt + freeze`
  - `MAE = 0.085395`
  - `MSE = 0.013151`
  - `RMSE = 0.114679`
  - `MAPE = 30.38%`
  - `WAPE = 18.42%`
- `residual + num_ckpt + unfreeze`
  - `MAE = 0.079863`
  - `MSE = 0.011620`
  - `RMSE = 0.107797`
  - `MAPE = 28.83%`
  - `WAPE = 17.23%`

### half-year 100 epoch，完整长文本 pt

- `direct + scratch`
  - `MAE = 0.088332`
  - `MSE = 0.013615`
  - `RMSE = 0.116683`
  - `MAPE = 33.22%`
  - `WAPE = 19.06%`
- `direct + num_ckpt + unfreeze`
  - `MAE = 0.077575`
  - `MSE = 0.010672`
  - `RMSE = 0.103305`
  - `MAPE = 29.91%`
  - `WAPE = 16.74%`
- `residual + num_ckpt + freeze`
  - `MAE = 0.085395`
  - `MSE = 0.013151`
  - `RMSE = 0.114679`
  - `MAPE = 30.38%`
  - `WAPE = 18.42%`
- `residual + num_ckpt + unfreeze`
  - `MAE = 0.075699`
  - `MSE = 0.010511`
  - `RMSE = 0.102523`
  - `MAPE = 27.30%`
  - `WAPE = 16.33%`

## 当前下一轮实验

下一轮要跑的是：

- half-year 的 4 个 fusion 脚本
- `train_epochs = 20`
- `patience = 100`
- `caption_emb_path = ./dataset/FIT_DualSG/pt/fit_dualsg_structured.pt`

这 4 个脚本已经改好：

1. `fit_fusion_halfyear_direct.sh`
2. `fit_fusion_halfyear_direct_from_ckpt.sh`
3. `fit_fusion_halfyear_residual.sh`
4. `fit_fusion_halfyear_residual_unfreeze.sh`

## 当前脚本约定

### 训练数据

当前仍然读：

- `./dataset/FIT_DualSG/fit_dualsg_all.json`

### 文本 embedding

当前这轮 half-year 脚本读：

- `./dataset/FIT_DualSG/pt/fit_dualsg_structured.pt`

### 数值 checkpoint

当前这轮 3 个 from-ckpt 脚本直接写死为：

- `./model_checkpoints/fit_halfyear_num_with_meta_20260413_083428/checkpoint.pth`

### 其它默认值

- `batch_size = 200`
- `fusion_optimizer_mode = split`
- `learning_rate = 0.001`
- `lr_num = 0.0001`
- `lr_text = 0.0005`
- `adjust = 0`
- `fit_scaler_mode = train_only`

## 运行顺序

建议顺序：

1. `fit_fusion_halfyear_residual_unfreeze.sh`
2. `fit_fusion_halfyear_direct_from_ckpt.sh`
3. `fit_fusion_halfyear_residual.sh`
4. `fit_fusion_halfyear_direct.sh`

## 如何判断结果

优先看：

1. `MAE`
2. `MAPE`
3. `WAPE`
4. `MSE`

当前判断规则：

- 如果 `residual + unfreeze` 仍然最好
  - 先把这条路线作为主路线
  - 再考虑复制到 one-year
- 如果 `direct + from_ckpt + unfreeze` 追平或反超
  - 保留 direct / residual 两条路线同时汇报
- 如果 4 组都明显差于 baseline
  - 暂停继续扩展实验
  - 回头继续改 fusion 结构

## 结果记录模板

| setting | script | text pt | epochs | MAE | MAPE | WAPE | MSE | 备注 |
|---|---|---|---:|---:|---:|---:|---:|---|
| fit_num_with_meta | `fit_halfyear_num_with_meta.sh` | - | 20 |  |  |  |  | baseline |
| fit_fusion_halfyear_joint_direct | `fit_fusion_halfyear_direct.sh` | structured | 20 |  |  |  |  | scratch |
| fit_fusion_halfyear_direct_freeze | `fit_fusion_halfyear_direct_freeze.sh` | structured | 20 |  |  |  |  | direct + ckpt + freeze |
| fit_fusion_halfyear_direct_from_ckpt | `fit_fusion_halfyear_direct_from_ckpt.sh` | structured | 20 |  |  |  |  | direct |
| fit_fusion_halfyear_residual_correction | `fit_fusion_halfyear_residual.sh` | structured | 20 |  |  |  |  | freeze |
| fit_fusion_halfyear_residual_unfreeze | `fit_fusion_halfyear_residual_unfreeze.sh` | structured | 20 |  |  |  |  | best candidate |

## 2026-04-13 addendum

Added one more half-year script for a cleaner direct ablation:

- `fit_fusion_halfyear_direct_freeze.sh`

Meaning:

- load numerical checkpoint
- freeze numerical backbone
- train only text/fusion-side parameters
- `text_mode = direct`

Why add it:

- this completes the direct-side comparison:
  - `direct + scratch`
  - `direct + ckpt + freeze`
  - `direct + ckpt + unfreeze`
- `residual + scratch` is intentionally not added for now because residual is easier to interpret as a correction model when a strong numerical predictor already exists

## 2026-04-13 structured / multi-view results

Baseline:

- `fit_num_with_meta`
  - `MAE = 0.085858`
  - `MAPE = 30.23%`
  - `WAPE = 18.52%`

### structured full text, 20 epoch

- `direct + ckpt + freeze`
  - `MAE = 0.087139`
  - `MAPE = 33.10%`
  - `WAPE = 18.80%`
- `direct + scratch`
  - `MAE = 0.105514`
  - `MAPE = 40.66%`
  - `WAPE = 22.76%`
- `direct + ckpt + unfreeze`
  - `MAE = 0.081992`
  - `MAPE = 30.98%`
  - `WAPE = 17.69%`
- `residual + ckpt + freeze`
  - `MAE = 0.085368`
  - `MAPE = 30.39%`
  - `WAPE = 18.42%`
- `residual + ckpt + unfreeze`
  - `MAE = 0.079866`
  - `MAPE = 28.95%`
  - `WAPE = 17.23%`

### global structural view, 20 epoch

- `direct + ckpt + freeze`
  - `MAE = 0.087178`
  - `MAPE = 33.00%`
  - `WAPE = 18.81%`
- `direct + scratch`
  - `MAE = 0.105468`
  - `MAPE = 40.78%`
  - `WAPE = 22.75%`
- `direct + ckpt + unfreeze`
  - `MAE = 0.081979`
  - `MAPE = 30.98%`
  - `WAPE = 17.69%`
- `residual + ckpt + freeze`
  - `MAE = 0.085385`
  - `MAPE = 30.76%`
  - `WAPE = 18.42%`
- `residual + ckpt + unfreeze`
  - `MAE = 0.079282`
  - `MAPE = 28.82%`
  - `WAPE = 17.22%`

### dynamic behavior view, 20 epoch

- `direct + ckpt + freeze`
  - `MAE = 0.087238`
  - `MAPE = 33.46%`
  - `WAPE = 18.82%`
- `direct + scratch`
  - `MAE = 0.105344`
  - `MAPE = 40.46%`
  - `WAPE = 22.73%`
- `direct + ckpt + unfreeze`
  - `MAE = 0.081835`
  - `MAPE = 31.09%`
  - `WAPE = 17.65%`
- `residual + ckpt + freeze`
  - `MAE = 0.085409`
  - `MAPE = 30.31%`
  - `WAPE = 18.43%`
- `residual + ckpt + unfreeze`
  - `MAE = 0.079980`
  - `MAPE = 28.72%`
  - `WAPE = 17.25%`

### event-centric view, 20 epoch

- `direct + ckpt + freeze`
  - `MAE = 0.087092`
  - `MAPE = 33.24%`
  - `WAPE = 18.79%`
- `direct + scratch`
  - `MAE = 0.105500`
  - `MAPE = 40.61%`
  - `WAPE = 22.76%`
- `direct + ckpt + unfreeze`
  - `MAE = 0.081763`
  - `MAPE = 31.14%`
  - `WAPE = 17.64%`
- `residual + ckpt + freeze`
  - `MAE = 0.085376`
  - `MAPE = 30.64%`
  - `WAPE = 18.42%`
- `residual + ckpt + unfreeze`
  - `MAE = 0.079809`
  - `MAPE = 28.90%`
  - `WAPE = 17.22%`

### semantic caption view, 20 epoch

- `direct + ckpt + freeze`
  - `MAE = 0.087139`
  - `MAPE = 33.10%`
  - `WAPE = 18.80%`
- `direct + scratch`
  - `MAE = 0.105514`
  - `MAPE = 40.66%`
  - `WAPE = 22.76%`
- `direct + ckpt + unfreeze`
  - `MAE = 0.081992`
  - `MAPE = 30.98%`
  - `WAPE = 17.69%`
- `residual + ckpt + freeze`
  - `MAE = 0.085368`
  - `MAPE = 30.39%`
  - `WAPE = 18.42%`
- `residual + ckpt + unfreeze`
  - `MAE = 0.079866`
  - `MAPE = 28.95%`
  - `WAPE = 17.23%`

### current conclusion

- All new 20-epoch text variants stay in the same ranking:
  - best: `residual + ckpt + unfreeze`
  - second: `direct + ckpt + unfreeze`
  - weak ablation: `residual + ckpt + freeze`
  - weak ablation: `direct + ckpt + freeze`
  - worst: `direct + scratch`
- Compared with the 100-epoch long-text run, these new 20-epoch variants do not show a clear improvement trend.
- `structured full text` and `semantic caption view` are numerically identical in the table above; if they were intended to be different text assets, the corresponding `.pt` files should be checked later.

## 2026-04-14 random / filler controls

Added two direct PT-generation scripts under [dataset/FIT_DualSG/scripts](/D:/zhangjing/project/Dualsg_refined/dataset/FIT_DualSG/scripts):

- [generate_random_text_pt.py](/D:/zhangjing/project/Dualsg_refined/dataset/FIT_DualSG/scripts/generate_random_text_pt.py)
- [generate_filler_text_pt.py](/D:/zhangjing/project/Dualsg_refined/dataset/FIT_DualSG/scripts/generate_filler_text_pt.py)

Both controls are built against:

- `./dataset/FIT_DualSG/fit_dualsg_all.json`

They do not need new json files:

- `random_text.pt`: shuffle the real long-text captions across samples while keeping sample count and order aligned with `fit_dualsg_all.json`
- `filler_text.pt`: use the same fixed placeholder sentence for every sample while keeping sample count and order aligned with `fit_dualsg_all.json`

Current half-year fusion scripts are temporarily switched to:

- `./dataset/FIT_DualSG/pt/fit_dualsg_random_text.pt`

So the next run is the random-text control round on top of the same `fit_dualsg_all.json` numerical data.
