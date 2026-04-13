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
| fit_fusion_halfyear_direct_from_ckpt | `fit_fusion_halfyear_direct_from_ckpt.sh` | structured | 20 |  |  |  |  | direct |
| fit_fusion_halfyear_residual_correction | `fit_fusion_halfyear_residual.sh` | structured | 20 |  |  |  |  | freeze |
| fit_fusion_halfyear_residual_unfreeze | `fit_fusion_halfyear_residual_unfreeze.sh` | structured | 20 |  |  |  |  | best candidate |
