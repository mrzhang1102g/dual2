# FIT 服务器执行顺序与结果记录模板

最后更新：2026-04-13

## 1. 当前判断

当前不建议继续改 FIT fusion 结构。

原因：

- 数值主干 `fit_num_with_meta` 已经稳定且强。
- FIT fusion 刚刚完成两项关键修正：
  - `split lr` 不再被 scheduler 覆盖
  - 官方脚本默认关闭了当前无效的 `20 epoch + type3` 调度干扰
- 在这些修正真正跑完之前，继续改模型容易把“结构问题”和“训练 recipe 问题”混在一起。

当前更合理的策略是：

1. 先跑一轮新的正式对照实验。
2. 先看 recipe 修正后，文本流是否仍然整体无效。
3. 只有在这轮仍然明显不如 baseline 时，再继续改 fusion 结构。

## 1.1 当前 20 epoch 半年结果

你已经跑完的半年结果说明当前方向是有信号的：

- `fit_num_with_meta`
  - `MSE = 0.013315`
- `direct + scratch`
  - `MSE = 0.018546`
- `direct + num_ckpt + unfreeze`
  - `MSE = 0.011850`
- `residual + num_ckpt + freeze`
  - `MSE = 0.013151`
- `residual + num_ckpt + unfreeze`
  - `MSE = 0.011620`

当前判断：

- `direct + scratch` 明显不优，100 epoch 仍然更像对照组。
- 最值得继续追的是：
  - `direct + num_ckpt + unfreeze`
  - `residual + num_ckpt + unfreeze`
- `residual + freeze` 也值得保留，因为它代表“纯纠偏”思路。

## 2. 运行前准备

默认在项目根目录执行：

```bash
cd /path/to/Dualsg_refined
```

如果服务器 conda 环境激活路径和脚本里的不一致，先改这些脚本首行：

- `fit_halfyear_num_with_meta.sh`
- `fit_oneyear_num_with_meta.sh`
- 所有 `fit_fusion*.sh`

## 3. 推荐执行顺序

建议分两阶段：

- 第一阶段先跑半年任务，确认趋势
- 第二阶段只有在半年任务里看到价值时，再复制到一年任务

### 3.1 第一阶段：半年任务

#### Step 1. 跑数值 baseline

```bash
bash ./fit_halfyear_num_with_meta.sh
```

#### Step 2. 记录 baseline 对应 checkpoint

```bash
export NUM_CKPT="$(ls -td ./model_checkpoints/fit_halfyear_num_with_meta_* | head -1)/checkpoint.pth"
echo "$NUM_CKPT"
```

#### Step 3. 跑 direct from scratch

```bash
bash ./fit_fusion_halfyear_direct.sh
```

#### Step 4. 跑 direct from ckpt + unfreeze

```bash
bash ./fit_fusion_halfyear_direct_from_ckpt.sh
```

#### Step 5. 跑 residual + freeze

```bash
bash ./fit_fusion_halfyear_residual.sh
```

#### Step 6. 跑 residual + unfreeze

```bash
bash ./fit_fusion_halfyear_residual_unfreeze.sh
```

### 3.2 第二阶段：一年任务

只有在半年任务里至少有 1 组接近或超过 baseline，再继续一年任务。

#### Step 1. 跑一年 baseline

```bash
bash ./fit_oneyear_num_with_meta.sh
```

#### Step 2. 记录一年 checkpoint

```bash
export NUM_CKPT="$(ls -td ./model_checkpoints/fit_oneyear_num_with_meta_* | head -1)/checkpoint.pth"
echo "$NUM_CKPT"
```

#### Step 3. 跑一年 direct from scratch

```bash
bash ./fit_fusion_oneyear_direct.sh
```

#### Step 4. 跑一年 direct from ckpt + unfreeze

```bash
bash ./fit_fusion_oneyear_direct_from_ckpt.sh
```

#### Step 5. 跑一年 residual + freeze

```bash
bash ./fit_fusion_oneyear_residual.sh
```

#### Step 6. 跑一年 residual + unfreeze

```bash
bash ./fit_fusion_oneyear_residual_unfreeze.sh
```

## 4. 当前 4 个关键 fusion 组合

半年任务优先关注这 4 组：

| 组别 | 脚本 | 初始化 | 数值流是否冻结 | 说明 |
|---|---|---|---|---|
| A | `fit_fusion_halfyear_direct.sh` | 从头训练 | 否 | 当前 direct 基线 |
| B | `fit_fusion_halfyear_direct_from_ckpt.sh` | 加载数值 ckpt | 否 | 更公平的 direct 对照 |
| C | `fit_fusion_halfyear_residual.sh` | 加载数值 ckpt | 是 | 纯纠偏思路 |
| D | `fit_fusion_halfyear_residual_unfreeze.sh` | 加载数值 ckpt | 否 | 纠偏 + 联合微调 |

## 5. 当前脚本的默认超参

当前 FIT fusion 脚本默认：

- `train_epochs = 100`
- `batch_size = 200`
- `patience = 30`
- `fusion_optimizer_mode = split`
- `learning_rate = 0.001`
- `lr_num = 0.0001`
- `lr_text = 0.0005`
- `adjust = 0`
- `fit_scaler_mode = train_only`

这意味着当前正式推荐设置是：

- 不启用 scheduler
- 全程使用 split 学习率

同时，脚本现在支持环境变量覆盖，例如：

```bash
TRAIN_EPOCHS=20 PATIENCE=20 bash ./fit_fusion_halfyear_direct.sh
```

当前支持覆盖的常用变量：

- `TRAIN_EPOCHS`
- `BATCH_SIZE`
- `PATIENCE`
- `ADJUST`
- `LEARNING_RATE`
- `LR_NUM`
- `LR_TEXT`
- `CAPTION_EMB_PATH`
- `DATA_PATH`

## 6. 每次实验后看哪里

checkpoint：

```bash
./model_checkpoints/<setting>/checkpoint.pth
```

结果文件：

```bash
./model_outputs/<setting>/results/result.txt
```

如果想快速查看最近一次结果目录：

```bash
ls -td ./model_outputs/* | head -1
```

如果想快速查看最近一次结果：

```bash
cat "$(ls -td ./model_outputs/*/results | head -1)/result.txt"
```

## 7. 结果记录模板

建议每轮直接把结果补进下面这张表。

### 7.1 半年任务记录表

| setting | script | init | freeze_num | opt_mode | adjust | epochs | batch | MAE | MSE | RMSE | MAPE | WAPE | 备注 |
|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| fit_halfyear_num_with_meta | `fit_halfyear_num_with_meta.sh` | baseline | - | unified | 默认 | 20 | 按脚本 |  |  |  |  |  |  |
| fit_fusion_halfyear_joint_direct | `fit_fusion_halfyear_direct.sh` | scratch | no | split | 0 | 20 | 200 |  |  |  |  |  |  |
| fit_fusion_halfyear_direct_from_ckpt | `fit_fusion_halfyear_direct_from_ckpt.sh` | num_ckpt | no | split | 0 | 20 | 200 |  |  |  |  |  |  |
| fit_fusion_halfyear_residual_correction | `fit_fusion_halfyear_residual.sh` | num_ckpt | yes | split | 0 | 20 | 200 |  |  |  |  |  |  |
| fit_fusion_halfyear_residual_unfreeze | `fit_fusion_halfyear_residual_unfreeze.sh` | num_ckpt | no | split | 0 | 20 | 200 |  |  |  |  |  |  |

### 7.2 一年任务记录表

| setting | script | init | freeze_num | opt_mode | adjust | epochs | batch | MAE | MSE | RMSE | MAPE | WAPE | 备注 |
|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| fit_oneyear_num_with_meta | `fit_oneyear_num_with_meta.sh` | baseline | - | unified | 默认 | 20 | 按脚本 |  |  |  |  |  |  |
| fit_fusion_oneyear_joint_direct | `fit_fusion_oneyear_direct.sh` | scratch | no | split | 0 | 20 | 200 |  |  |  |  |  |  |
| fit_fusion_oneyear_direct_from_ckpt | `fit_fusion_oneyear_direct_from_ckpt.sh` | num_ckpt | no | split | 0 | 20 | 200 |  |  |  |  |  |  |
| fit_fusion_oneyear_residual_correction | `fit_fusion_oneyear_residual.sh` | num_ckpt | yes | split | 0 | 20 | 200 |  |  |  |  |  |  |
| fit_fusion_oneyear_residual_unfreeze | `fit_fusion_oneyear_residual_unfreeze.sh` | num_ckpt | no | split | 0 | 20 | 200 |  |  |  |  |  |  |

## 8. 结果判断规则

优先看：

1. `MSE`
2. `RMSE`
3. `MAE`
4. `WAPE`

当前建议的判断方式：

- 如果 4 个 fusion 组合全部明显差于 `fit_num_with_meta`
  - 暂停继续大规模跑一年任务
  - 回来继续改 fusion 结构
- 如果有 1 到 2 组接近 baseline
  - 先继续跑一年任务验证稳定性
- 如果有组合稳定优于 baseline
  - 再考虑做更细的超参搜索

## 9. 额外建议

这轮先不要同时改太多东西。

建议保持：

- 文本描述生成方式不变
- `caption_emb` 不重算
- 不改数值主干
- 先只比较当前这 4 个融合组合

补充说明：

- 当前训练代码仍然读取 `FIT_DualSG/fit_dualsg_all.json`
- 还没有接你新整理的 `dataset/FIT_DualSG/pt/fit_dualsg_all.pt`
- 如果后面要切到 dataset pt 直接加载，需要先确认 `.pt` 的数据结构与当前 json 契约一致

如果这轮依然全线不提升，再回头考虑：

- 给 direct 加更强的 `y_num` skip
- 对 `caption_emb` 做归一化或轻量投影约束
- 回头检查文本描述本身是否有效
