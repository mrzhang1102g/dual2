# FIT 架构文档

最后更新：2026-04-14

## 文档定位

这份文档只描述当前仓库里 FIT 相关代码的真实实现，以及当前阶段 legacy 恢复诊断的入口。

配套文档：

- [fit_server_runbook.md](/D:/zhangjing/project/Dualsg_refined/md/fit_server_runbook.md)
  当前实验结果、服务器执行记录、legacy 诊断脚本
- [fit_refactor_worklog.md](/D:/zhangjing/project/Dualsg_refined/md/fit_refactor_worklog.md)
  本轮整理、重构、当前阶段结论
- [项目文件结构.md](/D:/zhangjing/project/Dualsg_refined/md/项目文件结构.md)
  当前项目文件树

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

## FIT 关键文件

数据层：

- `data_provider/fit_dataset_utils.py`
- `data_provider/data_loader_fit_num_meta.py`
- `data_provider/data_loader_fit_fusion.py`
- `data_provider/data_factory.py`

实验层：

- `exp/exp_fit_num.py`
- `exp/exp_fit_fusion.py`

模型层：

- `models/model_fit_num.py`
- `models/model_fit_num_with_meta.py`
- `models/model_fit_fusion.py`

## 数据层

当前 FIT 数值数据仍然读取：

- `dataset/FIT_DualSG/fit_dualsg_all.json`

当前文本输入仍然是离线 embedding：

- `caption_emb_path -> *.pt`

`fit_dataset_utils.py` 统一负责：

- `CITY_MAP / GENDER_MAP / AGE_MAP`
- `group` 解析
- `element_map`
- train / val / test 切分
- 标准化
- 时间特征
- `caption_emb` 对齐
- 进程内缓存

当前默认 scaler 语义：

- `fit_scaler_mode=train_only`

含义：

- 只用 train split 拟合 scaler
- val / test 复用同一个 train-fit scaler

## 数值流

### `fit_num`

链路：

`run.py` -> `Exp_Fit_Num` -> `Dataset_DualSG_Fit_Num_Meta` -> `Model_Fit_Num`

### `fit_num_with_meta`

链路：

`run.py` -> `Exp_Fit_Num` -> `Dataset_DualSG_Fit_Num_Meta` -> `Model_Fit_Num_With_Meta`

这是当前 FIT 最强的数值 baseline，也是 fusion 当前复用的数值 backbone。

### `Model_Fit_Num_With_Meta`

当前保持原有训练语义不变，只额外提供：

- `extract_features()`

返回：

- `forecast`
- `encoded_tokens`
- `summary_state`

这个接口只给 fusion 读取中间特征，不改变数值 baseline 的训练方式。

## FIT Fusion

### 总体接口

当前 FIT fusion 统一通过：

- `Model_Fit_Fusion`

对外关键参数：

- `fusion_version = modern / legacy`
- `text_mode = direct / residual`
- `num_model_path`
- `freeze_numerical`
- `disable_text`
- `fusion_optimizer_mode = unified / split`

### 数值 backbone 复用方式

FIT fusion 当前统一复用：

- `Model_Fit_Num_With_Meta`

并通过：

- `extract_features()`

拿到：

- 数值预测 `y_num`
- 中间特征 `encoded_tokens`
- 数值摘要 `summary_state`

### `fusion_version = modern`

当前新版 fusion 的核心思路是：

- 文本先调制数值 backbone 的中间特征
- 再从共享上下文输出 `direct / residual`

核心链路：

- `caption_emb -> text_adapter`
- 文本生成 `gamma / beta`
- `gamma / beta` 调制 `encoded_tokens`
- 再把：
  - 文本上下文
  - 调制后的数值摘要
  - 原始数值预测 `y_num` 的摘要
  - 历史统计特征
  拼成共享上下文

#### modern direct

- `text_mode=direct`
- 从共享上下文直接输出最终未来序列

注意：

- 当前新版 `direct` 没有旧版那种显式 `y_num` 硬 skip
- 这也是新版更容易“绕开文本”的重要原因之一

#### modern residual

- `text_mode=residual`
- 先得到数值主预测 `y_num`
- 再输出一个有边界的纠偏量 `delta`
- 最终：
  - `y_final = y_num + delta`

### `fusion_version = legacy`

当前已恢复 legacy 头，但仍运行在现在这套清理后的训练框架里。

也就是说：

- 用的是当前的 data loader
- 用的是当前的 train-only scaler
- 用的是当前的 `Exp_Fit_Fusion`
- 只是把 fusion 头切回旧思路

#### legacy direct

旧版显式数值 skip：

- `caption_emb -> y_text`
- `y_final = (1 - w) * y_num + w * y_text`

这里的 `w` 继续支持：

- `direct_w_mode = learned / fixed`
- `direct_w_fixed`

legacy direct 的关键特点是：

- 文本分支单独生成一条预测
- 数值主预测 `y_num` 直接参与最终输出
- 因此比新版 direct 更保守、更稳定

#### legacy residual

旧版纠偏路径：

- 输入为 `[caption_emb, phi(y_num.detach()), vol]`
- 输出 `delta_y` 和 `gain`
- 最终：
  - `y_final = y_num + gain * delta_y`

legacy residual 的关键特点是：

- 数值主预测始终保留
- 文本主要承担纠偏角色
- `phi(y_num)` 继续 `detach`
- `delta_scale / force_gain / use_vol_prior` 只在这条分支里参与

### `disable_text`

参数：

- `--disable_text`

当前两种 `fusion_version` 的语义已经统一：

- loader 仍然会读入 `caption_emb`
- 但模型会直接返回数值预测 `y_num`
- 并冻结所有非数值参数

所以：

- `direct + disable_text`
- `residual + disable_text`

在当前实现里本质上是同一个纯数值对照。

## 当前实验结论

当前 modern 版本已经被以下对照基本坐实：

- 真实长文本
- random text
- filler text
- disable_text

结果几乎一样。

因此当前最准确的结论是：

- 当前 modern fusion 的主要增益不是来自文本语义
- 更像来自：
  - `num_ckpt + unfreeze`
  - `residual` 的硬数值 skip
  - 数值侧辅助特征本身

换句话说：

- current modern 结构允许模型几乎完全忽略文本

## 当前阶段的重点

当前不再优先继续堆更多文本版本。

当前最有价值的事情是：

- 在相同训练框架下比较 `modern` 和 `legacy`

首轮只跑：

- `legacy direct + ckpt + unfreeze + real long text`
- `legacy residual + ckpt + unfreeze + real long text`

如果 legacy 能明显追回旧版优势，再补 `random / filler` 对照。
