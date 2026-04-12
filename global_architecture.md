# FIT Architecture

最后更新：2026-04-12

## 1. 文档定位

这份文档只描述当前仓库里 FIT 相关的真实已实现结构。

配套文档：
- `fit_refactor_worklog.md`：本轮整理进度、真实 smoke 结果、剩余风险。
- `fit_fusion_redesign.md`：这次 fusion 重写背后的设计原则。

## 2. 当前 FIT 入口

统一入口：
- `run.py`

### 2.1 task -> experiment

- `fit_num` -> `exp/exp_fit_num.py`
- `fit_num_with_meta` -> `exp/exp_fit_num.py`
- `fit_fusion` -> `exp/exp_fit_fusion.py`

### 2.2 model

- `Model_Fit_Num` -> `models/model_fit_num.py`
- `Model_Fit_Num_With_Meta` -> `models/model_fit_num_with_meta.py`
- `Model_Fit_Fusion` -> `models/model_fit_fusion.py`

### 2.3 data

- `FIT_Meta` -> `data_provider/data_loader_fit_num_meta.py`
- `FIT_Fusion` -> `data_provider/data_loader_fit_fusion.py`

## 3. FIT 关键文件

### 3.1 数据层

- `data_provider/fit_dataset_utils.py`
- `data_provider/data_loader_fit_num_meta.py`
- `data_provider/data_loader_fit_fusion.py`
- `data_provider/data_factory.py`

### 3.2 实验层

- `exp/exp_fit_num.py`
- `exp/exp_fit_fusion.py`

### 3.3 模型层

- `models/model_fit_num.py`
- `models/model_fit_num_with_meta.py`
- `models/model_fit_fusion.py`

### 3.4 脚本

- `fit_halfyear_num.sh`
- `fit_halfyear_num_with_meta.sh`
- `fit_oneyear_num.sh`
- `fit_oneyear_num_with_meta.sh`
- `fit_fusion_halfyear_direct.sh`
- `fit_fusion_halfyear_residual.sh`
- `fit_fusion_oneyear_direct.sh`
- `fit_fusion_oneyear_residual.sh`

## 4. FIT 数据层

### 4.1 当前原始样本

当前 FIT 原始 json 里会用到：
- `series`
- `target`
- `annotations`
- `metadata.group`
- `metadata.element`
- `metadata.norm`

### 4.2 FIT 公共数据工具

`data_provider/fit_dataset_utils.py` 当前统一负责：
- `CITY_MAP / GENDER_MAP / AGE_MAP`
- `group` 解析
- `element_map`
- train / val / test split
- 标准化
- 时间特征
- `caption_emb` 对齐
- 进程内缓存

### 4.3 当前 scaler 语义

当前支持：
- `fit_scaler_mode=train_only`
- `fit_scaler_mode=split_fit`

默认：
- `train_only`

含义：
- `train_only`
  - 只用 train split 拟合 scaler
  - val/test 共享这一套 scaler
- `split_fit`
  - 保留旧行为
  - train/val/test 各自 fit 各自 split

### 4.4 本地 smoke 截断

当前支持：
- `max_train_samples`
- `max_val_samples`
- `max_test_samples`

约定：
- `<= 0` 表示不截断
- 截断发生在正式 split 之后
- 默认不会影响服务器正式训练

## 5. FIT 数值流

### 5.1 fit_num

链路：

`run.py`
-> `Exp_Fit_Num`
-> `Dataset_DualSG_Fit_Num_Meta`
-> `Model_Fit_Num`

特点：
- 只消费数值序列
- dataset 仍返回 meta tuple，但模型本身不使用 meta

### 5.2 fit_num_with_meta

链路：

`run.py`
-> `Exp_Fit_Num`
-> `Dataset_DualSG_Fit_Num_Meta`
-> `Model_Fit_Num_With_Meta`

特点：
- 使用数值序列 + `city/gender/age/element`
- 也是 FIT fusion 当前复用的数值 backbone

### 5.3 Model_Fit_Num_With_Meta

当前保持原有训练语义不变：
1. Instance Norm
2. Patch Embedding
3. 融合 group / element 元数据
4. AIM
5. Transformer Encoder
6. Head 输出未来序列
7. De-Norm

本轮新增：
- `extract_features()`

返回：
- `forecast`
- `encoded_tokens`
- `summary_state`

这个接口只给 fusion 调用，不改变数值 baseline 的训练方式。

## 6. FIT fusion

### 6.1 当前定位

当前 FIT fusion 仍然继续使用预处理好的 `caption_emb`，不在训练时接 raw text encoder。

文本侧流程仍然是：
1. 先生成文本描述
2. 再离线编码成 `caption_emb`
3. 训练时直接读取 `caption_emb`

### 6.2 当前数值 backbone 复用方式

FIT fusion 当前复用：
- `Model_Fit_Num_With_Meta`

复用方式已经不是简单调它的 `forward()`，而是显式调用：
- `extract_features()`

这样 fusion 能稳定拿到：
- 数值预测 `y_num`
- 中间特征 `encoded_tokens`
- 数值摘要 `summary_state`

### 6.3 当前 fusion 主体结构

当前 fusion 的核心不是“文本单独再预测一下”，而是“文本先调制数值中间特征”。

结构：
- 文本适配器：`caption_emb -> text_adapter`
- 文本生成条件参数：
  - `gamma`
  - `beta`
- 数值中间特征：
  - `encoded_tokens`
- 条件调制：
  - `encoded_tokens * (1 + gamma) + beta`

之后再从下列信息构造融合上下文：
- 文本上下文
- 调制后的数值摘要
- 原始数值预测摘要 `y_num`
- 历史统计特征

### 6.4 历史统计特征

当前用到：
- `last`
- `mean`
- `std`
- `slope`
- `range`

这些特征直接从原始 `x_enc` 提取。

### 6.5 direct 模式

参数：
- `text_mode=direct`

当前语义：
- 文本先调制数值 latent
- 再从融合后的上下文直接输出最终未来序列

当前已经不再使用旧版的：
- `(1 - w) * y_num + w * y_text`

也就是说，direct 不再是浅层加权融合，而是直接的条件化预测。

### 6.6 residual 模式

参数：
- `text_mode=residual`

当前语义：
- 先得到数值 backbone 的主预测 `y_num`
- 文本分支只负责生成一个“有边界的纠偏量”

当前 residual 的约束方式：
- 不再使用 `beta_delta`
- 不再使用 `force_gain`
- 直接用历史 `std + range` 来限制纠偏幅度

形式上可以理解为：
- 先预测 `raw_delta`
- 再通过 `tanh(raw_delta)` 限制方向和幅度
- 再乘以由历史波动决定的半径

最终：
- `y_final = y_num + delta`

### 6.7 freeze / joint

当前 FIT fusion 支持：
- 从头 joint 训练
- 从数值 ckpt 启动
- 冻结数值流
- 不冻结数值流联合更新

关键参数：
- `num_model_path`
- `freeze_numerical`
- `text_mode`
- `fusion_optimizer_mode`

## 7. FIT fusion 优化器

### 7.1 unified

参数：
- `fusion_optimizer_mode=unified`

语义：
- 所有可训练参数共用一个 `learning_rate`

### 7.2 split

参数：
- `fusion_optimizer_mode=split`

语义：
- 数值 backbone 参数使用 `lr_num`
- 文本 / 融合头参数使用 `lr_text`
- 文本分支可单独设 `weight_decay_text`

## 8. 当前官方脚本语义

### 8.1 数值流

- `fit_halfyear_num.sh`
- `fit_halfyear_num_with_meta.sh`
- `fit_oneyear_num.sh`
- `fit_oneyear_num_with_meta.sh`

这些脚本都已经显式带上：
- `fit_scaler_mode=train_only`

### 8.2 joint_direct

脚本：
- `fit_fusion_halfyear_direct.sh`
- `fit_fusion_oneyear_direct.sh`

语义：
- 从头 joint 训练
- `text_mode=direct`
- 默认不加载数值 ckpt
- 默认不冻结数值流

### 8.3 residual_correction

脚本：
- `fit_fusion_halfyear_residual.sh`
- `fit_fusion_oneyear_residual.sh`

语义：
- 从数值 ckpt 启动
- `text_mode=residual`
- 默认冻结数值流

说明：
- 通过 shell 变量 `NUM_CKPT` 显式指定数值 checkpoint

## 9. 当前 FIT 已经不再使用的旧 fusion 项

FIT 当前实现已经不再依赖：
- `beta_delta`
- `force_gain`
- 文本侧写死输入维度 `llm_dim`

注意：
- `run.py` 中仍保留 `llm_dim / delta_scale / use_vol_prior / force_gain`
- 这是为了 Geo 兼容
- 它们当前不再参与 FIT fusion

## 10. 本地验证状态

### 10.1 已装依赖

当前 base 环境已装：
- `torch` CPU
- `numpy`
- `scipy`
- `scikit-learn`
- `pandas`
- `matplotlib`
- `tqdm`
- `einops`
- `reformer_pytorch`

### 10.2 已通过真实 smoke

已真实跑通：
- `fit_num`
- `fit_num_with_meta`
- `fit_num_with_meta` 重跑验证
- 新版 `fit_fusion direct + split`
- 新版 `fit_fusion residual + split`
- 新版 `fit_fusion direct + unified`

说明：
- 正式 `fit_caption_emb_all.pt` 当前不在本地
- 为了验证新版 fusion 代码路径，本轮用过临时小子集和临时 `caption_emb`
- 临时资产只用于 smoke，后面已删除

## 11. 当前已知限制

### 11.1 正式提升还要靠服务器实验验证

当前已经确认：
- 代码路径通
- 新 direct / residual 通
- split / unified 通

但仍未确认：
- 新结构在正式 `caption_emb` 和全量训练上是否一定优于旧结构

### 11.2 文本上限仍由描述质量决定

即使融合结构更合理，文本分支上限仍然受以下因素影响：
- 文本描述是否真的捕捉了趋势模式
- 文本描述是否和数值走势强相关
- 预处理 embedding 是否足够表达这些描述
