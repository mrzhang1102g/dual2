# FIT 架构文档

最后更新：2026-04-19

## 文档定位

这份文档描述当前 `0417` 分支里 FIT 的 active 实现和实验主线。

历史阶段：
- `0411`：embedding-fusion 诊断与归档
- `0414`：`v2-v5` residual/direct 结构重写
- `0417-v6`：DualSG-style raw-text fusion 尝试，结果已证明不如强数值 baseline

历史脚本和结果快照仍然保留在：
- [D:\zhangjing\project\Dualsg_refined\md\fit_halfyear_0411_manifest.md](D:/zhangjing/project/Dualsg_refined/md/fit_halfyear_0411_manifest.md)
- [D:\zhangjing\project\Dualsg_refined\scripts_archive\fit_halfyear_0411\README.md](D:/zhangjing/project/Dualsg_refined/scripts_archive/fit_halfyear_0411/README.md)

## 当前主线

当前 FIT 的改进主线已经从“文本作为输入模态做融合”切换成：

- 数值主干继续负责预测
- 结构化文本只提供语义监督
- 用多任务学习约束数值表示学习

当前 active 任务：
- `fit_num`
- `fit_num_with_meta`
- `fit_num_with_meta_semantic`
- `fit_fusion`

其中当前最值得继续推进的是：
- `fit_num_with_meta_semantic`

## 数据契约

### 纯数值 / 数值+Meta

仍然使用：
- `dataset/FIT_DualSG/fit_dualsg_all.json`

### 语义监督

默认使用：
- `dataset/FIT_DualSG/fit_dualsg_structured.json`

说明：
- 这份文件当前不一定会提交到仓库，但脚本和代码默认按它的路径读取
- `annotations` 应该是结构化趋势描述，推荐是 json 字符串或字典

当前语义解析器支持从 `annotations` 中提取 4 类监督标签：
- `overall_trend`
- `recent_regime` 或 `recent_trend`
- `volatility`
- `major_turning_points`

即使只有部分字段，代码也会通过 `mask` 机制自动跳过缺失标签。

## 标准化

当前 FIT 默认使用：
- `fit_scaler_mode=train_only`

含义：
- 只用 train split 拟合 scaler
- val / test 复用 train-fit scaler

## 数值主干

### `fit_num`

链路：
- `run.py`
- `exp/exp_fit_num.py`
- `data_provider/data_loader_fit_num_meta.py`
- `models/model_fit_num.py`

### `fit_num_with_meta`

链路：
- `run.py`
- `exp/exp_fit_num.py`
- `data_provider/data_loader_fit_num_meta.py`
- `models/model_fit_num_with_meta.py`

这是当前 FIT 最强的纯数值 baseline。

## 0417 语义监督主线

### 目标

不再把文本作为输入模态去和强数值 backbone 做 late fusion，而是把结构化文本转成语义标签，监督数值模型学习更有判别力的趋势表示。

### 当前任务

任务名：
- `fit_num_with_meta_semantic`

数据集：
- `FIT_Meta_Semantic`

模型：
- `Model_Fit_Num_With_Meta_Semantic`

训练类：
- `exp/exp_fit_num_semantic.py`

### 模型结构

数值主干完全复用：
- [D:\zhangjing\project\Dualsg_refined\models\model_fit_num_with_meta.py](D:/zhangjing/project/Dualsg_refined/models/model_fit_num_with_meta.py)

新模型只是在 `summary_state` 上增加一个很薄的语义监督头：
- `semantic_trunk`
- `overall_head`
- `recent_head`
- `volatility_head`
- `turning_head`

主预测输出仍然是：
- `forecast`

辅助输出是：
- `semantic_logits["overall"]`
- `semantic_logits["recent"]`
- `semantic_logits["volatility"]`
- `semantic_logits["turning"]`

### 语义任务定义

当前 4 个辅助任务：
- `overall_trend`：3 类，`falling / stable / rising`
- `recent_regime`：3 类，`falling / stable / rising`
- `volatility`：3 类，`low / moderate / high`
- `major_turning_points`：4 类，`none / early / middle / late`

### 损失函数

总损失：
- `L = L_forecast + λ * L_semantic`

其中：
- `L_forecast`：沿用原数值任务损失，默认 `MAE`
- `L_semantic`：4 个分类任务交叉熵的均值
- `λ = semantic_loss_weight`

缺失标签通过 `mask` 跳过，不会强行参与损失。

### 主干初始化

当前支持：
- `pretrained_num_model_path`

用途：
- 先加载强数值 baseline 的 checkpoint
- 再在其基础上增加语义监督进行微调

这比完全从零训练更符合当前项目实际情况。

## 0417 v6 raw-text fusion

`v6` 仍保留在代码里作为历史尝试，但已经不是推荐主线。

核心思路：
- 直接从 json 里读取 raw text
- 冻结 GPT2
- pooling 后投影到预测空间
- 与 `y_num` 做 forecast-space 融合

当前结论：
- 这条路线结果明显差于强数值 baseline 和 `direct_v3`
- 不建议继续在这条路线上投入主要精力

## 当前 active 脚本

当前和 0417 新方向直接相关的脚本：
- `fit_halfyear_num_with_meta.sh`
- `fit_halfyear_num_with_meta_semantic_v1.sh`
- `fit_fusion_halfyear_v6.sh`

其中现在更推荐优先跑：
- `fit_halfyear_num_with_meta_semantic_v1.sh`
