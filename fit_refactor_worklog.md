# FIT Refactor Worklog

最后更新：2026-04-12

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
