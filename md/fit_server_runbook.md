# FIT 鏈嶅姟鍣ㄦ墽琛屾墜鍐?
鏈€鍚庢洿鏂帮細2026-04-19

## 鐩爣

褰撳墠 0417 鍒嗘敮涓嶅啀浼樺厛鎺ㄨ繘 embedding-fusion锛岃€屾槸杞悜锛?
- 寮烘暟鍊间富骞?- 缁撴瀯鍖栨枃鏈涔夌洃鐫?- 澶氫换鍔″涔?
## 鍘嗗彶缁撹

### baseline

| setting | MAE | MSE | RMSE | MAPE | WAPE |
|---|---:|---:|---:|---:|---:|
| `fit_num_with_meta` | 0.085858 | 0.013315 | 0.115391 | 30.23% | 18.52% |

### 鍘嗗彶鏈€浣宠瀺鍚堢粨鏋?
| setting | MAE | MSE | RMSE | MAPE | WAPE |
|---|---:|---:|---:|---:|---:|
| `legacy direct + ckpt + unfreeze (100)` | 0.075869 | 0.010493 | 0.102437 | 27.84% | 16.37% |
| `legacy residual + ckpt + unfreeze (100)` | 0.076114 | 0.010665 | 0.103272 | 27.12% | 16.42% |

### 0414 鍏抽敭缁撹

- `direct_v3` 鏄洰鍓嶆渶绋崇殑 direct baseline
  - `MAE 0.079424`
  - `MAPE 29.03%`
  - `WAPE 17.13%`
- residual 瀹舵棌鐩村埌 `v5` 浠嶆湭鍦?`MAE/WAPE` 涓婂帇杩?`direct_v3`
- `random / filler / disable_text` 涓庣湡瀹炴枃鏈樊璺濆緢灏忥紝璇存槑鏃х殑 embedding-fusion 璺嚎娌℃湁鐪熸鍧愬疄鏂囨湰璇箟澧炵泭

### 0417 v6 缁撹

- DualSG-style raw-text fusion 宸插皾璇?- 缁撴灉杩涗竴姝ュ彉宸細
  - `MAE 0.082309`
  - `MAPE 31.14%`
  - `WAPE 17.76%`
- 褰撳墠涓嶅缓璁户缁部杩欐潯 raw-text fusion 璺嚎鍋氬ぇ閲忓疄楠?
## 褰撳墠鎺ㄨ崘涓荤嚎锛歋emantic Supervision

### 鑴氭湰

褰撳墠鎺ㄨ崘浼樺厛璺戯細
- `fit_halfyear_num_with_meta_semantic_v1.sh`

### 渚濊禆鏁版嵁

榛樿鑴氭湰浣跨敤锛?- `./dataset/FIT_DualSG/fit_dualsg_structured.json`

娉ㄦ剰锛?- 杩欎唤鏂囦欢闇€瑕佸湪鏈嶅姟鍣ㄤ笂瀛樺湪
- 褰撳墠浠撳簱閲屼笉涓€瀹氬寘鍚畠

### 褰撳墠鑴氭湰閰嶇疆

- `task_name = fit_num_with_meta_semantic`
- `model = Model_Fit_Num_With_Meta_Semantic`
- `data = FIT_Meta_Semantic`
- `seq_len = 48`
- `pred_len = 12`
- `train_epochs = 20`
- `batch_size = 200`
- `patience = 100`
- `fit_scaler_mode = train_only`
- `semantic_text_field = annotations`
- `semantic_hidden = 64`
- `semantic_dropout = 0.1`
- `semantic_loss_weight = 0.2`
- `pretrained_num_model_path = ./model_checkpoints/fit_halfyear_num_with_meta_20260413_083428/checkpoint.pth`

### 杩愯鏂瑰紡

```bash
bash ./fit_halfyear_num_with_meta_semantic_v1.sh
```

## 缁撴灉璁板綍妯℃澘

### 0417 semantic supervision

| setting | MAE | MSE | RMSE | MAPE | WAPE | note |
|---|---:|---:|---:|---:|---:|---|
| `fit_halfyear_num_with_meta_semantic_v1` |  |  |  |  |  |  |

## 瑙ｈ椤哄簭

褰撳墠 FIT half-year 缁熶竴鎸変笅闈㈤『搴忕湅锛?1. `MAE`
2. `MAPE`
3. `WAPE`
4. `MSE`

濡傛灉 semantic 鐗堟湰鑳界ǔ瀹氫紭浜?`fit_num_with_meta`锛屽啀鍐冲畾鏄惁锛?- 璋?`semantic_loss_weight`
- 鍙繚鐣欓儴鍒嗚涔変换鍔?- 鎹笉鍚?structured view 鏁版嵁鏂囦欢

