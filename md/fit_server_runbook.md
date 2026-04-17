# FIT 鏈嶅姟鍣ㄦ墽琛屾墜鍐?
鏈€鍚庢洿鏂帮細2026-04-17

## 鏂囨。瀹氫綅

杩欎唤鏂囨。鍚屾椂鎵挎媴涓や欢浜嬶細

1. 璁板綍 `0411/0414` 闃舵宸茬粡楠岃瘉杩囩殑鍏抽敭缁撹
2. 璁板綍 `0417 v6` 鐨勫綋鍓?active 鎵ц鍏ュ彛

閰嶅鏂囨。锛?
- [fit_halfyear_0411_manifest.md](/D:/zhangjing/project/Dualsg_refined/md/fit_halfyear_0411_manifest.md)
- [fit_refactor_worklog.md](/D:/zhangjing/project/Dualsg_refined/md/fit_refactor_worklog.md)
- [global_architecture.md](/D:/zhangjing/project/Dualsg_refined/md/global_architecture.md)

## 缁撴灉瑙ｈ浼樺厛绾?
褰撳墠 FIT half-year 缁熶竴鎸変笅闈㈤『搴忕湅锛?
1. `MAE`
2. `MAPE`
3. `WAPE`
4. `MSE`

## 宸茬‘璁ょ殑鍘嗗彶缁撹

### baseline

| setting | MAE | MSE | RMSE | MAPE | WAPE |
|---|---:|---:|---:|---:|---:|
| `fit_num_with_meta` | 0.085858 | 0.013315 | 0.115391 | 30.23% | 18.52% |

### 0411 鍘嗗彶鏈€濂借瀺鍚堢粨鏋?
| setting | MAE | MSE | RMSE | MAPE | WAPE |
|---|---:|---:|---:|---:|---:|
| `legacy direct + ckpt + unfreeze (100)` | 0.075869 | 0.010493 | 0.102437 | 27.84% | 16.37% |
| `legacy residual + ckpt + unfreeze (100)` | 0.076114 | 0.010665 | 0.103272 | 27.12% | 16.42% |

### 0411/0414 鐨勫叧閿垽鏂?
宸茬粡琚疄楠屽潗瀹炵殑鐐癸細

1. `emb.pt + 澶氱増 residual fusion` 杩欐潯璺嚎涓€鐩村彧鏈夊皬骞呭閲?2. `random / filler / disable_text` 涓庣湡瀹炴枃鏈樊璺濊繃灏?3. 褰撳墠鐡堕鏇村儚鏄€滄枃鏈浣曡浣跨敤鈥濓紝鑰屼笉鏄€滄枃鏈敱璋佺敓鎴愨€?4. `direct_v3` 鏄洰鍓嶆渶绋崇殑 direct 鍩虹嚎锛?   - `MAE = 0.079424`
   - `MSE = 0.011435`
   - `RMSE = 0.106934`
   - `MAPE = 29.03%`
   - `WAPE = 17.13%`
5. 鍒?`residual_v5` 涓烘锛宺esidual 瀹舵棌铏芥湁鏀硅繘锛屼絾浠嶆湭鍘嬭繃 `direct_v3`锛?   - `MAE = 0.079558`
   - `MSE = 0.011536`
   - `RMSE = 0.107406`
   - `MAPE = 28.90%`
   - `WAPE = 17.16%`

鍥犳 `0417` 宸茬粡鍋滄缁存姢 `emb.pt + v3/v4/v5` active 璺嚎銆?
## 0417 v6锛氬綋鍓?active 涓荤嚎

### 褰撳墠鑴氭湰

褰撳墠鏍圭洰褰曞敮涓€鐨?FIT half-year active fusion 鍏ュ彛锛?
- `fit_fusion_halfyear_v6.sh`

### 褰撳墠閰嶇疆

`v6` 鍥哄畾涓猴細

- FIT half-year
- `num_model_path = ./model_checkpoints/fit_halfyear_num_with_meta_20260413_083428/checkpoint.pth`
- `train_epochs = 20`
- `patience = 100`
- `adjust = 0`
- `fusion_optimizer_mode = split`
- `fit_scaler_mode = train_only`
- `text_model_path = ./weights/gpt2`
- `text_model_type = gpt2`
- `text_field = annotations`
- `text_pool_type = avg`
- `text_max_length = 256`

### 璁捐璇箟

`v6` 涓嶅啀璇诲彇 `caption_emb.pt`銆?
瀹冪殑 active 璇箟鏄細

1. 浠?`fit_dualsg_all.json` 璇诲彇 raw text
2. raw text 鍦ㄧ嚎杩涘叆鍐荤粨 GPT2
3. 瀵?token hidden 鍋?pooling
4. 鏂囨湰琛ㄧず鎶曞奖鍒伴娴嬬┖闂达紝寰楀埌 `y_text`
5. 涓庢暟鍊奸娴?`y_num` 鐩存帴鍦?forecast space 铻嶅悎

鍗筹細

- `y_final = (1 - w_t) * y_num + w_t * y_text`

### 绗竴杞墽琛?
鐩存帴杩愯锛?
```bash
bash ./fit_fusion_halfyear_v6.sh
```

### 绗簩杞帶鍒跺疄楠岀殑瑙﹀彂鏉′欢

鍙湁褰?`v6` 棣栬疆缁撴灉鑷冲皯涓嶅樊浜?`direct_v3` 鏃讹紝鍐嶇户缁ˉ锛?
- `disable_text`
- `random text`
- `filler text`

鍦?`v6` 棣栬疆缁撴灉鍑烘潵涔嬪墠锛屼笉寤鸿鎻愬墠閾烘洿澶氭帶鍒跺疄楠屻€?
