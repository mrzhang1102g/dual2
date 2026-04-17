# FIT 鏋舵瀯鏂囨。

鏈€鍚庢洿鏂帮細2026-04-17

## 鏂囨。瀹氫綅

杩欎唤鏂囨。鍙弿杩板綋鍓?`0417` 鍒嗘敮閲?FIT 鐨?active 瀹炵幇銆?
鍘嗗彶闃舵璇存槑锛?
- `0411`锛氳瘖鏂笌褰掓。鍒嗘敮锛岄獙璇佷簡 `random / filler / disable_text` 瀵规棫 fusion 鐨勫奖鍝?- `0414`锛歚v2-v5` 缁撴瀯閲嶅啓鍒嗘敮锛岄獙璇佷簡澶氱増 residual 鏀归€犵殑涓婇檺

鏃у疄楠屽拰鑴氭湰蹇収浠嶄繚鐣欏湪锛?
- [fit_halfyear_0411_manifest.md](/D:/zhangjing/project/Dualsg_refined/md/fit_halfyear_0411_manifest.md)
- [scripts_archive/fit_halfyear_0411/README.md](/D:/zhangjing/project/Dualsg_refined/scripts_archive/fit_halfyear_0411/README.md)

閰嶅鏂囨。锛?
- [fit_server_runbook.md](/D:/zhangjing/project/Dualsg_refined/md/fit_server_runbook.md)
- [fit_refactor_worklog.md](/D:/zhangjing/project/Dualsg_refined/md/fit_refactor_worklog.md)
- [椤圭洰鏂囦欢缁撴瀯.md](/D:/zhangjing/project/Dualsg_refined/md/椤圭洰鏂囦欢缁撴瀯.md)

## 褰撳墠鍏ュ彛

缁熶竴鍏ュ彛锛?
- `run.py`

浠诲姟鍒板疄楠岀被锛?
- `fit_num` -> `exp/exp_fit_num.py`
- `fit_num_with_meta` -> `exp/exp_fit_num.py`
- `fit_fusion` -> `exp/exp_fit_fusion.py`

浠诲姟鍒版ā鍨嬶細

- `Model_Fit_Num` -> `models/model_fit_num.py`
- `Model_Fit_Num_With_Meta` -> `models/model_fit_num_with_meta.py`
- `Model_Fit_Fusion` -> `models/model_fit_fusion.py`

浠诲姟鍒版暟鎹泦锛?
- `FIT_Meta` -> `data_provider/data_loader_fit_num_meta.py`
- `FIT_Fusion` -> `data_provider/data_loader_fit_fusion.py`

## FIT 鏁版嵁濂戠害

褰撳墠 FIT 鏁板€兼暟鎹鍙栵細

- `dataset/FIT_DualSG/fit_dualsg_all.json`

褰撳墠 `fit_fusion` 鏂囨湰鏁版嵁涔熺洿鎺ユ潵鑷悓涓€涓?json锛?
- 瀛楁榛樿浣跨敤 `annotations`

涔熷氨鏄锛?
- `json` 鍐冲畾鏍锋湰椤哄簭銆佹暟鍊煎簭鍒椼€佺洰鏍囧€煎拰鍏冩暟鎹?- `fit_fusion` 涓嶅啀璇?`caption_emb.pt`

## 鏍囧噯鍖?
褰撳墠 FIT 榛樿浣跨敤锛?
- `fit_scaler_mode=train_only`

鍚箟锛?
- 鍙敤 train split 鎷熷悎 scaler
- val / test 澶嶇敤 train-fit scaler

## 鏁板€兼祦

### `fit_num`

閾捐矾锛?
`run.py` -> `Exp_Fit_Num` -> `Dataset_DualSG_Fit_Num_Meta` -> `Model_Fit_Num`

### `fit_num_with_meta`

閾捐矾锛?
`run.py` -> `Exp_Fit_Num` -> `Dataset_DualSG_Fit_Num_Meta` -> `Model_Fit_Num_With_Meta`

杩欐槸褰撳墠 FIT 鏈€寮虹殑鏁板€?baseline锛屼篃鏄?`fit_fusion` 褰撳墠缁х画澶嶇敤鐨?backbone銆?
### `Model_Fit_Num_With_Meta`

褰撳墠淇濇寔璁粌璇箟涓嶅彉锛屽彧棰濆鎻愪緵锛?
- `extract_features()`

杩斿洖锛?
- `forecast`
- `encoded_tokens`
- `summary_state`

鍏朵腑 `fit_fusion v6` 瀹為檯鍙渶瑕?`forecast`銆?
## 0417 FIT Fusion 涓荤嚎锛歷6

### 璁捐鐩爣

`v6` 涓嶅啀娌跨敤锛?
- 绂荤嚎 `caption_emb.pt`
- deep residual correction family
- `v3 / v4 / v5` 澶氬垎鏀?active 缁存姢

鑰屾槸鐩存帴璐磋繎 DualSG 鐨勬寮忛娴嬫柟寮忥細

1. 鏁板€间富骞插厛杈撳嚭 `y_num`
2. 鐩存帴璇诲彇 json 涓殑 raw caption 鏂囨湰
3. 鏂囨湰鍦ㄧ嚎閫佸叆鍐荤粨鐨勬湰鍦版枃鏈紪鐮佹ā鍨?4. 瀵?token hidden 鍋?pooling锛屽緱鍒版枃鏈〃绀?5. 鏂囨湰琛ㄧず鎶曞奖鍒伴娴嬬┖闂达紝寰楀埌 `y_text`
6. 鍦ㄩ娴嬬┖闂寸洿鎺ヨ瀺鍚堬細
   - `y_final = (1 - w_t) * y_num + w_t * y_text`

### 鏂囨湰娴?
褰撳墠 active 鏂囨湰娴侀厤缃細

- `text_model_type = gpt2`
- `text_model_path = ./weights/gpt2`
- `text_field = annotations`
- `text_pool_type = avg`
- `text_max_length = 256`

鏂囨湰缂栫爜妯″瀷锛?
- 浣跨敤鏈湴 GPT2 tokenizer + GPT2Model
- 鍏ㄩ儴鍐荤粨锛屼笉鍙備笌璁粌
- 鍙缁冿細
  - `caption_proj`
  - `fusion_weight`
  - 鏁板€间富骞蹭腑鏈喕缁撶殑鍙傛暟

### 铻嶅悎鏂瑰紡

褰撳墠 active 铻嶅悎鍙繚鐣欎竴鏉¤矾寰勶細

- `y_num`锛氭暟鍊间富骞茶緭鍑?- `y_text`锛歳aw text 缁忓喕缁撴枃鏈紪鐮佸櫒鍚庢姇褰卞緱鍒?- `w_t`锛氬舰鐘?`[pred_len, 1]` 鐨勫彲瀛︿範铻嶅悎鏉冮噸锛屽 batch 鍏变韩

鏈€缁堬細

- `y_final = (1 - w_t) * y_num + w_t * y_text`

杩欐槸 forecast-space fusion锛屼笉鍐嶅仛 latent-space 娣辫瀺鍚堛€?
### `disable_text`

褰撳墠浠嶄繚鐣欙細

- `--disable_text`

寮€鍚椂璇箟鍥哄畾涓猴細

- 鐩存帴杩斿洖 `y_num`
- 鍐荤粨鎵€鏈夐潪鏁板€煎弬鏁?
鍥犳瀹冧粛鐒舵槸鏈€骞插噣鐨勭函鏁板€煎鐓с€?
## 褰撳墠 active 鍙傛暟

褰撳墠 `fit_fusion` 鐪熸浣跨敤鐨勫弬鏁帮細

- `num_model_path`
- `freeze_numerical`
- `disable_text`
- `fusion_optimizer_mode`
- `text_model_path`
- `text_model_type`
- `text_field`
- `text_pool_type`
- `text_max_length`
- `lr_num`
- `lr_text`
- `weight_decay_text`
- `num_feat_dim`
- `fusion_hidden`
- `fusion_dropout`

## 褰撳墠 active 鑴氭湰

鏍圭洰褰曞綋鍓嶅彧淇濈暀涓€浠?FIT half-year active fusion 鑴氭湰锛?
- `fit_fusion_halfyear_v6.sh`

鏃х殑 `v2-v5` half-year / one-year fusion 鑴氭湰宸茬粡浠?active 鏍圭洰褰曠Щ闄わ紝涓嶅啀浣滀负褰撳墠瀹為獙鍏ュ彛銆?
