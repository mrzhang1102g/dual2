# FIT 閲嶆瀯宸ヤ綔璁板綍

鏈€鍚庢洿鏂帮細2026-04-17

## 0411 闃舵鍥為【

`0411` 宸茬粡瀹屾垚锛?
- FIT 鏁版嵁灞傚叕鍏遍€昏緫鎶藉彇
- `train_only` scaler 淇
- `fit_num_with_meta` 涓诲共淇濈暀涓嶅姩
- fusion 璁粌鑴氭湰鏁寸悊
- random / filler / disable_text 鎺у埗瀹為獙
- structured / multi-view 鏂囨湰瀹為獙
- legacy 澶存仮澶嶈瘖鏂?- 鑴氭湰涓庣粨鏋滃揩鐓у綊妗ｅ埌 `scripts_archive/fit_halfyear_0411/`

杩欎竴闃舵鏈€澶х殑浠峰€间笉鏄€滃緱鍒版渶缁堢粨鏋勨€濓紝鑰屾槸鎶婇棶棰樺畾浣嶆竻妤氾細

- 鏃х殑 `emb.pt + fusion` 璺嚎鍏佽妯″瀷缁曞紑鏂囨湰
- `random / filler / disable_text` 涓庣湡瀹炴枃鏈樊寮傚お灏?
## 0414 闃舵鍥為【

`0414` 渚濇楠岃瘉浜嗭細

- `v2`锛氭樉寮忔暟鍊?skip 鐨勫弻娴?direct / basis-text residual
- `v3`锛? 涓€欓€?expert routing residual
- `v4`锛氳秼鍔跨骇銆佸垎娈靛父鏁?correction residual
- `v5`锛歵ext-aligned segment residual

缁熶竴缁撹锛?
1. `direct_v3` 鎴愪负褰撳墠鏈€绋崇殑 direct 鍩虹嚎
2. residual 瀹舵棌姣忎竴浠ｉ兘鏈変竴鐐硅繘姝?3. 浣嗙洿鍒?`v5`锛屼粛鐒舵病鏈夊湪 `MAE / WAPE` 涓婂帇杩?`direct_v3`
4. 鍥犳缁х画鍦?`emb.pt + residual` 瀹舵棌涓婅凯浠ｏ紝棰勬湡鏀剁泭宸茬粡寰堜綆

鍏抽敭瀵规瘮锛?
- `direct_v3`
  - `MAE = 0.079424`
  - `MAPE = 29.03%`
  - `WAPE = 17.13%`
- `residual_v5`
  - `MAE = 0.079558`
  - `MAPE = 28.90%`
  - `WAPE = 17.16%`

## 涓轰粈涔堝仠姝㈢淮鎶?v3-v5 active 璺嚎

褰撳墠鍒ゆ柇宸茬粡姣旇緝鏄庣‘锛?
- 闂涓嶅啀鏄€滄枃鏈槸涓嶆槸澶熷己鈥?- 鏇村儚鏄€滄枃鏈渶鍚庢€庝箞琚?forecasting 浣跨敤鈥?
鎴戜滑浠?DualSG 璁烘枃鍜屽畼鏂瑰疄鐜伴噷寰楀埌鐨勫惎鍙戞槸锛?
1. DualSG 骞朵笉寮鸿皟 latent-space 娣卞榻?2. 瀹冨厛鍗曠嫭璁粌 TSCG锛岀敓鎴?caption
3. 姝ｅ紡棰勬祴闃舵锛屾暟鍊兼祦鍜屾枃鏈祦鐩稿鐙珛
4. 鏂囨湰娴佸湪 forecast space 閲岀粰鍑鸿涔変慨姝?
杩欏拰鎴戜滑涔嬪墠鐨?`emb.pt + deep residual fusion` 璺嚎骞朵笉涓€鏍枫€?
## 0417 褰撳墠鍐冲畾

`0417` 寮€濮嬶紝FIT active fusion 鏀规垚 `v6`锛?
- 涓嶅啀璇诲彇 `caption_emb.pt`
- 鐩存帴浠?FIT json 閲岃鍙?`annotations`
- 鍦ㄧ嚎杩涘叆鍐荤粨鐨勬湰鍦版枃鏈紪鐮佹ā鍨?- 鍋?token pooling
- 鎶曞奖鍒伴娴嬬┖闂?- 涓?`y_num` 鐩存帴鍋?forecast-space 铻嶅悎

涔熷氨鏄細

- `y_final = (1 - w_t) * y_num + w_t * y_text`

## 0417 v6 鐩爣

杩欒疆涓嶆槸缁х画杩芥眰鈥滄洿澶嶆潅鐨?residual鈥濓紝鑰屾槸鍏堝洖绛斾竴涓洿鍩虹鐨勯棶棰橈細

- 濡傛灉鐩存帴鎸?DualSG 椋庢牸鏀规垚 raw-text forecast-space fusion锛岀粨鏋滀細涓嶄細鏇村悎鐞嗭紵

鍥犳 `v6` 绗竴杞彧鍋氾細

- FIT half-year
- `ckpt + unfreeze`
- `real raw text`
- 鍐荤粨鏈湴 GPT2
- 鍙窇 `20 epoch`

鍚庣画鍙湁鍦ㄩ杞粨鏋滆嚦灏戜笉宸簬 `direct_v3` 鏃讹紝鎵嶄細缁х画琛ワ細

- `disable_text`
- `random`
- `filler`

