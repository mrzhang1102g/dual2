# FIT 閲嶆瀯宸ヤ綔璁板綍

鏈€鍚庢洿鏂帮細2026-04-19

## 0411

瀹屾垚浜嗭細
- FIT 鏁版嵁灞傚叕鍏遍€昏緫鎶藉彇
- `train_only` scaler 淇
- 鏁板€?baseline 淇濇寔涓嶅姩
- embedding-fusion 鑴氭湰鏁寸悊
- random / filler / disable_text 鎺у埗瀹為獙
- structured / multi-view 鏂囨湰瀹為獙
- legacy 澶存仮澶嶄笌褰掓。

杩欎竴姝ユ渶閲嶈鐨勪骇鍑轰笉鏄渶缁堟ā鍨嬶紝鑰屾槸闂瀹氫綅锛?- 鏃х殑 embedding-fusion 璺嚎闈炲父瀹规槗缁曞紑鏂囨湰
- 鏂囨湰鏄惁鐪熷疄銆佹湁鎰忎箟銆侀殢鏈恒€佹棤璇箟妯℃澘锛岀粨鏋滃樊璺濋兘寰堝皬

## 0414

渚濇灏濊瘯浜嗭細
- `v2`
- `v3`
- `v4`
- `v5`

鎬讳綋瑙勫緥锛?- `direct_v3` 鎴愪负鏈€绋崇殑 direct baseline
- residual 瀹舵棌姣忎竴鐗堥兘鏈変竴鐐硅繘姝?- 浣嗙洿鍒?`v5`锛屼粛娌℃湁鍦?`MAE/WAPE` 涓婂帇杩?`direct_v3`

杩欎竴姝ョ殑鏈€缁堢粨璁烘槸锛?- 缁х画鍦?`emb.pt + residual family` 涓婄粏璋冿紝鏀剁泭宸茬粡鎺ヨ繎涓婇檺

## 0417-v6

鍩轰簬 DualSG 鎬濊矾锛屽皾璇曚簡锛?- 涓嶅啀璇?`emb.pt`
- 鐩存帴浠?json 閲岃鍙?raw text
- 鍐荤粨鏈湴 GPT2
- 鍦ㄧ嚎 tokenizer + pooling
- forecast-space fusion

缁撴灉锛?- 鎸囨爣姣斿墠闈㈡渶濂界殑鐗堟湰鏇村樊
- 璇存槑鈥滄洿璐磋鏂囩殑琛ㄩ潰褰㈠紡鈥濅笉绛変簬鍦ㄥ綋鍓嶄换鍔￠噷鏇存湁鏁?
鍥犳 0417 鐨勫垽鏂繘涓€姝ユ槑纭細
- 闂涓嶅彧鏄?fusion 缁撴瀯
- 涔熷寘鎷€滄枃鏈埌搴曞簲涓嶅簲璇ヤ綔涓鸿緭鍏ユā鎬佲€?
## 褰撳墠杞悜锛歋emantic Supervision

褰撳墠 active 鏂版柟鍚戯細
- 涓嶅啀鎶婄粨鏋勫寲鏂囨湰褰撹緭鍏ユā鎬?- 鎶婄粨鏋勫寲鏂囨湰杞垚璇箟鏍囩
- 鐢ㄨ繖浜涙爣绛惧仛鏁板€间富骞茬殑澶氫换鍔＄洃鐫?
褰撳墠璁捐锛?- 涓绘ā鍨嬩粛鐒舵槸 `Model_Fit_Num_With_Meta`
- 鏂版ā鍨?`Model_Fit_Num_With_Meta_Semantic` 鍙湪 `summary_state` 涓婇澶栨帴 4 涓涔夊ご
- 4 涓换鍔★細
  - `overall_trend`
  - `recent_regime`
  - `volatility`
  - `major_turning_points`
- 鎬绘崯澶憋細
  - `forecast_loss + semantic_loss_weight * semantic_loss`

## 褰撳墠鍒ゆ柇

杩欐潯 semantic supervision 璺嚎姣旂户缁仛 fusion 鏇村€煎緱鎺ㄨ繘锛屽師鍥犳湁涓夌偣锛?
1. 鏂囨湰鏈潵灏辨潵鑷悓涓€鏉℃暟鍊煎簭鍒楋紝鏇撮€傚悎浣滀负璇箟鐩戠潱锛岃€屼笉鏄吉瑁呮垚鐙珛妯℃€併€?2. 鏁板€间富骞插凡缁忓緢寮猴紝杈呭姪浠诲姟姣旇緭鍏ヨ瀺鍚堟洿鑷劧銆?3. structured json 鏈韩灏辨彁渚涗簡鍙互鐩存帴鍒╃敤鐨勮秼鍔挎爣绛剧粨鏋勩€?
