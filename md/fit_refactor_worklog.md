# FIT Refactor Worklog

鏈€鍚庢洿鏂帮細2026-04-13

## 1. 鏂囨。鐢ㄩ€?

杩欎唤鏂囨。璁板綍 FIT 渚ф暣鐞嗕笌閲嶆瀯鐨勫綋鍓嶇姸鎬併€佺湡瀹?smoke 缁撴灉鍜屽悗缁緟鍔炪€?

寤鸿闃呰椤哄簭锛?
1. `fit_refactor_worklog.md`
2. `global_architecture.md`
3. `fit_server_runbook.md`

## 2. 褰撳墠闃舵缁撹

FIT 渚у凡缁忓畬鎴愪袱杞暣鐞嗭細
- 绗竴杞細姊崇悊鏁版嵁灞傘€佸疄楠屽眰銆佽剼鏈拰鍩虹鏂囨。銆?
- 绗簩杞細淇 scaler 璇箟銆侀噸鍐?FIT fusion銆佽ˉ榻愭湰鍦?CPU 楠岃瘉銆?

褰撳墠缁撹锛?
- 鏁板€兼祦璁粌閫昏緫淇濇寔鍘熸牱銆?
- `fit_num_with_meta` 鍙柊澧炰簡缁?fusion 璇诲彇涓棿鐗瑰緛鐨勬帴鍙ｃ€?
- FIT fusion 宸茬粡浠庢棫鐨勬祬铻嶅悎锛屽垏鎹㈡垚鈥滄枃鏈潯浠跺寲鏁板€间腑闂寸壒寰佲€濈殑鐗堟湰銆?

## 3. 宸插畬鎴愪簨椤?

### 3.1 鏁版嵁灞?

宸插畬鎴愶細
- 鎶藉嚭 FIT 鍏叡鏁版嵁閫昏緫鍒?`data_provider/fit_dataset_utils.py`
- 缁熶竴绠＄悊锛?
  - `CITY_MAP / GENDER_MAP / AGE_MAP`
  - `group` 瑙ｆ瀽
  - train / val / test split
  - `element_map`
  - 鏍囧噯鍖?
  - 鏃堕棿鐗瑰緛
  - `caption_emb` 瀵归綈
- 鏂板 `fit_scaler_mode`
  - `train_only`
  - `split_fit`
- 褰撳墠榛樿锛?
  - `fit_scaler_mode=train_only`

### 3.2 鏁板€兼祦

宸插畬鎴愶細
- `Exp_Fit_Num` 鏀舵暃涓哄彧璐熻矗 FIT 鏁板€兼祦銆?
- 鏁板€兼祦鏈韩璁粌閫昏緫涓嶆敼銆?
- `Model_Fit_Num_With_Meta` 鏂板 `extract_features()`锛?
  - 杩斿洖 `forecast`
  - 杩斿洖 `encoded_tokens`
  - 杩斿洖 `summary_state`
- 杩欎釜鎺ュ彛鍙粰 fusion 浣跨敤锛屼笉鏀瑰彉鍘?baseline 璁粌璇箟銆?

### 3.3 FIT fusion

宸插畬鎴愶細
- 淇濈暀棰勫鐞?`caption_emb` 璺嚎锛屼笉鎺?raw text encoder銆?
- 閲嶅啓 `Model_Fit_Fusion` 涓烘枃鏈潯浠跺寲鐗堟湰锛?
  - 鏂囨湰閫傞厤鍣?`text_adapter`
  - 鏂囨湰鐢熸垚 `gamma / beta`
  - 鏂囨湰璋冨埗鏁板€?`encoded_tokens`
  - 鍐嶈繘鍏?`direct / residual`

鏂扮殑 `direct`锛?
- 涓嶅啀鍋?`(1-w) * y_num + w * y_text`
- 鐩存帴浠庘€滄枃鏈皟鍒跺悗鐨勬暟鍊肩壒寰佲€濊緭鍑烘渶缁堥娴?

鏂扮殑 `residual`锛?
- 涓嶅啀浣跨敤 `beta_delta`
- 涓嶅啀浣跨敤 `force_gain`
- 杈撳嚭鈥滃巻鍙叉尝鍔ㄦ湁鐣屸€濈殑绾犲亸閲?

### 3.4 CLI 涓庤剼鏈?

宸插畬鎴愶細
- `run.py` 宸叉寜鍙傛暟缁勬暣鐞嗐€?
- FIT 渚у睍绀哄眰宸茬粡鍘绘帀鏃?fusion 瓒呭弬鐨勬墦鍗般€?
- FIT fusion 鑴氭湰宸叉竻鎺夛細
  - `--beta_delta`
  - `--llm_dim`
- FIT fusion 瀹樻柟鑴氭湰宸叉樉寮忚ˉ涓婏細
  - `--train_epochs 20`
  - `--batch_size 200`

娉ㄦ剰锛?
- `run.py` 涓?`llm_dim / delta_scale / use_vol_prior / force_gain` 浠嶄繚鐣欏湪 parser 閲?
- 杩欐槸涓轰簡涓嶇牬鍧忓綋鍓?Geo 璺緞
- FIT 褰撳墠瀹炵幇宸茬粡涓嶅啀浣跨敤杩欎簺鍙傛暟

## 4. 鏈湴鐜鐘舵€?

褰撳墠 base Python 鐜宸茶濂斤細
- `torch` CPU
- `numpy`
- `scipy`
- `scikit-learn`
- `pandas`
- `matplotlib`
- `tqdm`
- `einops`
- `reformer_pytorch`

闈欐€佹鏌ュ凡閫氳繃锛?
- `python run.py --help`
- `python -m compileall -q run.py exp models data_provider utils layers`

## 5. 鐪熷疄 smoke 缁撴灉

### 5.1 鏁板€兼祦

宸茬湡瀹炶窇閫氾細

1. `fit_num`
- CPU
- 灏忔牱鏈?
- 璁粌 / 楠岃瘉 / 娴嬭瘯 / checkpoint / 缁撴灉鏂囦欢鍏ㄩ儴姝ｅ父

2. `fit_num_with_meta`
- CPU
- 灏忔牱鏈?
- 绗竴杞暣鐞嗗悗璺戦€氳繃

3. `fit_num_with_meta` 澶嶆祴
- 鍦ㄦ柊澧?`extract_features()` 涔嬪悗閲嶆柊璺戣繃
- 璁粌 / 楠岃瘉 / 娴嬭瘯浠嶆甯?
- 璇存槑鏁板€兼祦鍘熻涔夋病鏈夎鏀瑰潖

### 5.2 scaler 璇箟

宸茬湡瀹為獙璇侊細
- `fit_scaler_mode=train_only` 涓?
  - train / val / test 鍏变韩鍚屼竴涓?train-fit scaler
  - 涓嶅啀鍚勮嚜 fit 鍚勮嚜 split

### 5.3 鏂扮増 FIT fusion

鐢变簬姝ｅ紡 FIT caption embedding 褰撴椂涓嶅湪鏈湴锛屾湰杞娇鐢ㄤ复鏃跺皬瀛愰泦鍜屼复鏃?`caption_emb` 鍋氫唬鐮佽矾寰?smoke銆備复鏃惰祫浜у彧鐢ㄤ簬娴嬭瘯锛屽悗闈㈠凡鍒犻櫎銆?
宸茬湡瀹炶窇閫氾細

1. 鏂扮増 `direct + split`
- 璇箟锛歚joint_direct`
- 缁撴灉锛氳缁?/ 楠岃瘉 / 娴嬭瘯姝ｅ父

2. 鏂扮増 `residual + split`
- 璇箟锛歚residual_correction`
- 鍔犺浇鏈湴鏁板€?smoke checkpoint
- `freeze_numerical=True`
- 缁撴灉锛氳缁?/ 楠岃瘉 / 娴嬭瘯姝ｅ父

3. 鏂扮増 `direct + unified`
- 楠岃瘉缁熶竴瀛︿範鐜囨ā寮?
- 缁撴灉锛氳缁?/ 楠岃瘉 / 娴嬭瘯姝ｅ父

## 6. 鏈疆鍏抽敭璁捐鍙樺寲

### 6.1 鍒犻櫎鐨勬棫 FIT fusion 鎬濊矾

FIT 褰撳墠宸茬粡涓嶅啀渚濊禆锛?
- `beta_delta`
- `force_gain`
- 鏂囨湰渚у啓姝昏緭鍏ョ淮搴?`llm_dim=768`

### 6.2 鏂扮殑鏂囨湰浣滅敤鏂瑰紡

鏃х増闂锛?
- 鏂囨湰鍙湪杈撳嚭绔仛涓€涓祬 head
- 寰堝鏄撻€€鍖栨垚鈥滄暟鍊奸娴?+ 灏忎慨灏忚ˉ鈥?

鏂扮増鏀规垚锛?
- 鏂囨湰鍏堣皟鍒舵暟鍊?backbone 鐨勪腑闂?`encoded_tokens`
- 鍐嶈繘鍏?direct / residual

### 6.3 鏂?residual 鐨勭害鏉熸柟寮?

鏃х増锛?
- 闈犻澶栨鍒欓檺鍒剁籂鍋忓箙搴?

鏂扮増锛?
- 鐩存帴鐢ㄥ巻鍙?`std + range` 缁?residual 璁句笂鐣?
- 绾犲亸骞呭害澶╃劧鏈夎竟鐣?
- 璇箟鏇存竻妤氾紝涔熸洿瀹规槗瑙ｉ噴

## 7. 褰撳墠鍓╀綑椋庨櫓

### 7.1 鏈湴杩樻病鐢ㄦ寮?FIT caption embedding

鐜板湪鑳界‘璁ょ殑鏄細
- 浠ｇ爜璺緞閫氫簡
- 鏂?direct / residual 閫氫簡
- split / unified 閮介€氫簡

鐜板湪杩樹笉鑳界‘璁ょ殑鏄細
- 鍦ㄦ寮?`caption_emb` 鍜屾湇鍔″櫒鍏ㄩ噺璁粌涓婏紝鏂扮粨鏋勪竴瀹氫紭浜庢棫缁撴瀯

### 7.2 鏂囨湰鐡堕浠嶅湪鎻忚堪绛栫暐鏈韩

鍗充娇璁粌妗嗘灦鐞嗛『鍚庯紝鏂囨湰渚т笂闄愪粛鐒跺彇鍐充簬锛?
- 鏂囨湰鎻忚堪鏄惁鐪熺殑鏈変俊鎭噺
- 鏂囨湰鎻忚堪鏄惁鍜屾暟鍊艰秼鍔挎湁鍏?
- 棰勫鐞?embedding 鏄惁瓒冲琛ㄨ揪杩欎簺淇℃伅

## 8. 涓嬩竴姝ュ缓璁?

涓嬩竴姝ュ缓璁『搴忥細
1. 鍦ㄦ湇鍔″櫒涓婄敤姝ｅ紡 FIT caption embedding 璺戞柊鐗?`joint_direct`
2. 鍐嶈窇鏂扮増 `residual_correction`
3. 姣旇緝锛?
   - baseline 鏁板€兼祦
   - `direct + unified`
   - `direct + split`
   - `residual + freeze_numerical`
   - `residual + 涓嶅喕缁撴暟鍊兼祦`
4. 濡傛灉鏂扮増 fusion 浠嶇劧娑ㄤ笉鍔紝鍐嶅洖澶村鐞嗏€滄枃鏈弿杩板浣曠敓鎴愨€濊繖涓洿涓婃父鐨勯棶棰?

## 9. 涓嬩竴鐗?fusion 璁捐鏂囨。

濡傛灉鍚庨潰杩樿缁х画鎵╁睍 FIT fusion锛岃鍏堢湅锛?
- `global_architecture.md`

杩欎唤鏂囨。璁板綍鐨勬槸锛?
- 杩欐閲嶅啓閲囩敤鐨勮璁″師鍒?
- 涓轰粈涔堝垹鎺?`beta_delta / force_gain / llm_dim`
- direct / residual 鐨勬柊璇箟

## 10. 2026-04-13 鏈嶅姟鍣ㄩ杞粨鏋?

### 10.1 鍗婂勾 `fit_num_with_meta` baseline

褰撳墠鐢ㄦ埛鍙嶉鐨勬寮忕粨鏋滐細
- `MAE  = 0.085858`
- `MSE  = 0.013315`
- `RMSE = 0.115391`
- `MAPE = 30.23%`
- `WAPE = 18.52%`

### 10.2 鍗婂勾鏂扮増 `fit_fusion_halfyear_joint_direct`

褰撳墠鐢ㄦ埛鍙嶉鐨勬寮忕粨鏋滐細
- `MAE  = 0.091474`
- `MSE  = 0.014394`
- `RMSE = 0.119975`
- `MAPE = 35.42%`
- `WAPE = 19.73%`

缁撹锛?
- 褰撳墠鏂扮増 `direct` 棣栬疆姝ｅ紡瀹為獙寮变簬鏁板€间富骞?baseline
- 鍥犳鐜伴樁娈典笉鑳藉垽鏂€滄枃鏈祦璁捐宸茬粡鏈夋晥鈥?

## 11. 褰撳墠宸茬‘璁ら棶棰?

### 11.1 瀛︿範鐜囧垎缁勮 scheduler 瑕嗙洊

褰撳墠 `fusion_optimizer_mode=split` 铏界劧鍒濆鍖栨椂浼氬垎鎴愶細
- num group = `lr_num`
- text group = `lr_text`

浣?`adjust_learning_rate()` 浼氬湪姣忎釜 epoch 缁撴潫鍚庢妸鎵€鏈?param group 缁熶竴瑕嗙洊鎴?`args.learning_rate`銆?

杩欐剰鍛崇潃锛?
- `split` 鐜板湪涓嶆槸鍏ㄧ▼鍒嗙粍瀛︿範鐜?
- 鍙槸鍒濆鍖栨椂鐭殏鍒嗙粍锛屽悗闈㈠氨琚姽骞?

### 11.2 `type3` 鍦?20 epoch 璁粌閲屽嚑涔庝笉璧蜂綔鐢?

褰撳墠 `type3` 鐨勫啓娉曟槸锛?
- `lr = learning_rate * (0.1 ** (epoch // 20))`

鑰岃剼鏈粯璁わ細
- `train_epochs = 20`

鍐嶅姞涓?lr 璋冩暣鍙戠敓鍦?epoch 缁撴潫鍚庯紝鎵€浠ワ細
- 鍓?19 涓?epoch 閮芥槸 `0.001`
- 鍒扮 20 涓?epoch 缁撴潫鎵嶄細鍑嗗闄?lr
- 浣嗚缁冨叾瀹炲凡缁忕粨鏉?

涔熷氨鏄锛?
- 褰撳墠 20 epoch 閰嶇疆涓嬶紝`type3` 鍩烘湰娌℃湁瀹為檯璋冨害鏁堟灉

### 11.3 褰撳墠 `joint_direct` 涓嶆槸鍏钩鐨勨€滄枃鏈寮?baseline鈥?

鐜板湪瀹樻柟 `direct` 鑴氭湰璇箟鏄細
- 浠庡ご joint 璁粌
- 涓嶅姞杞芥暟鍊?ckpt
- 涓嶅喕缁撴暟鍊兼祦

杩欐剰鍛崇潃瀹冧笉鏄€滃湪寮烘暟鍊间富骞插熀纭€涓婂姞鏂囨湰鈥濓紝鑰屾槸锛?
- 涓€涓甫鏂囨湰鏉′欢鍖栫殑鏂版ā鍨嬶紝浠庡ご閲嶈

鎵€浠ュ畠棣栬疆寮变簬宸叉湁寮?baseline锛屽苟涓嶆剰澶栥€?

## 12. 褰撳墠鍒ゆ柇

鐜伴樁娈垫洿鍍忔槸锛?
- 鏁板€间富骞叉湰韬槸绋冲畾涓斿己鐨?
- 鏂囨湰瀹為獙鐨勯瑕侀棶棰橈紝鏈繀鏄枃鏈俊鎭湰韬棤鐢?
- 鏇村彲鑳芥槸璁粌 recipe 鍜屽疄楠岃璁¤繕娌℃湁绔欏湪涓€涓叕骞宠捣鐐逛笂

浼樺厛绾у垽鏂細
1. 鍏堜慨瀛︿範鐜囦笌璋冨害闂
2. 鍐嶉噸鏂拌窇 direct / residual
3. 鍐嶅垽鏂枃鏈璁℃湰韬槸鍚︽湁鏁?

## 13. 涓嬩竴杞疄楠屽缓璁?

寤鸿鎸変笅闈㈤『搴忓仛锛岃€屼笉鏄户缁洿鎺ュ爢鏂扮粨鏋勶細

1. 鍏堜慨 `split lr` 琚鐩栫殑闂
- 鑷冲皯淇濊瘉 `lr_num` 鍜?`lr_text` 鑳藉叏绋嬬嫭绔?

2. 鍏堟妸 fusion 鑴氭湰鏀规垚涓嶄娇鐢ㄥ綋鍓嶆棤鏁堢殑 `type3`
- 涓€涓畝鍗曞彲琛屾柟妗堟槸 `--adjust 0`

3. 鍗婂勾浠诲姟浼樺厛閲嶈窇涓嬮潰 4 缁?
- `direct + from scratch + split + adjust=0`
- `direct + load num ckpt + unfreeze + split + adjust=0`
- `residual + load num ckpt + freeze + split + adjust=0`
- `residual + load num ckpt + unfreeze + split + adjust=0`

4. 濡傛灉涓婅堪浠嶆棤鎻愬崌锛屽啀鐪嬬粨鏋勫眰
- 鏄惁缁?direct 淇濈暀鏇村己鐨?`y_num` skip
- 鏄惁鍏堝 `caption_emb` 鍋氭洿绋崇殑褰掍竴鍖?
- 鏄惁闇€瑕侀噸鏂版鏌ユ枃鏈弿杩版湰韬殑璐ㄩ噺

## 14. 2026-04-13 涓嬩竴杞慨鏀硅鍒?

褰撳墠鍒ゆ柇宸茬粡姣旇緝鏄庣‘锛?
- 鏁板€间富骞?`fit_num_with_meta` 浠嶇劧绋冲畾涓斿己
- 鏂扮増鏂囨湰娴侀杞寮忕粨鏋滃急浜?baseline
- 鐜板湪浼樺厛瑕佹帓鏌ヨ缁?recipe 涓庡疄鐜扮粏鑺傦紝鑰屼笉鏄户缁洸鐩敼鏇村鏉傜殑鏂囨湰缁撴瀯

涓嬩竴杞慨鏀规寜杩欎釜椤哄簭鎵ц锛?

1. 鍏堜慨 `split lr` 琚?scheduler 瑕嗙洊鐨勯棶棰?
- 鐩爣锛氳 `lr_num` 鍜?`lr_text` 鍦ㄦ暣涓缁冭繃绋嬩腑閮借兘淇濇寔鍒嗙粍璇箟
- 鍋氭硶锛氫慨鏀瑰涔犵巼璋冨害閫昏緫锛岃姣忎釜 param group 鍩轰簬鑷繁鐨?`base_lr` 琛板噺锛岃€屼笉鏄粺涓€瑕嗙洊鎴?`args.learning_rate`

2. 鍚屾椂淇 FIT fusion 瀹樻柟鑴氭湰鐨勯粯璁よ缁冪瓥鐣?
- 褰撳墠 20 epoch + `type3` 鍑犱箮娌℃湁鏈夋晥璋冨害
- 涓嬩竴杞粯璁ゅ厛鏀规垚锛?
  - `--adjust 0`
- 杩欐牱鍏堟妸 scheduler 骞叉壈鎷挎帀锛屼繚璇佸疄楠岀粨璁烘洿骞插噣

3. 琛ヤ竴缁勬洿鍏钩鐨勫畼鏂瑰疄楠屽叆鍙?
- 涓嶆槸鍙繚鐣欙細
  - `joint_direct`
  - `residual_correction`
- 杩樿琛ヤ笂鏇村叧閿殑缁勫悎锛?
  - `direct + load num ckpt + unfreeze`
  - `residual + load num ckpt + unfreeze`

4. 瀹屾垚鍚庡仛鏈湴闈欐€佹鏌?
- `compileall`
- `run.py --help`
- 鏍稿鑴氭湰鍙傛暟涓庢枃妗ｄ竴鑷?

5. 鏂囨。鎸佺画鍚屾
- `fit_refactor_worklog.md`
- `global_architecture.md`
- 濡傛湁闇€瑕侊紝琛ュ厖 `fit_server_runbook.md`

## 15. 2026-04-13 瀛︿範鐜囦笌鑴氭湰淇

鏈疆宸插畬鎴愶細

1. 淇 `split lr` 琚?scheduler 瑕嗙洊鐨勯棶棰?
- `exp/exp_fit_fusion.py` 閲岀殑 optimizer param group 鐜板湪浼氭樉寮忎繚瀛樿嚜宸辩殑 `base_lr`
- `utils/tools.py` 閲岀殑 `adjust_learning_rate()` 涓嶅啀鎶婃墍鏈?group 涓€璧疯鐩栨垚 `args.learning_rate`
- 鐜板湪 scheduler 浼氭寜姣忎釜 group 鑷繁鐨?`base_lr` 鍋氱浉瀵圭缉鏀?

2. 淇 FIT fusion 瀹樻柟鑴氭湰榛樿 recipe
- 4 涓師鏈?FIT fusion 鑴氭湰鐜板湪閮芥樉寮忓姞涓婁簡 `--adjust 0`
- 杩欐牱鍙互鍏堝幓鎺夊綋鍓?`20 epoch + type3` 鐨勬棤鏁堣皟搴﹀共鎵?

3. 琛ュ厖鏇村叕骞崇殑鎺ㄨ崘瀹為獙鑴氭湰
- `fit_fusion_halfyear_direct_from_ckpt.sh`
- `fit_fusion_halfyear_residual_unfreeze.sh`
- `fit_fusion_oneyear_direct_from_ckpt.sh`
- `fit_fusion_oneyear_residual_unfreeze.sh`

4. 鏈疆闈欐€侀獙璇?
- `python -m compileall -q run.py exp models data_provider utils layers` 宸查€氳繃
- 鍗曠嫭鍋氫簡涓€涓?scheduler 灏忔祴璇曪紝纭 `type3` 鍦?split 妯″紡涓嬩細鎶婏細
  - numerical group: `1e-4 -> 1e-5`
  - text_fusion group: `5e-4 -> 5e-5`

褰撳墠鍒ゆ柇锛?- 褰撳墠鏈€鍊煎緱閲嶆柊璺戠殑杩樻槸 4 缁勫崐骞村疄楠岋細
  - `direct + from scratch + split + adjust=0`
  - `direct + load num ckpt + unfreeze + split + adjust=0`
  - `residual + load num ckpt + freeze + split + adjust=0`
  - `residual + load num ckpt + unfreeze + split + adjust=0`

琛ュ厖锛?- 鏈嶅姟鍣ㄦ墽琛岄『搴忎笌缁撴灉璁板綍妯℃澘宸插崟鐙啓鍏?`fit_server_runbook.md`

## 16. 2026-04-13 鍗婂勾 20 epoch 姝ｅ紡缁撴灉

鐢ㄦ埛宸插畬鎴愬崐骞翠换鍔?5 缁勭粨鏋滐細

- `fit_num_with_meta`
  - `MAE  = 0.085858`
  - `MSE  = 0.013315`
  - `RMSE = 0.115391`
  - `MAPE = 30.23%`
  - `WAPE = 18.52%`
- `direct + scratch`
  - `MAE  = 0.105100`
  - `MSE  = 0.018546`
  - `RMSE = 0.136183`
  - `MAPE = 40.06%`
  - `WAPE = 22.67%`
- `direct + num_ckpt + unfreeze`
  - `MAE  = 0.081830`
  - `MSE  = 0.011850`
  - `RMSE = 0.108857`
  - `MAPE = 31.00%`
  - `WAPE = 17.65%`
- `residual + num_ckpt + freeze`
  - `MAE  = 0.085395`
  - `MSE  = 0.013151`
  - `RMSE = 0.114679`
  - `MAPE = 30.38%`
  - `WAPE = 18.42%`
- `residual + num_ckpt + unfreeze`
  - `MAE  = 0.079863`
  - `MSE  = 0.011620`
  - `RMSE = 0.107797`
  - `MAPE = 28.83%`
  - `WAPE = 17.23%`

褰撳墠缁撹锛?
- `direct + scratch` 鍙互瑙嗕负褰撳墠鏃犳晥瀵圭収缁勩€?- `direct + num_ckpt + unfreeze` 宸茬粡鏄庢樉浼樹簬 baseline銆?- `residual + num_ckpt + freeze` 鐣ヤ紭浜?baseline锛屼絾鎻愬崌鏈夐檺銆?- `residual + num_ckpt + unfreeze` 褰撳墠鏄崐骞翠换鍔￠噷鏈€寮虹殑涓€缁勩€?
## 17. 2026-04-13 涓嬩竴姝ワ細鍏堣窇 100 epoch

褰撳墠涓嶇户缁敼 fusion 缁撴瀯锛屽厛鎶婅缁?recipe 鎷夐暱鍒?100 epoch銆?
宸插畬鎴愶細

- 8 涓?FIT fusion 鑴氭湰鏀逛负榛樿 `train_epochs=100`
- 榛樿 `patience=100`
- 淇濈暀 `adjust=0`
- 鑴氭湰鏀寔鐜鍙橀噺瑕嗙洊锛屼笉闇€瑕佷互鍚庡弽澶嶆墜鏀?
褰撳墠寤鸿锛?
- 鍏堢户缁窇鍗婂勾杩?4 涓?fusion 缁勫悎鐨?100 epoch 鐗堟湰
- 璺戝畬鍐嶅喅瀹氭槸鍚︾户缁敼缁撴瀯锛岃繕鏄洿鎺ュ鍒跺埌涓€骞翠换鍔?
