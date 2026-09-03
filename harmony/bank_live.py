"""bank_live — G 音庫取樣器（第 1 階，08-11 雙開工 A 線）

原理＝v7.1「譜是定的＝不渲染」推廣到「**音高空間是定的＝不即時渲染**」：
天使唱「啊」＋半音階和聲的可能輸出是有限集合——開場把每聲部整個音域
逐半音預渲成可循環音庫（reflow 正典品質、磁碟快取），live 只做：
  偵測他的音（自相關，~46ms 窗）→ 半音量化＋遲滯 → 三聲部查表觸發
  → 循環播放＋30ms crossfade 換音 → 起音/釋音包絡 → 混音。
零 ML 在迴圈裡＝延遲 ≈ 偵測 ~36ms（實測）＋I/O ≈ **~50-60ms**，品質＝正典。

v10（08-12 穩定性總 review 修復場）：兩位 fresh 審查員（正確性/併發＋
行為/症狀）共 25+ findings，S 級全修——音域外 clip 順序（28.6Hz 重啟迴
圈）、互補 crossfade（+6dB 過衝）、不應期時鐘從未寫入（v9 死碼）、v9
新音高放行判死（bass unison/pad 和弦音自我否決＝頓挫主因，回到 v2 嘴閉
即殺）、拒斥層納入 pads（墊放不掉主因）、鏡頭門控凍結修復＋30Hz、
underrun 淡出、次諧波防護、cents 無聲衰減、搶拍去調內綁定（冷啟動）、
fast 路徑補 hist、cb 鎖內配置移出。詳 review 檔＋worklog 08-12。
⚠ 台架限制（review A15）：--file 不經嘴部門控路徑＝離線全綠不代表門控
時序沒問題，門控類改動必須 live 驗。

v17（08-13 連續母音場，Harry 第一性定調）：反應式離散選層（v16 tilt→
16.1 唇距→16.2 聲音指紋→16.3 嘴形二維最近鄰）整個範式判死——鏡子不是
夥伴、永遠遲到半顆母音。母音＝連續口腔形狀不是符號：mediapipe 嘴形
(高,寬) 連續值 → 對五母音錨點反距離權重 → 逐 hop 逐樣本連續混合五層
loop（同 midi 各層等長＝樣本對齊、零切換事件）；分類器/遲滯/dwell 整類
病消失。已知待驗：跨層混音可能輕微 chorus（同 seed 同 f0 應相近，耳裁）。

聲部（v42 起四嘴；數值以 VOICES 為準）：
  bass1＝-8（C3 區，08-16 音域定案）｜tenor＝male8 spk7｜alto3-40k(spk2)＝+12
  ｜sop3＝+12 再上方全音階三度（KeyTracker auto 定調，respond2 08-04 驗過那顆）。

和弦鎖定（v6①）免費送：他的音準偏移（對量化音的 cents 差、EMA 平滑）
乘進所有聲部的播放速率＝團跟著他的音準彎，拍頻消失。--lock 0 關。

音量走「有聲/無聲＋包絡」不走 mic 位準（08-11 血訓：USB PnP 輸入
~−61dBFS＝位準不可信；偵測用自相關對位準無感）。

v5（08-11 夜，自主迭代）：--file 離線台架（同一條 per-hop 管線＝離線結
論對 live 成立）＋已知音符拒斥（第二道回授防線：偵測音貼著「離他音域
>6 半音」的在播音＝喇叭繞回來的 alto/sop，擋掉；bass unison 歸嘴部門控
管）。台架驗證：唱聲段 rms 0.062 有聲、模擬 alto 回授段 0.015 不再自激；
合成音頭實測演算法延遲 36ms（＋I/O ≈ 端到端 ~55-60ms）。

v1 裸奔實測（08-11）：回授實錘——他一停，喇叭裡的天使（完美週期訊號）
被偵測器當成他＝自激不停唱。v2 接**嘴部門控**（score_live v3.1 同款：
mediapipe 開口度＋遲滯＋0.8s 寬限；嘴閉＝音高不算你的）＝回授騙不了
嘴唇。鏡頭掛/--mouth 0＝退回裸奔（耳機輸出時裸奔即可）。

門控狀態機（v14.2 明文規格；法醫腳本 bank_gate_forensics.py 對 dump 驗收）：
  訊號：face＝mediapipe 這一幀有臉；mouth_on＝開口度過閾（開 0.015/閉 0.008
  遲滯）；hb＝鏡頭 worker 心跳（<1s 算活）。
  狀態（f0 是否算他的）：
    S1 唱歌可用   face＋mouth_on            → f0 有效（last_open 刷新）
    S2 寬限       last_open 距今 <0.8s      → f0 有效（子音閉唇/低頭不斷線）
    S3 關門       last_open ≥0.8s（含走開） → f0 一律 0（回授/房聲免疫）
    S4 鏡頭死     hb 斷 >1s                 → 裸奔（f0 全收）＋終端警告
  轉移時間常數：S1→S3 恰 0.8s；S3→S1 ≤ 嘴 worker 週期 ~33ms＋偵測窗；
  S4 進出各印一行。設計裁決（08-12 Harry）：臉不在＝關門（不是 fail-open）。

跑: cd harmony && ../../../260724_ddsp_svc_6x/venv/bin/python bank_live.py \
      [--in-name "USB PnP"] [--lock 0.5] [--gain 1.0]
"""
import argparse
import hashlib
import os
import signal
import threading
import time

# 非互動 shell 用 `&` 背景啟動時 SIGINT 繼承 SIG_IGN，CPython 就不裝
# KeyboardInterrupt handler ＝ Ctrl-C/kill -INT 全聾、收場 dump 陪葬
# （08-12 判決場 r3/r4 實案）。這裡無條件裝回，收場永遠走得到 dump。
signal.signal(signal.SIGINT, signal.default_int_handler)

import numpy as np
import soundfile as sf

SR, HOP = 44100, 512
MAJ = [0, 2, 4, 5, 7, 9, 11]

ap = argparse.ArgumentParser()
ap.add_argument("--in-name", default="USB PnP")
ap.add_argument("--out-name", default="MacBook Pro的揚聲器")
ap.add_argument("--sblock", type=int, default=512)
ap.add_argument("--hop", type=float, default=0.035,
                help="偵測/合成粒度（秒）。窗＝2×hop：35ms→70ms 窗仍蓋住 "
                     "65Hz 兩週期；換音全程 ~70ms（原 46ms 時 ~92ms）")
ap.add_argument("--gain", type=float, default=1.0)
ap.add_argument("--out-map", default=None,
                help="每聲部一個輸出聲道，逗號分隔，順序＝VOICES 現值（--tenor 1"
                     " 時＝bass,tenor,alto,sop，例 '0,1,2,3'）。整聲部（領唱＋"
                     "團員＋pad）進同一道；殘響 mono 匯流排回灌所有聲道。"
                     "只路由不加寬（G11）；要單一多聲道裝置（F7）。"
                     "None＝原立體聲 pan 路徑逐位元不變。")
ap.add_argument("--lock", type=float, default=0.5,
                help="和弦鎖定量：0=天使唱平均律、1=完全跟他的 cents")
ap.add_argument("--attack", type=float, default=0.12, help="起音（秒）")
ap.add_argument("--release", type=float, default=0.25, help="釋音（秒）")
ap.add_argument("--vowel-src", default="scratchpad/take17.wav")
ap.add_argument("--vowels", type=int, default=0,
                help="v17 連續母音場（0=關＝行為完全不變）：每聲部多渲 5 個"
                     "母音層音庫，live 用 mediapipe 嘴形連續值對五錨點反距"
                     "離加權、逐 hop 混合五層＝嘴形連續變天使跟著變。開此"
                     "旗標＝走另一組 bank_ 快取目錄")
ap.add_argument("--layers", default="",
                help="v20 只留部分母音層（逗號分隔 index；空=全部）。"
                     "v22 起 index＝0欸 1咿 2嗚 3啊（喔渲不出來已除名）。"
                     "Harry 08-13：「為了對應口型、音的轉換變來變去不好"
                     "聽」＝母音愈少切換愈少。例：--layers 1,3＝只留咿/啊")
ap.add_argument("--vw-tau", type=float, default=0.0,
                help="v20 權重平滑時間常數（秒；0=關＝跟嘴形一樣快）。"
                     "層數減完仍覺得音色飄＝把這個開到 0.15-0.3")
ap.add_argument("--young", type=float, default=0.0,
                help="v26 年輕音保護（秒；0=關＝舊行為）：剛換過去的音在這"
                     "段時間內，要換走需要多 --young-need 票、且不給搶拍的"
                     "一票通道。彈跳都發生在音很年輕的時候，唱穩之後完全"
                     "不受影響＝砍來回但不犧牲反應。取代 v24 的音高死區"
                     "（三份審查判死：靜默漏音／永久卡音／滑音率飆升）")
ap.add_argument("--young-need", type=int, default=2,
                help="v26 年輕音要多幾票才准換（配合 --young）")
ap.add_argument("--deadzone", type=float, default=0.0,
                help="v26 音高死區（半音；0=關）：離現行音不到 0.5+此值就"
                     "算「他還在這個音」（清候選票）。**上限 0.5**——逃逸"
                     "門檻 ≤1.0 半音＝鄰音中心，保證可達，不會像 v24 那樣"
                     "卡音/漏音。擋的是慢速跨格線游移（票數擋不住的那種）")
ap.add_argument("--maxlag", type=int, default=3,
                help="v25 output backlog cap (blocks; 1 block = 35ms). 3 = "
                     "old behaviour: one worker stall permanently adds up to "
                     "105ms and xrun cannot see it. 1 = latency clamped to "
                     "35ms, at the cost of a 5ms fade-in when a block drops")
ap.add_argument("--rebound", type=float, default=0.25,
                help="v24 回跳防護窗（秒）：多久內回到前一個音要算顫音抖動"
                     "（需 4 票／擋搶拍）。⚠ 審查 A4：拉大到 0.6 會讓「換氣"
                     "後回唱前一個音」多等 ~70ms，且蓋住 ♩=100 的八分音符"
                     "鄰音音型——預設留 0.25，要動先用台架量")
ap.add_argument("--rebuild", action="store_true", help="強制重渲音庫")
ap.add_argument("--mouth", type=int, default=1, help="嘴部門控（0=關）")
ap.add_argument("--mouth-open", type=float, default=0.015,
                help="v27 開口門檻（內唇距/臉高）：要張到這麼開才算他在唱。"
                     "Harry 08-13：「微小張嘴就觸發」＝調高這個。原寫死 0.015")
ap.add_argument("--mouth-close", type=float, default=0.008,
                help="v27 閉口門檻（遲滯下緣，要 < --mouth-open）。原寫死 0.008")
ap.add_argument("--min-level", type=float, default=-100.0,
                help="v33 偵測音量門檻（dBFS；-100=關＝舊行為）。自相關"
                     "只測週期性、對音量無感——08-13 dump 實測閉嘴時 mic "
                     "只有 -54dBFS 仍有 43%% 的幀報出音高＝門控寬限一開，"
                     "房間噪音就讓天使唱起來。他唱歌是 -35dBFS，取 -45 "
                     "乾淨分開。（-100 是 08-11 麥克風壞在 -61dBFS 時代的"
                     "遺留設計，麥克風修好後就變成漏洞）")
ap.add_argument("--bleed", type=float, default=0.0,
                help="v33 天使→麥克風的耦合量（dB，負值；0=關）。實測 -18。"
                     "偵測要求 mic 比「天使當下輸出＋此耦合」再高 "
                     "--bleed-margin dB 才算是他在唱＝張嘴不唱時喇叭漏音"
                     "不會自己觸發")
ap.add_argument("--bleed-margin", type=float, default=6.0,
                help="v33 高過漏音多少 dB 才算數")
ap.add_argument("--balance", type=int, default=0,
                help="v32 依實測音庫響度自動拉齊各聲部（0=關）。設定的"
                     "渲染增益與實際輸出對不上——每顆嘴天生響度不同，"
                     "08-13 實測 sop 比 bass 低 7.0dB、tenor 低 3.8dB、"
                     "alto 高 0.5dB")
ap.add_argument("--trim", default="",
                help="v32 逐聲部微調（dB，逗號分隔，如 sop=+2,bass=-1）"
                     "＝耳朵說了算的那一層，疊在 --balance 之上")
ap.add_argument("--tenor", type=int, default=0,
                help="v31 多開一個 tenor 聲部；0=關＝回三聲部。v42 起領唱嘴"
                     "＝reflow-male8 spk7（不再是同一顆 bass 嘴——bass1 在 "
                     "65-68 是壞音）、音域 43-68、位移 +6（--vl 2 下上聲部"
                     "的位移只剩 fallback 會讀，實際音由 voice-leading 決定）")
ap.add_argument("--gate-log", type=int, default=0,
                help="v34 每次門控開/關印一行（含當下機率）＝誤觸"
                     "可事後定位，不必靠回憶")
ap.add_argument("--mic-gate", type=int, default=0,
                help="v43 麥克風門（0=關；08-18 Harry：live 只用麥克風就能"
                     "觸發）：門的權威從鏡頭改成 mic 電平。hop RMS 要贏過 "
                     "max(--mic-open, 最近 8 hop 輸出電平+--bleed+"
                     "--bleed-margin) 且連續 --mic-frames 個 hop 才開門；"
                     "低於地板 --mic-hold 秒才關。開著時鏡頭照跑但只當畫面"
                     "（view/frame-b64），看門狗與 sing-gate 不再管門。"
                     "與鏡頭門不同：--file 台架也生效＝門控可離線驗證。"
                     "誠實邊界（F26）：能量域判別抓真唱 89%%／把回授誤判"
                     "在唱 37%%，生死判準在耳測。")
ap.add_argument("--mic-open", type=float, default=-45.0,
                help="麥克風門絕對開門電平 dBFS（hop RMS）。閉嘴底噪實測 "
                     "-54、--min-level 凍結 -50，預設再留 5dB 餘裕")
ap.add_argument("--mic-frames", type=int, default=2,
                help="開門要連續 N 個 hop 過地板（2×35ms=70ms）＝單 hop "
                     "瞬態（衣物摩擦、點擊）不開門，語意同 --bs-frames")
ap.add_argument("--mic-hold", type=float, default=0.8,
                help="低於地板多久才關門（秒）＝字間、子音的短暫落下不斬"
                     "句，值抄鏡頭門的 0.8s 寬限。內部以 hop 計數換算＝"
                     "--file 台架不受牆鐘影響")
ap.add_argument("--sing-gate", type=int, default=0,
                help="v34 用訓練出來的『他在不在唱』模型當門控（53 參數，"
                     "scratchpad/sing_gate_train.py 產）。手挑係數是一個"
                     "母音一個母音打地鼠：jawOpen 撈不到咿（下巴不開）、"
                     "pucker 撈不到啊。實測七態機率：閉嘴 0.000／微張 "
                     "0.019／講話 0.008 vs 唱啊 0.986／咿 1.000／嗚 1.000／"
                     "欸 0.996＝門檻 0.30 時唱過 100%%、誤觸 0.7%%")
ap.add_argument("--gate-model", default="auto",
                help="v40 門控模型路徑；auto＝有 sing_gate_model3.npz 就用它"
                     "（三場次＋自校＋運動否決），沒有就退回 sing_gate_model"
                     "（08-13 單場次）。要強制回舊的就明寫路徑")
ap.add_argument("--gate-calib", type=float, default=5.0,
                help="v40 開場基線自校秒數（**這段要閉嘴**，門一律關）。"
                     "修的是場次飄移：同一個人同一個動作 jawOpen 靜止值"
                     "08-13=0.031 / 08-15=0.017＝閉嘴誤觸 44%%。只有新契約"
                     "的模型（calib=1）會用它")
ap.add_argument("--gate-motion-th", type=float, default=-1.0,
                help="v40 運動否決門檻（|ΔjawOpen| 0.5s 窗均值）；"
                     "<0＝用模型內建值（0.0025）。講話實測 0.0079-0.0173、"
                     "唱歌最高 0.0023。調小＝更擋講話但唱歌漏接變多"
                     "（實測 0.0020 時講話 0%%、漏接 17.5%%）")
ap.add_argument("--sing-open", type=float, default=0.7,
                help="v34 起唱門檻（機率）")
ap.add_argument("--sing-hold", type=float, default=0.3,
                help="v34 維持門檻（遲滯下緣）")
ap.add_argument("--bs-gate", type=int, default=0,
                help="v30 用 mediapipe blendshape 當門控（0=關＝用手算幾何）。"
                     "四態實測：mouthPucker 嗚 0.932 vs 閉嘴/微張 0.415＝"
                     "分離度 2.25（全 52 個 blendshape 最高，贏過手算嘴寬 "
                     "2.07、開口值 1.39）；門檻 0.60 時嗚過 94%%、誤觸 3%%。"
                     "**副作用：用嗚起唱也能開門了**（pucker 在張嘴前就到位）")
ap.add_argument("--bs-open", type=float, default=0.20,
                help="v30 jawOpen 起唱門檻（實測：微張 0.131、唱啊 0.343）")
ap.add_argument("--bs-close", type=float, default=0.08,
                help="v30 jawOpen 維持門檻（遲滯下緣）")
ap.add_argument("--bs-pucker", type=float, default=0.75,
                help="v30.1 嘟嘴門檻。⚠ 訂門檻要看**尾巴不是中位數**："
                     "閉嘴 pucker 中位 0.493 但 max 0.682，門檻 0.60 時每秒"
                     "都有幀越線、而**一幀就開門 0.8s**＝Harry 實測「閉著嘴"
                     "還是會觸發」。0.70 以上閉嘴/微張誤觸 0.0%%、唱嗚仍過 80%%")
ap.add_argument("--bs-frames", type=int, default=2,
                help="v30.1 開門防抖：要連續這麼多幀成立才開（單幀雜訊開門"
                     "0.8s 是上一版的失敗模式）；維持不受影響")
ap.add_argument("--mouth-narrow", type=float, default=0.0,
                help="v28 嘟嘴保持（嘴寬/臉寬；0=關）：**已經在唱**時，只要"
                     "嘴比這個窄就維持開門——唱嗚的開口值只有 0.008（跟閉嘴"
                     "的 0.006 分不開）但嘴寬 0.279 vs 閉嘴 0.341＝窄 18%%，"
                     "這是唯一分得開的維度。08-13 三態實測。開門仍只看張口"
                     "＝不增加誤觸風險；代價＝**用嗚起唱仍然開不了門**（誠實"
                     "邊界：鏡頭看不見嘟嘴與閉嘴的差別，只看得見寬窄）")
ap.add_argument("--view", type=int, default=1,
                help="嘴部監看視窗（鏡頭＋內唇線＋開口度＋門控狀態；0=關）")
ap.add_argument("--cam", type=int, default=-1, help="鏡頭 index（-1=自動）")
ap.add_argument("--vl", type=int, default=1,
                # v36：2 = 新版合唱配置（音域向心＋禁同音＋禁交叉）

                help="v4 voice-leading：alto/sop 各自選離上一個音最近的和弦音"
                     "（三/五/八度系候選）＝聲部小步進行；0=關＝固定 +12/+三度"
                     "平行跳")
ap.add_argument("--mem-real", type=int, default=0,
                help="v44 團員可否借別顆模型的真人歌手（0=一律複製領唱＝"
                     "v38 行為）。**08-16 判決後預設改 0**：兩份不同錄音的"
                     "同一個音疊加＝必然的微小音高差＝「一下一下的雜音」"
                     "（干涉，五支儀器全照不到），且音樂上零損失（Harry 耳裁"
                     "真人音色與失諧複製分不太出來）。凍結配置也是 0；"
                     "要試真人音色必須明寫 1，且先逐音重驗 MEM_SPK 名單")
ap.add_argument("--vib-sync", type=int, default=0,
                help="v43 團員與領唱共用同一個顫音相位（1=同步）。0＝各自"
                     "隨機相位（原行為）＝同一個音的副本瞬時頻率互相錯開，"
                     "在高諧波上落進 15-300Hz 差頻區＝roughness（聽得到、"
                     "頻譜找不到，因為它不是新成分而是干涉）")
ap.add_argument("--porta", type=float, default=0.08,
                help="v15 portamento：**前音停留 ≥0.3s** 且級進 ≤2 半音才滑"
                     "（慢轉音＝表情滑音；正常唱速＝乾淨換音）。線性定長"
                     "（秒；0=關）")
ap.add_argument("--fold", type=int, default=1,
                help="v37 音域外折八度（保留音級）而不是 clip（會改音級）")
ap.add_argument("--vib", type=float, default=0.0,
                help="v37 顫音深度（cents，0=關）。審查實測天使的 3-8Hz "
                     "能量只有他的 2.2%%＝完全沒有顫音，是「修過音」最強"
                     "的指紋。建議 20-30")
ap.add_argument("--vib-rate", type=float, default=5.5, help="顫音速率 Hz")
ap.add_argument("--vib-delay", type=float, default=0.35,
                help="音齡超過這麼久才漸入顫音（換音當下要乾淨）")
ap.add_argument("--human", type=float, default=6.0,
                help="v3 人味離散：每聲部獨立微音準慢漂（±cents；0=關）"
                     "＝殺 pad 同質感（v6② humanization）")
ap.add_argument("--per-part", type=int, default=1,
                help="v38 每聲部幾個人（1=現況）。團員**共用同一份音庫**——"
                     "不重渲、不動 VOICES（快取 key 是 repr(VOICES)，動它＝"
                     "十五個層庫全部重渲），只在播放層多開幾個游標。差別＝"
                     "靜態失諧＋起音錯開＋各自的微漂/顫音相位＋站位微散。"
                     "FINDINGS F3：約 7 條去相關的線 ≈ 10 個真人，再多遞減")
ap.add_argument("--spread-cents", type=float, default=25.0,
                help="v38 團員間靜態失諧 std（cents）。F3 甜蜜點 ~25c"
                     "（真團團員間 F0 散布 0-50c、平均 ~20c）")
ap.add_argument("--spread-ms", type=float, default=20.0,
                help="v38 團員起音錯開（ms，均勻 0~2x）。F3：~20ms 是甜蜜點、"
                     ">40ms 反而爛。它同時是延遲線＝把同一份 loop 讀在不同"
                     "相位上，這是去相關的主力（純失諧只會拍頻）")
ap.add_argument("--spread-pan", type=float, default=0.18,
                help="v38 團員站位相對聲部中心的散開量（±，-1~1 座標）")
ap.add_argument("--wet", type=float, default=0.3,
                help="v3 悠遠層：殘響 send（reverb.py IR 串流卷積；0=關）")
ap.add_argument("--pad", type=float, default=1.5,
                help="v6 慢層和弦墊（音庫做的、零 ML）：錨音至少駐留這麼多秒"
                     "才換和弦；0=關（只剩快層）")
ap.add_argument("--pad-gain", type=float, default=0.45, help="慢層音量")
ap.add_argument("--pad-hold", type=float, default=1.5,
                help="你停多久慢層才放（快層 0.25s 就放＝墊會活得比你久）")
ap.add_argument("--predict", default="scratchpad/melody_lm_v0.json",
                help="v8 個人化搶拍：Harry 旋律習慣模型（melody_lm）——換音在"
                     "模型 top-3 內＝一票 commit（涵蓋級進外的慣用跳進）。"
                     "空字串=關（退回級進捷徑）")
ap.add_argument("--fast", type=int, default=1,
                help="v6 搶拍：級進（≤2 半音）且在調內的換音一票 commit"
                     "（~35ms；0=關＝一律兩票 ~70ms）")
ap.add_argument("--frame-b64", type=float, default=0,
                help="每秒 N 幀把嘴部畫面以「FRAME <base64 jpeg>」印到 "
                     "stdout（respond_shell UI 用；0=關＝行為不變）")
ap.add_argument("--dump", default="", help="收錄 mic/out 到 <path>_mic/out.wav")
ap.add_argument("--dump-max-min", type=float, default=30.0,
                help="v44 dump 累積上限（分鐘）。buffer 原本無上限（~32MB/分"
                     "常駐），長 session 收場時 concatenate 會撐爆記憶體或"
                     "超出 shell 的 8s SIGKILL 預算＝整份錄音陪葬。滿了停錄"
                     "並印一行，演出不受影響")
ap.add_argument("--file", nargs=2, metavar=("IN", "OUT"),
                help="離線台架：整檔跑同一條 per-hop 管線（無音訊裝置、"
                     "無鏡頭）＝開發自驗用")
a = ap.parse_args()
# 審查 B2：荒謬旗標值原本靜默通過（--hys 12 實測 60s 只換 7 次音、
# 最長持音 18.45s，程式一聲不吭）。範圍檢查放這裡＝開場就死，
# 不要讓它變成「聽起來怪但查不出原因」。
assert 0 <= a.young <= 2.0, "--young 合理範圍 0-2 秒"
# **0.5 是硬上限不是品味**：死區逃逸門檻 = 0.5+deadzone，超過 0.5
# 就與合理性門（離格 ≤0.35）不相交＝回不去剛離開的音（審查 S2 實測
# 卡死 5 秒）、且 0.55 起會靜默漏掉半音級進（S3 實測每趟漏 3 音）。
# 實測 0.5 已開始吃真音（41 個穩定平台漏 4 個），故上限訂 0.45。
assert 1 <= a.per_part <= 8, "--per-part 合理範圍 1-8（F3：>7 條線遞減報酬）"
assert 0 <= a.spread_cents <= 60, "--spread-cents 合理範圍 0-60（F3：真團 0-50c）"
assert 0 <= a.spread_ms <= 40, "--spread-ms 上限 40（F3：>40ms 起音散開反而爛）"
assert 0 <= a.spread_pan <= 0.5, "--spread-pan 合理範圍 0-0.5"
assert 0 <= a.deadzone <= 0.45, "--deadzone 上限 0.45（見碼內說明）"
assert 0 <= a.young_need <= 6, "--young-need 合理範圍 0-6 票"
assert 0 <= a.rebound <= 2.0, "--rebound 合理範圍 0-2 秒"
assert 1 <= a.maxlag <= 16, "--maxlag 合理範圍 1-16 塊"
# 08-17 review #11：--attack/--release 0 會在 render 的 n/(atk*SR) 除以零＝
# worker 執行緒死、輸出永久靜音而儀表全綠。本檔的「0=關」慣用法不適用這
# 兩個——包絡永遠存在，沒有「關」。
assert a.attack > 0, "--attack 必須 > 0（0 會除以零殺掉音訊 worker）"
assert a.release > 0, "--release 必須 > 0（0 會除以零殺掉音訊 worker）"
# 審查：--bs-frames 0 會讓 `arm >= 0` 恆真＝門永遠開著，與機率無關
assert 1 <= a.bs_frames <= 10, "--bs-frames 至少 1（0 = 門永遠開）"
assert a.sing_hold <= a.sing_open, "--sing-hold 應 <= --sing-open（遲滯）"
assert -120 <= a.min_level <= 0, "--min-level 合理範圍 -120~0 dBFS"   # -100 = 關
assert 0 < a.mouth_close < a.mouth_open < 0.5, \
    "--mouth-close 必須 < --mouth-open（遲滯上下緣）"

# 聲部：(名, 模型, spk, gain, 音庫 MIDI 範圍, 半音位移, 是否加全音階三度)
# ⚠ 快取 key 只吃前五欄（名/模型/spk/gain/音域，見 _ckv）——調位移不重渲。
# v41（08-16）：bass 的移調從 0 改 **-8**。（當天一度因為 live 雜音被回退，
# 雜音後來查出元兇是 **--mem-real**（不同錄音疊加的干涉，見該旗標 help；
# --frame-b64 曾被誤判、已翻案）、與移調無關 → 接回。）Harry「bass tenor 男生的那個共鳴
# 感一直沒有出來」——查下去不是音色也不是音量，是**音域**：08-15 dump 實測
# bass 中位 G#3(56)＝**跟他自己一模一樣**（他 207.7Hz、bass 207.7Hz），
# tenor A#3(58)，兩部男聲只差 2 個半音＝不是兩條線，是一條加厚的線，而且
# 都待在中音區，男低音的共鳴（G2-C3）完全沒有人在。
# 為什麼是 -8 不是 -12：先試 -12 他判「有了」，但掃描全部移調值後 -12 只有
# 54.8% 落在標準 bass 核心(G2-C4)、還有 8.9% 掉出音庫下限被折八度；**-8 是
# 最適解**（核心內 86.4%、掉出 0.4%、不進 71-75 破音區）。低頻能量兩者幾乎
# 相同（<120Hz 13.2% vs 13.8%），所以 -8 拿到一樣的共鳴、沒有折八度瑕疵。
VOICES = [("bass", "reflow-bass1/model_32000.pt", 1, 1.0, (36, 68), -8, False),
          ("alto", "reflow-alto3/model_40000.pt", 2, 0.85, (50, 80), 12,
           False),
          ("sop", "reflow-sop3/model_20000.pt", 1, 0.7, (53, 84), 12, True)]
# v31（Harry：「男生可以多加一個聲部嗎，現在是不是只有 bass」——是，
# 現役三嘴只有 bass 是男聲）。手上沒有第二個男聲權重（bass1 是 n_spk=1，
# 只有 alto3 是三嗓合訓可換 spk_id），最短路徑＝**同一顆 bass 嘴多開一
# 個聲部唱不同的線**：真實合唱團本來就同嗓分部，加上各聲部獨立的人味
# 微漂（HUM）與站位（PAN）就是兩個人。真的要新音色＝訓 M4Singer 的
# Tenor-1~7 / Bass-2,3（都還沒用過），那要 GPU。
if a.tenor:
    # 上緣 68 不是 75：這個上限是 bass1 時代訂的（bass1 渲到 71-75 是破音
    # 區：midi 75 迴圈內音高 SD 59.9 cents、週期性 0.51，正常音 0.99，而
    # tenor 曾有 37% 的時間待在那裡）。v42 換 male8 spk7 後 65-68 實測乾淨
    # （週期性 0.987-0.990），**68 以上沒量過**——要拉高上緣先逐音掃，別
    # 沿用舊上限也別盲目放寬。
    # v42（08-16）：領唱嘴從 **bass1 換成 male8 spk7（Tenor-5）**。
    # Harry live「唱 B3 出現雜音、唱 C4 沒有、唱 A3 沒有」＋「B3 也有正常的
    # 時候」→ 逐音掃音庫抓到病灶：**bass1 在 MIDI 65-68 是壞音**（週期性
    # 0.790-0.881、音高 SD 14-25 cents，正常音是 0.99），而 voice leading
    # 依和聲脈絡決定當下配哪個音 ⇒ 同一個唱名有時撞上、有時沒有＝「時有時無」。
    # dump 實測撞壞音比例：舊配置 bass 18.04%／tenor 5.15%，v41 後 bass 0%／
    # tenor 3.65%（所以 v41 其實已經改善了 bass，剩下的是 tenor）。
    # 為什麼換模型而不是縮音域：上緣降到 64 雖然也能歸零，代價是 3.65% 的音
    # 被折八度＝用一個瑕疵換另一個。而 **male8 spk7 在同一段音域全部乾淨**
    # （65-68 週期性 0.987-0.990、SD 5.0-7.6c），且那顆權重已經在硬碟上
    # （團員一直在用它），零額外成本。
    # v31 當初用 bass1 是因為「手上沒有第二個男聲權重」——08-15 訓完 male8
    # 之後那個前提就不成立了，這裡是補上那次沒跟到的改動。
    VOICES.insert(1, ("tenor", "reflow-male8/model_40000.pt", 7, 0.8,
                      # v41（08-16）：+7 → **+6**。掃描顯示 +6 落核心 (E3-E4)
                      # 比例最高（96.3% vs +7 的 94.9%）；與 bass(-8) 的間距
                      # ＝14 半音（08-17 更正：原註「+9」是算錯的）。
                      # ⚠ 08-17 review #10：**--vl 2 下這個位移是惰性的**——
                      # sh 只有 vi==0（bass）與空候選 fallback 會讀，上聲部
                      # 實際唱哪個音由 voice-leading 的音域/評分決定，所以那
                      # 個 96.3% 描述的是掃描器的口徑、不是現行系統；要動
                      # tenor 的音區，真正的旋鈕是音域 (43,68) 與其中心。
                      (43, 68), 6, False))
# --out-map（08-18 Harry：「neural/live/live+neural 都要能分聲道」）。要在
# VOICES 定案（--tenor 插入）之後才解析，長度才對得上。None＝下面所有
# OMAP 分支都不走＝原路徑逐位元不變（Regime A，--file 前後比對驗證）。
OMAP = None
NCH = 2
if a.out_map:
    OMAP = [int(t) for t in str(a.out_map).split(",")]
    if len(OMAP) != len(VOICES) or min(OMAP) < 0:
        raise SystemExit(f"--out-map 要 {len(VOICES)} 個非負聲道（順序 "
                         f"{','.join(nm for nm, *_x in VOICES)}），"
                         f"拿到 {a.out_map!r}")
    NCH = max(max(OMAP) + 1, 2)
NOTE_S = 2.0                             # 每音渲染秒數
LOOP_A, LOOP_B = int(0.5 * SR), int(1.8 * SR)   # 循環區間
XF = int(0.05 * SR)                      # 循環回捲 crossfade

# v16 多母音層（--vowels 開才用）：0619 素材的明暗五層，Harry 08-13 耳測
# 認可。(tag, src_wav, t0, t1, src_tilt_dB)。⚠ src_tilt 欄位 v17 起已無
# 消費者（tilt 選層判死），但整個 tuple 的 repr 進快取 hash（下方 _ck）
# ——動任何一欄＝十五個層庫全部重渲，別動。
# ⚠ 這條路徑的字串進 VOWEL_LAYERS 的 repr、再進下方 _ck 的快取 hash。
# 換機器＝快取 key 換一次＝層庫重渲一次（凍結配置不帶 --vowels，不影響上台）。
import sys as _sys, pathlib as _pl  # noqa: E402
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
from config import VOWEL_CLIPS as _VOWEL_CLIPS  # noqa: E402
_C0619 = str(_VOWEL_CLIPS)
# v21（08-13 驗屍後重挑）：**依母音挑，不再依亮度**。舊版五層是照
# spectral tilt 排序撈的，實測母音＝[嗚,喔,咿,喔,啊]——沒有欸、喔佔三席，
# 「喔欸咿嗚啊」標籤純屬順序假設（Harry 三次耳判逐條命中：咿啊順、欸分
# 不出來、喔嗚沒試出來、喔聽起來像欸）。新選法＝vowel_layer_rebuild：
# 對語料每 300ms 窗量 F1/F2 → 併同母音連續段 → 取 ≥0.7s 最像的 run 的
# 中心 ±0.5s（渲染 encode 會吃窗中心 ±0.93s 上下文，短窗會混進隔壁音素）。
# 括號內＝該段實測 (F1,F2)；順序必須與校準 labels 同序（層 i ↔ 類 i）。
# ⚠ v22 選法改成**渲後選**（vowel_render_select）：源端像不像不是判準
#   ——Harry 現場錄的嗚源端 d=0.11（全場最準）渲完仍歪，語料某段源端普通
#   渲完卻乾淨＝哪段 units 活得過模型像抽籤。所以候選全渲一遍、用渲後
#   距離挑。存活率（每母音 30-38 候選）：啊 33、欸 29、咿 29、嗚 6、
#   **喔 0＝判死**（最佳候選也只渲成嗚 381/943）。這是模型邊界不是
#   發音問題，別再拿他的耳朵試。
# ⚠ v23 血訓（Harry「這版本不如上個」，當場量出四層 HNR 全掉 1-6dB）：
#   只用共振峰距離挑＝優化了準確度、賠掉乾淨度（同一個 clip 只差窗口
#   位置就差 6dB）。判準改**兩段式**：先過母音正確＋d≤0.25，再取 HNR
#   最高者。現行四層有兩層來自他 08-13 現場錄音（欸/嗚），憑音質贏。
VOWEL_LAYERS = [("喔", f"{_C0619}/clip_0022.wav", 1.75, 2.75, 0.0),   # 479/700
                ("欸", f"{_C0619}/clip_0143.wav", 1.67, 2.67, 0.0),   # 609/1749
                ("咿", f"{_C0619}/clip_0155.wav", 3.47, 4.47, 0.0),   # 264/2331
                ("嗚", f"{_C0619}/clip_0035.wav", 2.75, 3.75, 0.0),   # 320/794
                ("啊", f"{_C0619}/clip_0009.wav", 0.93, 1.93, 0.0)]   # 717/1101
# ⚠ Harry 08-13 耳裁：**回 v21 這組**（HNR 量測說 v23 那組每層乾淨 1-3dB，
# 他聽起來仍偏好這組＝指標與耳朵不一致時以耳朵為準，量測只擋工程退化）。
# v23 那組（欸/咿/嗚/啊，兩層來自他現場錄音）留在 commit 10bdf953 可回。
# 分類器的五個類別（校準儀式 lip_ring_calib.LABELS 同序）——層可以比類別
# 少（喔渲不出來），所以層→類別要明寫，不能再靠 index 相同的巧合。
VOWEL_NAMES = ["喔", "欸", "咿", "嗚", "啊"]
L_MID = 4                                # 無鏡頭/退路層＝啊（正典音庫同款）
# v20 母音子集（--layers）：模型仍出五類機率，取子集後重正規化＝
# 「在這幾個母音之中他唱的是哪個」的條件機率。庫/快取一個 byte 不動，
# 換子集不用重渲。
_LSEL = list(range(len(VOWEL_LAYERS)))
if a.layers.strip():
    _LSEL = sorted({int(s) for s in a.layers.split(",") if s.strip() != ""})
    assert _LSEL and all(0 <= i < len(VOWEL_LAYERS) for i in _LSEL), _LSEL
_NSEL = len(_LSEL)
# v22：層 → 分類器類別（層數 < 類別數，喔沒有層＝它的機率被重正規化掉）
_LCLS = [VOWEL_NAMES.index(t) for (t, *_r) in VOWEL_LAYERS]
assert len(set(_LCLS)) == len(_LCLS), "兩層對到同一個母音類別"
_LDEF = _LSEL.index(L_MID) if L_MID in _LSEL else _NSEL // 2   # 退路層
# v17：無鏡頭/無校準檔時的權重退路＝預設層一熱向量
_VW_MID = np.zeros(_NSEL, dtype="float32")
_VW_MID[_LDEF] = 1.0


def _sub(p5):
    """分類器五類機率 → 在用層的權重（取對應類別後重正規化；近零＝退
    預設層）。喔沒有層＝它的機率在這裡被歸掉＝「在做得到的母音之中，
    他唱的是哪個」。"""
    q = p5[[_LCLS[i] for i in _LSEL]]
    s = float(q.sum())
    return (q / s).astype("float32") if s > 1e-6 else _VW_MID


_SEL_NAMES = [VOWEL_NAMES[_LCLS[i]] for i in _LSEL]   # 層 index → 母音名
if a.vowels:
    print(f"[field] vowel set {'/'.join(_SEL_NAMES)}"
          + (f", weight smoothing {a.vw_tau}s" if a.vw_tau > 0 else ""),
          flush=True)

# vowels 開＝快取 key 納入層定義＋"vowels1"＝落到**不同的 bank_ 目錄**
# ＝現行快取一個 byte 都不動＝隨時 --vowels 0 退回
# 08-17（review #3）：key 只取**會進渲染**的欄位（名/模型/spk/gain/音域）。
# sh 與 th 是播放層的和聲邏輯，改它們渲出來的音庫一個 byte 都不變——舊 key
# 用 repr(VOICES) 整包，v41/v42 調移調時八個 bank 全部陪葬重渲（~1 分鐘/個），
# 首次切進 bank 模式還會撞 respond_shell 的 30s 切換看門狗。
_ckv = [(nm, mdl, sp, g, rng) for nm, mdl, sp, g, rng, _sh, _th in VOICES]
_ck = hashlib.md5((repr(_ckv) + a.vowel_src + str(NOTE_S) + "perloop1"
                   + (repr(VOWEL_LAYERS) + "vowels1" if a.vowels else ""))
                  .encode()).hexdigest()[:10]
BANK_DIR = f"scratchpad/bank_{_ck}"
BANKS = {}                               # name -> {midi: np.ndarray(loop 段)}
LBANKS = {}                              # v16: name -> [層][midi] -> ndarray
XBANKS = {}                              # v39: (name, model, spk) -> {midi: …}
# v38 團員音色（--per-part）：**其他 speaker 就是真的其他歌手**（M4Singer
# 的不同人合訓）。08-14 耳審 reflow-alto3（n_spk=3）：
#   spk1 / spk2 ＝兩個人、都乾淨（Harry：「聽起來有像不同人」）
#   spk3 **判死**——「明顯非人聲雜音以及音扭曲」。旁證：它的訓練資料最少
#     （1182 檔 vs 1890/1933）、質心 3564Hz 三者最高＝高頻垃圾。沒吃飽。
#
# v39（08-15）：欄位從 spk 擴成 **(模型, spk)** ＝借的人不必再跟領唱同一顆
# 模型。v38 那句「bass/sop 的模型 n_spk=1＝沒有第二個人可借」就此作廢——
# reflow-fem8 / reflow-male8（各 8 位 M4Singer 真人，40000 步）已訓好，
# Harry 08-15 耳裁 fem8「都可以用」、male8「都先過，分得開但不明顯」。
# **領唱刻意不換**（他的裁決）：現行三個聲音是他一路耳裁通過的，viva 前
# 11 天不動既有路徑。所以這是純加法——領唱照舊，團員從「複製領唱＋失諧」
# 升級成真的別人。
#
# 借誰照**音域**配，不照檔案數量（08-15 血訓：檔數不是門檻，455 檔的
# Soprano-2 在 40k 可用）：
#   bass (36-68) ← male8 三位 Bass
#   tenor(43-68) ← male8 的 Tenor-1/3/5（中位 61/57/52＝散開）
#   alto (50-80) ← fem8 的 Alto-1/4/5（避開 Alto-6＝領唱 alto3 spk2 同一人）
#   sop  (53-84) ← fem8 三位 Soprano
# ⚠ 團員音庫是照**該聲部的音域**渲的（lo/hi 取領唱那欄），不是照借來那位
#   的舒適區——F24：音域才是決定音色類型的東西。借高音的人來唱低聲部會
#   進破音區，所以上面才照中位數配，不是隨便抓三個。
_M8 = "reflow-male8/model_40000.pt"   # 1-3 Bass-1/2/3；4-8 Tenor-1/2/3/5/7
_F8 = "reflow-fem8/model_40000.pt"    # 1-3 Soprano-1/2/3；4-8 Alto-1/4/5/6/7
# ⚠ 08-15 耳裁（v39c）：初版一部借三位，Harry live 判「有雜音」，離線逐位
# 單獨聽（各在**該聲部音域**渲一段、等響）判出 **12 位裡 8 位是壞的**：
#   壞 Bass-1 Bass-3 / Tenor-1 Tenor-3 / Alto-1 Alto-4 / Soprano-1 Soprano-2
#   留 Bass-2      / Tenor-5        / Alto-5        / Soprano-3
# **同一批人早上在 probe 全部判可用**（fem8「都可以用」、male8「都先過」）。
# probe 用一組共用音、音庫用整個聲部音域跑滿 ⇒ **probe 的合格不轉移到音庫**，
# 新歌手一律要在音庫這條路徑上重聽一次（F26）。
# 音域假設被自己的資料推翻：Bass-1(壞)/Bass-2(留) 音域統計幾乎相同；被推得
# 最遠的 Tenor-5 反而是留下的那個。唯一與判決相關的是**訓練檔數**——存活四位
# 全 ≥1165 檔、判壞八位除 Bass-1(1758) 外全 ≤1073。當挑候選的線索，不當預測器
# （今天已被三支儀器騙過：週期性、頻譜平坦度、dump）。
MEM_SPK = {"bass":  [(_M8, 2)],      # Bass-2    1656 檔 ⚠ C2-D2 有壞音——
           #   bass -8（v41）之後那段可達；--mem-real 1 之前要先逐音重驗
           "tenor": [(_M8, 7)],      # Tenor-5   1224 檔 ⚠ v42 起＝領唱同一人
           #   （領唱嘴換成 male8 spk7）——播放端會自動排除（見 _avail 的
           #   領唱排除），留在名單只當紀錄；male8 其餘 Tenor 全數耳裁判壞
           #   ＝tenor 現況**沒有**可借的真人
           "alto":  [(_F8, 6)],      # Alto-5    1934 檔
           "sop":   [(_F8, 3)]}      # Soprano-3 1165 檔


def _savez_atomic(path, bank):
    """快取落檔（原子）：先寫 tmp 再 os.replace。np.savez 寫到一半被打斷
    （切換看門狗 SIGINT、關機、磁碟滿）會留下半個 zip——之後每次啟動都在
    np.load 炸掉；更糟的一型是「殘缺但合法」：zip entry 邊界被斷＝短 bank
    靜默載入＝某聲部整場被折進殘存的低八度、無任何錯誤（08-17 review #14
    三種毒法都實際重現過）。"""
    tmp = path + ".tmp.npz"
    np.savez(tmp, **{str(k): v for k, v in bank.items()})
    os.replace(tmp, path)


def _load_bank(path, lo, hi, what):
    """快取載入＋完整性驗證：載得起來、且音符正好蓋滿 lo..hi 才算命中；
    否則回 None＝當 cache miss 重渲（壞檔由 _savez_atomic 原子覆蓋）。"""
    try:
        z = np.load(path)
        bank = {int(k): z[k] for k in z.files}
    except Exception as e:                   # BadZipFile／半截 entry／權限…
        print(f"⚠ [bank] {what} cache unreadable ({e}) → re-rendering",
              flush=True)
        return None
    if sorted(bank) != list(range(lo, hi + 1)):
        print(f"⚠ [bank] {what} cache incomplete ({len(bank)}/{hi - lo + 1} "
              f"notes) → re-rendering", flush=True)
        return None
    return bank


def _render_layer(svc, UU, lo, hi):
    """v16：用給定單元 UU 渲整個音域 → {midi: 純循環段}（--vowels 專用）。

    ⚠ 本體是下面 _build_banks 主渲染迴圈的**刻意逐行複製**——「預設路徑
    零行為變化」是硬約束，優先於 DRY：既有那圈一個字都不動，正典品質就
    不可能被這次改動碰到。渲法（f0/vol/mask/seed/迴圈裁剪）若要改，兩處
    都要改。
    """
    import torch
    nb = UU.size(1)
    NF = int(NOTE_S * SR / HOP)
    bank = {}
    with torch.no_grad():
        for m in range(lo, hi + 1):
            f0 = np.full(NF, 440.0 * 2 ** ((m - 69) / 12.0))
            vol = np.full(NF, 0.06)
            vol[:4] = np.linspace(0, 0.06, 4)      # 去 onset 突波
            vol_t = (torch.from_numpy(vol).float()
                     .to(svc.device)[None, :, None])
            mask = torch.ones(1, NF * HOP, device=svc.device)
            torch.manual_seed(1234 + m)
            au = svc.infer(np.zeros(NF * HOP), 0.0, -60.0,
                           units_override=UU[:, np.arange(NF) % nb],
                           feats=(f0, vol_t, mask), ratios=[None]
                           )[0].cpu().numpy()
            seg = au[LOOP_A:LOOP_B + XF].astype("float32")
            T_ = SR / (440.0 * 2 ** ((m - 69) / 12.0))
            L_ = int(round(round((LOOP_B - LOOP_A) / T_) * T_))
            xfs = int(0.005 * SR)
            lin = np.linspace(0, 1, xfs, dtype="float32")
            out_ = seg[:L_].copy()
            out_[:xfs] = seg[:xfs] * lin + seg[L_:L_ + xfs] * (1 - lin)
            bank[m] = out_
            del au
    return bank


def _build_mem_banks():
    """v38：團員用的「其他歌手」音庫（--per-part ≥2 且該模型 n_spk>1 才跑）。

    ⚠ 渲染本體是 _build_banks 主迴圈的**刻意逐行複製**（同 _render_layer
    的理由）：既有那圈一個字都不動＝正典品質不可能被這次改動碰到。渲法
    （f0/vol/mask/seed/迴圈裁剪）若要改，三處都要改。
    快取檔名帶模型與 spk（`<nm>_<模型>_s<spk>.npz`，v39 起加模型）＝跟領唱
    的音庫、跟別顆模型的同號 spk 都互不覆蓋，也不必動 BANK_DIR 的 hash
    （換模型或換 spk 就是換檔名，不會讀到過期的東西）。
    """
    import torch
    import spike_stream6 as S
    pairs = [(nm, m2, s2) for nm, *_x in VOICES
             for (m2, s2) in MEM_SPK.get(nm, [])]
    if not pairs:
        return
    # v39：模型改由 MEM_SPK 那欄決定，這裡只留領唱的 gain 與**音域**
    spec = {nm: (g, lo, hi)
            for nm, _mdl, _sp, g, (lo, hi), _sh, _th in VOICES}
    mic0, _ = sf.read(a.vowel_src, dtype="float64", always_2d=True)
    mic0 = mic0[:, 0]
    NF = int(NOTE_S * SR / HOP)
    UU = None
    for nm, mdl, sp2 in pairs:
        tag = mdl.split("/")[0]              # reflow-male8 / reflow-fem8 …
        path = f"{BANK_DIR}/{nm}_{tag}_s{sp2}.npz"
        g, lo, hi = spec[nm]
        if os.path.exists(path) and not a.rebuild:
            bk0 = _load_bank(path, lo, hi, f"{nm} {tag} spk{sp2}")
            if bk0 is not None:
                XBANKS[(nm, mdl, sp2)] = bk0
                print(f"[bank] {nm} {tag} spk{sp2} cache hit "
                      f"({len(bk0)} notes)", flush=True)
                continue
        svc = S.Svc([(f"{S.DDSP}/exp/{mdl}", 0.0, g, sp2)],
                    step=2, t_start=0.85)
        if UU is None:
            _f, _v, _m, uv0 = svc.prep(mic0[:60 * SR], -60.0, want_uv=True)
            runs, i = [], 0
            while i < len(uv0):
                if not uv0[i]:
                    j = i
                    while j < len(uv0) and not uv0[j]:
                        j += 1
                    runs.append((i, j))
                    i = j
                else:
                    i += 1
            a0, b0 = max(runs, key=lambda r: r[1] - r[0])
            mid = (a0 + b0) // 2
            with torch.no_grad():
                UU = svc.encode(
                    mic0[max(0, (mid - 40) * HOP):(mid + 40) * HOP])[:, 8:-8]
        nb = UU.size(1)
        bank = {}
        t0 = time.time()
        with torch.no_grad():
            for m in range(lo, hi + 1):
                f0 = np.full(NF, 440.0 * 2 ** ((m - 69) / 12.0))
                vol = np.full(NF, 0.06)
                vol[:4] = np.linspace(0, 0.06, 4)
                vol_t = (torch.from_numpy(vol).float()
                         .to(svc.device)[None, :, None])
                mask = torch.ones(1, NF * HOP, device=svc.device)
                torch.manual_seed(1234 + m)
                au = svc.infer(np.zeros(NF * HOP), 0.0, -60.0,
                               units_override=UU[:, np.arange(NF) % nb],
                               feats=(f0, vol_t, mask), ratios=[None]
                               )[0].cpu().numpy()
                seg = au[LOOP_A:LOOP_B + XF].astype("float32")
                T_ = SR / (440.0 * 2 ** ((m - 69) / 12.0))
                L_ = int(round(round((LOOP_B - LOOP_A) / T_) * T_))
                xfs = int(0.005 * SR)
                lin = np.linspace(0, 1, xfs, dtype="float32")
                out_ = seg[:L_].copy()
                out_[:xfs] = seg[:xfs] * lin + seg[L_:L_ + xfs] * (1 - lin)
                bank[m] = out_
                del au
                if (m - lo) % 8 == 7:        # 進度列印＝餵 shell 的切換看門狗
                    print(f"[bank] {nm} {tag} spk{sp2} rendering "
                          f"{m - lo + 1}/{hi - lo + 1}", flush=True)
        _savez_atomic(path, bank)
        XBANKS[(nm, mdl, sp2)] = bank
        print(f"[bank] {nm} {tag} spk{sp2} rendered {len(bank)} notes "
              f"({time.time() - t0:.0f}s) → {path}", flush=True)


def _norm_layers(nm):
    """v17（審查 A4）：同 midi 各層 rms 對齊 mid 層——實測層間響度差可達
    4dB，權重和恆 1 仍會「改口型＝改音量」。只動記憶體不動快取檔；mid 層
    一個 byte 不動＝一熱退化仍與單層逐位元同。"""
    for m_, ref in LBANKS[nm][L_MID].items():
        r0 = float(np.sqrt((ref ** 2).mean()))
        for li in range(len(VOWEL_LAYERS)):
            if li == L_MID:
                continue
            b_ = LBANKS[nm][li][m_]
            r_ = float(np.sqrt((b_ ** 2).mean()))
            if r_ > 1e-9:
                LBANKS[nm][li][m_] = (b_ * (r0 / r_)).astype("float32")


def _build_banks():
    """正典渲音庫（首次 ~1 分鐘；之後快取秒開）。"""
    import torch
    import spike_stream6 as S
    os.makedirs(BANK_DIR, exist_ok=True)
    mic0, _ = sf.read(a.vowel_src, dtype="float64", always_2d=True)
    mic0 = mic0[:, 0]
    UU = None
    NF = int(NOTE_S * SR / HOP)
    for nm, mdl, sp, g, (lo, hi), _sh, _th in VOICES:
        path = f"{BANK_DIR}/{nm}.npz"
        # v16：層庫各存一檔；要全齊才算命中（缺一就整個 voice 走渲染補齊）
        lp = ([f"{BANK_DIR}/{nm}_L{li}.npz"
               for li in range(len(VOWEL_LAYERS))] if a.vowels else [])
        if (os.path.exists(path) and all(os.path.exists(p) for p in lp)
                and not a.rebuild):
            bk0 = _load_bank(path, lo, hi, nm)
            lb0 = []
            if bk0 is not None and a.vowels:
                for li, p_ in enumerate(lp):
                    b_ = _load_bank(p_, lo, hi, f"{nm} L{li}")
                    if b_ is None:
                        lb0 = None           # 任一層壞＝整個 voice 走重渲
                        break
                    lb0.append(b_)
            if bk0 is not None and lb0 is not None:
                BANKS[nm] = bk0
                if a.vowels:
                    LBANKS[nm] = lb0
                    _norm_layers(nm)
                print(f"[bank] {nm} cache hit ({len(bk0)} notes)", flush=True)
                continue
        svc = S.Svc([(f"{S.DDSP}/exp/{mdl}", 0.0, g, sp)],
                    step=2, t_start=0.85)
        if UU is None:
            _f, _v, _m, uv0 = svc.prep(mic0[:60 * SR], -60.0, want_uv=True)
            runs, i = [], 0
            while i < len(uv0):
                if not uv0[i]:
                    j = i
                    while j < len(uv0) and not uv0[j]:
                        j += 1
                    runs.append((i, j))
                    i = j
                else:
                    i += 1
            a0, b0 = max(runs, key=lambda r: r[1] - r[0])
            mid = (a0 + b0) // 2
            with torch.no_grad():
                UU = svc.encode(
                    mic0[max(0, (mid - 40) * HOP):(mid + 40) * HOP])[:, 8:-8]
        nb = UU.size(1)
        bank = {}
        t0 = time.time()
        with torch.no_grad():
            for m in range(lo, hi + 1):
                f0 = np.full(NF, 440.0 * 2 ** ((m - 69) / 12.0))
                vol = np.full(NF, 0.06)
                vol[:4] = np.linspace(0, 0.06, 4)      # 去 onset 突波
                vol_t = (torch.from_numpy(vol).float()
                         .to(svc.device)[None, :, None])
                mask = torch.ones(1, NF * HOP, device=svc.device)
                torch.manual_seed(1234 + m)
                au = svc.infer(np.zeros(NF * HOP), 0.0, -60.0,
                               units_override=UU[:, np.arange(NF) % nb],
                               feats=(f0, vol_t, mask), ratios=[None]
                               )[0].cpu().numpy()
                # 取循環區、回捲 crossfade 藏接縫 → 存純循環段
                seg = au[LOOP_A:LOOP_B + XF].astype("float32")
                # R2B v2：整數週期迴圈——頭尾相關實測 0.96＝近週期素材，
                # 等功率縫反而 +3dB 凸包；迴圈長取基頻週期整數倍＝相位
                # 對齊，殘餘噪聲成分 5ms 線性小縫收掉
                T_ = SR / (440.0 * 2 ** ((m - 69) / 12.0))
                L_ = int(round(round((LOOP_B - LOOP_A) / T_) * T_))
                xfs = int(0.005 * SR)
                lin = np.linspace(0, 1, xfs, dtype="float32")
                out_ = seg[:L_].copy()
                out_[:xfs] = seg[:xfs] * lin + seg[L_:L_ + xfs] * (1 - lin)
                bank[m] = out_
                del au
                if (m - lo) % 8 == 7:        # 進度列印＝餵 shell 的切換看門狗
                    print(f"[bank] {nm} rendering {m - lo + 1}/{hi - lo + 1}",
                          flush=True)
        _savez_atomic(path, bank)
        BANKS[nm] = bank
        print(f"[bank] {nm} rendered {len(bank)} notes "
              f"({time.time() - t0:.0f}s) → {path}", flush=True)
        if a.vowels:
            # v16：同一張嘴、同一套渲法，只換單元來源＝五個明暗層音庫
            LBANKS[nm] = []
            for li, (tag, src, lt0, lt1, _tl) in enumerate(VOWEL_LAYERS):
                lpath = f"{BANK_DIR}/{nm}_L{li}.npz"
                if os.path.exists(lpath) and not a.rebuild:
                    lb_ = _load_bank(lpath, lo, hi, f"{nm} L{li} {tag}")
                    if lb_ is not None:
                        LBANKS[nm].append(lb_)
                        print(f"[layer] {nm} L{li} {tag} cache hit", flush=True)
                        continue
                mv, _sr = sf.read(src, dtype="float64", always_2d=True)
                mv = mv[:, 0]
                mid = int((lt0 + lt1) / 2 * SR / HOP)   # 窗中點（hop 座標）
                with torch.no_grad():
                    UL = svc.encode(mv[max(0, (mid - 40) * HOP):
                                       (mid + 40) * HOP])[:, 8:-8]
                tl0 = time.time()
                lb = _render_layer(svc, UL, lo, hi)
                _savez_atomic(lpath, lb)
                LBANKS[nm].append(lb)
                print(f"[layer] {nm} L{li} {tag} rendered {len(lb)} notes"
                      f" ({time.time() - tl0:.0f}s) → {lpath}", flush=True)
            _norm_layers(nm)
        del svc


def _norm_mem_banks():
    """v39（Harry 08-15 live：「女聲好像蓋過男聲，至少我沒聽到男聲」）：
    把借來的團員音庫**逐音**對齊該聲部領唱音庫的 rms。

    為什麼需要：下面的 MIXG 聲部平衡是拿**領唱**音庫量的，借來的音庫沒進
    那個計算。實測（bank_853950ed4f 平均 rms）sop 借的三位比領唱大
    4.3/4.8/4.9dB、tenor 借的兩位小 1.9/2.0dB ⇒ sop 整部約 +3.9dB、tenor
    約 -1dB ＝中間開了 5dB 的縫。這是 --balance 被判死那條血訓的同一個坑
    （量錯音庫），差別只在這次量漏了新加的那半。
    ⚠ 逐音不是整體：不同歌手在音域兩端的響度曲線不一樣，整體對齊會讓某位
      在高音處消失（同 _norm_layers 選逐音的理由）。
    只動記憶體不動快取檔（同 _norm_layers）＝領唱與快取一個 byte 不動。
    """
    for (nm, mdl, sp), bk in XBANKS.items():
        ref_bank = BANKS.get(nm)
        if not ref_bank:
            continue
        ds = []
        for m_, b_ in bk.items():
            ref = ref_bank.get(m_)
            if ref is None:
                continue
            r0 = float(np.sqrt((ref ** 2).mean()))
            r_ = float(np.sqrt((b_ ** 2).mean()))
            if r0 > 1e-9 and r_ > 1e-9:
                bk[m_] = (b_ * np.float32(r0 / r_)).astype("float32")
                ds.append(20.0 * np.log10(r0 / r_))
        if ds:
            # 印出來＝這一步有沒有真的做到看得見（裸奔是無聲的失敗）
            print(f"[mem-norm] {nm} {mdl.split('/')[0]} spk{sp}: "
                  f"{np.mean(ds):+.1f}dB (逐音 {np.min(ds):+.1f}~"
                  f"{np.max(ds):+.1f})", flush=True)


_build_banks()
if a.per_part > 1 and a.mem_real and not a.vowels:
    # --mem-real 0＝誰都不借（凍結配置）＝連渲/載都跳過（08-17 review #3：
    # 原本照樣實例化 fem8+male8、渲/載 122 個音、常駐 ~28MB，而輸出端
    # _avail=[] 讓它們永遠播不出來；展場機器缺 fem8 權重時甚至開機即死）。
    # --vowels 開著時不借別的歌手：層庫（LBANKS）只有領唱那個 spk 的，
    # 團員會拿到別人的嘴配自己的層＝混層對不上。要並存得先渲團員的層庫，
    # 那是另一件事。母音線 08-13 已由 Harry 判「回到全啊」，不擋這次。
    _build_mem_banks()
    _norm_mem_banks()
# v32 聲部平衡：用**實測**音庫 rms 拉齊（+ --trim 的耳朵層）
MIXG = {nm: 1.0 for nm, *_x in VOICES}
if a.balance or a.trim:
    _rms = {nm: float(np.mean([np.sqrt((b ** 2).mean())
                               for b in BANKS[nm].values()]))
            for nm, *_x in VOICES}
    _ref = _rms[VOICES[0][0]]
    if a.balance:
        MIXG = {nm: _ref / max(v, 1e-9) for nm, v in _rms.items()}
    for _t in a.trim.split(","):
        if "=" in _t:
            _k, _v = _t.split("=")
            _k = _k.strip()
            assert _k in MIXG, f"--trim 不認得的聲部 {_k}"
            MIXG[_k] *= 10 ** (float(_v) / 20.0)
    print("[mix] " + "  ".join(
        f"{nm} {20 * np.log10(MIXG[nm]):+.1f}dB" for nm, *_x in VOICES),
          flush=True)
print(f"sample bank ready ({sum(len(b) for b in BANKS.values())} notes)",
          flush=True)


class KeyTracker:
    MIN = 120

    def __init__(self):
        self.h = np.zeros(12)

    def push(self, midi_f):
        self.h[int(round(midi_f)) % 12] += 1.0

    def root(self):
        if self.h.sum() < self.MIN:
            return 0
        cov = [sum(self.h[(r + d) % 12] for d in MAJ) for r in range(12)]
        return int(np.argmax(cov))


def dia_step(m, root, deg):
    """大調 root 上、m 的上方全音階 deg 度＝半音位移（deg 2=三度、4=五度）。"""
    rel = (m - root) % 12
    idx = int(np.argmin([min(abs(rel - s), 12 - abs(rel - s)) for s in MAJ]))
    return MAJ[(idx + deg) % 7] + 12 * ((idx + deg) // 7) - MAJ[idx]


def dia_third(m, root):
    return dia_step(m, root, 2)


def f0_autocorr(x):
    """~46ms 窗的正規化自相關 f0（65–800Hz）；無聲回 0。零 ML、位準無感。"""
    x = x - x.mean()
    e = float(np.sqrt(np.mean(x * x)))
    if e < 1e-5:
        return 0.0
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    if ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]
    lo, hi = int(SR / 800), min(int(SR / 65), n - 1)
    k0 = lo + int(np.argmax(ac[lo:hi]))
    if ac[k0] < 0.45:                     # 有聲判定只看原始峰（R2A#4：
        return 0.0                        # 防護不得把弱聲降格成無聲）
    k = k0
    # review A14：次諧波防護——k/2 的峰若幾乎一樣高＝真基頻在高八度
    while k >= 2 * lo and ac[int(round(k / 2))] > 0.85 * ac[k]:
        k = int(round(k / 2))
    if ac[k] < 0.45:
        k = k0                            # 減半候選太弱＝退回原峰
    # 拋物線內插精化
    if 0 < k < n - 1:
        d = (ac[k - 1] - ac[k + 1]) / (2 * (ac[k - 1] - 2 * ac[k]
                                            + ac[k + 1]) + 1e-12)
        k = k + float(np.clip(d, -1, 1))
    return SR / k


hopN = max(1, int(a.hop * SR))
# v17：嘴形二維錨點（lip_sweep2 全音域掃描校準；anchors 第 i 列對應
# VOWEL_LAYERS 第 i 層＝v16.3 建立的對應，不動）。缺檔＝權重釘 mid 一熱
# 並警告，不猜。（v16.2 聲音指紋 vowel_calib.npz 備用不再載——死於天使
# 漏音污染，屍檢 Daily 08-13；嘴看不到喇叭。）
if a.vowels:
    try:
        _zl = np.load("scratchpad/vowel_lip_calib.npz")
        _VL_A, _VL_S = _zl["anchors"], _zl["scales"]
        # 審查 S1/A5：錨點數不等於層數＝權重形狀炸 worker（全靜音假活著）；
        # scale ≤0 或非有限＝NaN 三重放大。校準檔不合格＝停用不猜。
        if (_VL_A.shape != (len(VOWEL_NAMES), 2)
                or not np.isfinite(_VL_A).all()
                or not np.isfinite(_VL_S).all() or (_VL_S <= 0).any()):
            print(f"⚠ [field] calibration file invalid (anchors {_VL_A.shape}, "
                  f"scales {_VL_S}) = vowel field disabled (fixed mid)",
                  flush=True)
            _VL_A = None
        else:
            print(f"[field] mouth-shape anchors loaded ({len(_VL_A)} vowels)", flush=True)
    except FileNotFoundError:
        _VL_A = None
        print("⚠ [field] no vowel_lip_calib.npz = vowel field disabled (fixed mid)",
              flush=True)
    # v17.1 整圈嘴錨點（scratchpad/lip_ring_calib.py 產；Harry 08-13
    # 「為什麼不測整圈嘴」）：有此檔＝優先走整圈嘴、2D 留退路。驗證同
    # S1/A5 標準（形狀/有限性/scale>0，不合格＝退 2D 不猜）。
    _VR_A = None
    try:
        _zr = np.load("scratchpad/vowel_ring_calib.npz")
        _VR_A, _VR_S = _zr["anchors"], _zr["scales"]
        if (_VR_A.ndim != 2 or _VR_A.shape[0] != len(VOWEL_NAMES)
                or _VR_S.shape != (_VR_A.shape[1],)
                or not np.isfinite(_VR_A).all()
                or not np.isfinite(_VR_S).all() or (_VR_S <= 0).any()):
            print(f"⚠ [field] full-ring mouth calibration invalid "
                  f"(anchors {_VR_A.shape}) = falling back to 2D", flush=True)
            _VR_A = None
        else:
            print(f"[field] full-ring mouth anchors loaded ({_VR_A.shape[0]} vowels x"
                  f" {_VR_A.shape[1]} dims)", flush=True)
    except FileNotFoundError:
        pass                                 # 沒整圈校準＝用 2D

# ⚠ 與 scratchpad/lip_ring_calib.py 的 LIP_RING/ring_vec 是**刻意複製**，
# 改一處要改兩處（_render_layer 同款紀律：校準與 live 特徵必須逐位元同）
LIP_RING = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405,
            314, 17, 84, 181, 91, 146,           # 外圈 20
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402,
            317, 14, 87, 178, 88, 95]            # 內圈 20


def _ring_vec(LA, aspect):
    """LA=(478,2) 正規化座標 → 80 維整圈嘴形：去平移（唇心）/去旋轉
    （太陽穴連線）/去尺度（太陽穴距離）。x 乘長寬比＝幾何等距。"""
    pts = LA[LIP_RING].copy()
    pts[:, 0] *= aspect
    e0 = np.array([LA[234, 0] * aspect, LA[234, 1]])
    e1 = np.array([LA[454, 0] * aspect, LA[454, 1]])
    ax = e1 - e0
    sc = float(np.hypot(ax[0], ax[1])) + 1e-9
    th = float(np.arctan2(ax[1], ax[0]))
    c, s = np.cos(-th), np.sin(-th)
    R = np.array([[c, -s], [s, c]])
    return ((pts - pts.mean(0)) @ R.T / sc).ravel()


# v18 個人母音模型（scratchpad/vowel_ml_train.py 產，logistic 405 參數）：
# 有此檔＝機率直接當混層權重，優先於錨點幾何。實驗判準（vowel_ml_probe，
# 同資料同時間分塊 CV）：最近鄰 acc 0.81/**真類機率 0.57**、本模型
# 0.89/**0.86**——live 那團「糊」量出來就是那個 0.57。驗證同 S1/A5。
_VM_W = None
if a.vowels:
    try:
        _zm = np.load("scratchpad/vowel_ring_model.npz")
        _VM_W, _VM_B = _zm["W"], _zm["b"]
        _VM_M, _VM_S = _zm["mean"], _zm["scale"]
        _D = len(LIP_RING) * 2
        if (_VM_W.shape != (len(VOWEL_NAMES), _D)
                or _VM_B.shape != (len(VOWEL_NAMES),)
                or _VM_M.shape != (_D,) or _VM_S.shape != (_D,)
                or not all(np.isfinite(x).all()
                           for x in (_VM_W, _VM_B, _VM_M, _VM_S))
                or (_VM_S <= 0).any()):
            print(f"⚠ [field] vowel model invalid (W {_VM_W.shape}) = back to anchors",
                  flush=True)
            _VM_W = None
        else:
            print(f"[field] vowel model loaded ({_VM_W.size + _VM_B.size} params, "
                  f"CV acc {float(_zm['cv_acc']):.2f})", flush=True)
    except FileNotFoundError:
        pass                                 # 沒模型＝用錨點幾何

# v19 融合母音模型（scratchpad/vowel_fusion_train.py 產）：整圈嘴 80D
# ＋MFCC 13D。Harry v18 live 判「欸分不出來、喔嗚也沒試出來、咿啊順」＝
# 唇物理盲點（欸/咿 差在舌位、喔/嗚 差在前突＝正面鏡頭無深度）；聲學
# 恰補這兩處（融合探針：欸/咿 0.79→0.94、總 acc 0.89→0.95、權重 0.94）。
# 天使 1:1 漏音實測只吃 0-1pp ⇒ **不做頻譜扣除**（省一層對齊管線）。
# ⚠ 下面的 MFCC 是 scratchpad/vowel_ml_probe.py::mfcc 的**刻意複製**——
# 參數存在模型檔裡逐項核對，不符即拒載，免得兩份定義悄悄分家。
_SG_W = None
_SG_BASE = None          # v40 開場基線（校正完才不是 None）
_SG_CAL = []             # v40 校正期間累積的特徵
_SG_JAWQ = []            # v40 jawOpen 的滑動窗（算運動量）
_SG_MTH = 0.0            # v40 運動否決門檻（0＝不否決＝舊行為）
_SG_JC = 2
_SG_MW = 15
_SG_NEEDCAL = False
if a.sing_gate:
    try:
        # v40：auto **刻意仍指向舊模型**＝這個 commit 不改變任何現行行為。
        # model3（三場次＋自校＋運動否決）要明寫路徑才會生效——它 08-15 live
        # 實測仍會連續誤觸（sp 飽和到 1.00，閉嘴與手遮嘴都中），沒有資格
        # 當預設。要試：--gate-model scratchpad/sing_gate_model3.npz
        _sgp = (a.gate_model if a.gate_model != "auto"
                else "scratchpad/sing_gate_model.npz")
        _zs = np.load(_sgp)
        _SG_W, _SG_B = _zs["W"], float(_zs["b"])
        _SG_M, _SG_S, _SG_N = _zs["mean"], _zs["scale"], list(_zs["names"])
        # v34.1：模型只吃**嘴/下巴**維度（idx）——全 52 維那版把眉毛/臉頰
        # 也學進去（佔總權重 49%），同場次 acc 98.8% 但**跨場次只有 72.2%、
        # 沒唱誤觸 47.5%**＝Harry 閉著嘴看到 sing 1.00 的原因。只留嘴/下巴
        # 後跨場次 acc 93.4%、誤觸 0.0%。
        _SG_I = _zs["idx"]
        if (_SG_W.shape != _SG_M.shape or _SG_S.shape != _SG_M.shape
                or len(_SG_N) != _SG_W.size or _SG_I.shape != _SG_W.shape
                or not np.isfinite(_SG_W).all() or (_SG_S <= 0).any()):
            print("⚠ [gate] 唱歌模型不合格＝退回幾何門控", flush=True)
            _SG_W = None
        else:
            _SG_CHK = [str(x) for x in _SG_N]   # 首幀對照（見下）
            # v40：新契約的四個欄位。舊模型沒有＝全部退回舊行為（校正關、
            # 否決關）＝**同一支碼同時支援兩顆模型**，不必分支兩條路。
            _SG_NEEDCAL = bool(int(_zs["calib"])) if "calib" in _zs else False
            _SG_JC = int(_zs["jaw_col"]) if "jaw_col" in _zs else 2
            _SG_MW = int(_zs["motion_win"]) if "motion_win" in _zs else 15
            _SG_MTH = (float(_zs["motion_th"]) if "motion_th" in _zs else 0.0)
            if a.gate_motion_th >= 0:            # 旗標覆蓋（現場可調）
                _SG_MTH = a.gate_motion_th
            # ⚠ cv 的語意：舊模型存的是**同場次**時間分塊 CV，卻被這行印成
            #   cross-session（08-15 查出的誤標）。新模型存的是**留一場**
            #   平均。所以標籤跟著模型走，不要再一律印 cross-session。
            _cvlab = "leave-one-session-out" if _SG_NEEDCAL else "same-session"
            print(f"[gate] sing model loaded ({_SG_W.size + 1} params, "
                  f"mouth/jaw only, {_cvlab} acc {float(_zs['cv']):.3f}) "
                  f"← {os.path.basename(_sgp)}", flush=True)
            if _SG_NEEDCAL:
                print(f"[gate] v40：開場 {a.gate_calib:.0f}s 基線自校（**閉嘴**）"
                      f"＋運動否決 {_SG_MTH:.4f}", flush=True)
    except FileNotFoundError:
        print("⚠ [gate] 無 sing_gate_model.npz＝門控退回 blendshape 幾何門檻",
              flush=True)
    except Exception as _e:                      # 審查：npz 損毀/缺鍵原本
        _SG_W = None                             #   直接 traceback 開不起來
        print(f"⚠ [gate] 唱歌模型載入失敗（{_e}）＝退回 blendshape 幾何門檻",
              flush=True)

_VF_W = None
if a.vowels:
    try:
        _zf = np.load("scratchpad/vowel_fusion_model.npz")
        _VF_W, _VF_B = _zf["W"], _zf["b"]
        _VF_M, _VF_S = _zf["mean"], _zf["scale"]
        _FN, _FH = int(_zf["nfft"]), int(_zf["hopf"])
        _NM, _NC = int(_zf["nmel"]), int(_zf["ncep"])
        _F0, _F1 = float(_zf["fmin"]), float(_zf["fmax"])
        _DIM = len(LIP_RING) * 2 + _NC
        if (str(_zf["feat"]) != "ring+mfcc"
                or _VF_W.shape != (len(VOWEL_NAMES), _DIM)
                or _VF_B.shape != (len(VOWEL_NAMES),)
                or _VF_M.shape != (_DIM,) or _VF_S.shape != (_DIM,)
                or not all(np.isfinite(x).all()
                           for x in (_VF_W, _VF_B, _VF_M, _VF_S))
                or (_VF_S <= 0).any()):
            print(f"⚠ [field] fusion model invalid (W {_VF_W.shape}) = back to vision",
                  flush=True)
            _VF_W = None
        else:
            def _melfb(nmel, fmin, fmax, nfft):
                def m(f):
                    return 2595 * np.log10(1 + f / 700)

                def im(x):
                    return 700 * (10 ** (x / 2595) - 1)
                pts = im(np.linspace(m(fmin), m(fmax), nmel + 2))
                bb_ = np.floor((nfft + 1) * pts / SR).astype(int)
                fb = np.zeros((nmel, nfft // 2 + 1))
                for i in range(nmel):
                    lo_, mid_, hi_ = bb_[i], bb_[i + 1], bb_[i + 2]
                    if mid_ > lo_:
                        fb[i, lo_:mid_] = np.linspace(0, 1, mid_ - lo_)
                    if hi_ > mid_:
                        fb[i, mid_:hi_] = np.linspace(1, 0, hi_ - mid_)
                return fb

            _FFB = _melfb(_NM, _F0, _F1, _FN)
            _FWIN = np.hanning(_FN)
            _FDCT = np.cos(np.pi / _NM * (np.arange(_NM) + 0.5)[None, :]
                           * np.arange(1, _NC + 1)[:, None])
            _ABUF = [np.zeros(_FN, dtype="float32")]   # 最近 NFFT 樣本
            print(f"[field] fusion model loaded ({_VF_W.size + _VF_B.size} params, "
                  f"CV acc {float(_zf['cv_acc']):.2f}, vision+acoustic)",
                  flush=True)
    except FileNotFoundError:
        pass                                 # 沒融合模型＝用視覺模型
kt = KeyTracker()
st = {"die": False, "xrun": 0, "cents": 0.0, "note": None, "cand": None,
      "cc": 0, "quiet": 0, "lastm": None, "pada": None, "padt": -1e9,
      "hist": [], "tsw": -1e9, "pn": None, "porta_ok": True}

# ── 嘴部門控（score_live v3.1 同款）：嘴閉＝音高不算你的＝回授殺 ──
M = {"ok": a.mouth == 0 or bool(a.file), "on": False, "last_open": 0.0,
     "val": -1.0, "t_on": -1e9, "face": False}


def _mouth_worker():
    # 這兩個在函式裡會被指派（模型不合格時停用）＝必須宣告 global，
    # 否則整個函式把它們當區域變數 ⇒ 第一幀 UnboundLocalError ⇒ 執行緒
    # 死在寬鬆 except 裡 ⇒ **沒畫面且整場門控全開**（08-13 實案）。
    # ⚠ _SG_BASE 在下面是**賦值**（不是 mutate）＝沒有這行就是
    # UnboundLocalError＝鏡頭執行緒整場死掉、門控全開（v35 被咬過一次）
    global _SG_W, _SG_CHK, _SG_BASE
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path="scratchpad/face_landmarker.task"),
            running_mode=vision.RunningMode.VIDEO, num_faces=1,
            output_face_blendshapes=bool(a.bs_gate or a.sing_gate))
        lmk = vision.FaceLandmarker.create_from_options(opts)
        if a.cam >= 0:
            cap = cv2.VideoCapture(a.cam)
            assert cap.isOpened(), f"鏡頭 {a.cam} 打不開"
        else:
            best = None
            for ci in range(3):
                c = cv2.VideoCapture(ci)
                if not c.isOpened():
                    continue
                time.sleep(0.4)
                okf, fr = c.read()
                b = float(fr.mean()) if okf else -1
                c.release()
                if best is None or b > best[1]:
                    best = (ci, b)
            assert best and best[1] > 10, f"找不到有畫面的鏡頭 {best}"
            print(f"[mouth] camera index {best[0]} (brightness {best[1]:.0f})",
                  flush=True)
            cap = cv2.VideoCapture(best[0])
        t0 = time.time()
        last_face = time.time()
        last_det = 0.0
        while not st["die"]:
            okf, frame = cap.read()
            now = time.time()
            M["face"] = now - last_face < 0.3     # 臉通道（dump/畫面用）
            if okf:
                M["hb"] = now    # R4A#1：心跳＝「真的讀到 frame」。read()
                #   失敗（鏡頭被搶/掉線常態）＝心跳停＝看門狗開火裸奔＋
                #   警告，而不是永久關門的無聲死亡
            if okf and now - last_det >= 0.033:      # ~30Hz（原 20Hz）
                last_det = now
                img = mp.Image(image_format=mp.ImageFormat.SRGB,
                               data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                res = lmk.detect_for_video(img, int((now - t0) * 1000))
                if (a.view or a.frame_b64 > 0) and not res.face_landmarks:
                    # R4A#8：臉丟也更新畫面（原本凍在最後一張 FACE Y）
                    h2 = int(frame.shape[0] * 480 / frame.shape[1])
                    fr2 = cv2.resize(frame, (480, h2))
                    cv2.putText(fr2, "NO FACE - gate closing", (10, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    M["frame"] = fr2
                if not res.face_landmarks:
                    # 審查：臉丟時 M["on"]/M["arm"] 原本停在舊值，臉一回來
                    # 走的是「維持」分支（門檻低、不吃防抖）＝轉頭再回正、
                    # 閉著嘴也會開門。丟臉＝重新來過。
                    M["on"] = False
                    M["arm"] = 0
                if res.face_landmarks:
                    last_face = now
                    f = res.face_landmarks[0]
                    val = abs(f[13].y - f[14].y) / (abs(f[10].y - f[152].y)
                                                    + 1e-9)
                    M["val"] = val
                    wd_ = (abs(f[61].x - f[291].x)
                           / (abs(f[234].x - f[454].x) + 1e-9))
                    M["wd"] = wd_
                    bs = (res.face_blendshapes[0]
                          if ((a.bs_gate or a.sing_gate)
                              and res.face_blendshapes) else None)
                    if bs is not None and _SG_W is not None:
                        # v34：訓練出來的「在不在唱」機率當門控（52 維
                        # blendshape → logistic）。負樣本含**講話**——手刻
                        # 判準從來擋不掉的那個。名稱順序啟動時已驗。
                        if _SG_CHK is not None:
                            # 審查：註解宣稱「名稱順序啟動時已驗」但實際
                            # 沒驗——mediapipe 換版就會靜默錯位。首幀真的
                            # 對一次，不符就停用模型（不猜）。
                            _live = [bs[i].category_name for i in _SG_I]
                            if _live != _SG_CHK:
                                print("⚠ [gate] blendshape 名稱順序與模型不符"
                                      "＝停用唱歌模型", flush=True)
                                _SG_W = None
                            _SG_CHK = None
                        if _SG_W is None:
                            bs = None
                    if bs is not None and _SG_W is not None:
                        _v = np.array([c.score for c in bs])[_SG_I]
                        # v40①：開場基線自校。08-15 量到同一個人同一個動作
                        # jawOpen 靜止值 08-13=0.031 / 08-15=0.017＝逐幀
                        # logistic 吃絕對值就會整批翻盤（閉嘴誤觸 44%）。
                        # 校正期間**門一律關**（他被要求閉嘴），校完才放行。
                        # ⚠ 一定要先給 _p 一個值：校正**完成的那一幀**會在
                        # 這個分支裡把 _SG_BASE 填上，下面的判斷式就不再是
                        # 「校正中」＝掉進 else 去讀 _p＝UnboundLocalError＝
                        # 鏡頭執行緒整場死掉、門控全開（今天真的炸過一次）
                        _p = 0.0
                        if _SG_NEEDCAL and _SG_BASE is None:
                            _SG_CAL.append(_v)
                            # 用**時間**不是幀數：幀率隨機器/光線變動，用幀數
                            # 會讓實際校正時間跟印出來的秒數不符
                            if now - M.setdefault("calt0", now) >= a.gate_calib \
                                    and len(_SG_CAL) >= 20:
                                _SG_BASE = np.mean(_SG_CAL, axis=0)
                                print(f"[gate] 基線校正完成（{len(_SG_CAL)} 幀 / "
                                      f"{now - M['calt0']:.1f}s）＝門控啟用",
                                      flush=True)
                            M["sp"] = 0.0
                            M["on"] = False
                            M["arm"] = 0
                            nv = False
                        else:
                            if _SG_BASE is not None:
                                _v = _v - _SG_BASE
                            # v40②：運動否決。講話＝3-8Hz 音節開闔，唱歌＝
                            # 姿勢維持。實測 |ΔjawOpen| 0.5s 窗均值：講話
                            # 0.0173/0.0130/0.0079（三場），唱歌最高的「啊」
                            # 只有 0.0023。差分量不吃基線＝跨場次站得住。
                            _SG_JAWQ.append(float(bs[_SG_I[_SG_JC]].score))
                            del _SG_JAWQ[:-(_SG_MW + 1)]
                            _mo = (float(np.abs(np.diff(_SG_JAWQ)).mean())
                                   if len(_SG_JAWQ) > 1 else 0.0)
                            M["mo"] = _mo
                            _veto = _SG_MTH > 0 and _mo > _SG_MTH
                            _p = 1.0 / (1.0 + np.exp(
                                -(np.dot((_v - _SG_M) / _SG_S, _SG_W) + _SG_B)))
                            M["sp"] = float(_p)
                            if _veto:            # 你在講話＝一律不算你
                                _p = 0.0
                        if _SG_NEEDCAL and _SG_BASE is None:
                            pass                 # 校正中，nv 已設 False
                        elif M["on"]:
                            nv = _p > a.sing_hold
                            if not nv:
                                M["arm"] = 0
                        else:
                            M["arm"] = (M.get("arm", 0) + 1
                                        if _p > a.sing_open else 0)
                            nv = M["arm"] >= a.bs_frames
                    elif bs is not None:
                        # v30：blendshape 門控——jawOpen 管張口、mouthPucker
                        # 管嘟嘴。兩者都是訓練出來的量、對頭部角度與距離
                        # 已正規化，不像手算比值那樣被鏡頭位置影響。
                        _b = {c.category_name: c.score for c in bs}
                        jaw = _b.get("jawOpen", 0.0)
                        puc = _b.get("mouthPucker", 0.0)
                        M["jaw"], M["puc"] = jaw, puc
                        raw = (jaw > (a.bs_close if M["on"] else a.bs_open)
                               or puc > a.bs_pucker)
                        if M["on"]:
                            nv = raw
                            if not raw:
                                M["arm"] = 0
                        else:
                            # 開門要連續 N 幀（防單幀雜訊開門 0.8s）
                            M["arm"] = (M.get("arm", 0) + 1) if raw else 0
                            nv = M["arm"] >= a.bs_frames
                    elif M["on"]:
                        # v28 幾何退路：在唱之中——張口夠大 或 嘴夠窄
                        nv = (val > a.mouth_close
                              or (a.mouth_narrow > 0 and wd_ < a.mouth_narrow
                                  and val > 0.004))
                    else:
                        nv = val > a.mouth_open      # 起唱一律看張口
                    if nv != M["on"] and a.gate_log:
                        print(f"[gate] {'open' if nv else 'close'}"
                              f"  p {M.get('sp', -1):.3f}"
                              f"  open {val:.3f}"
                              f"  t {time.time() - t0:.1f}s",
                              flush=True)
                    if nv and not M["on"]:
                        M["t_on"] = now  # v7 呼吸預備：開口瞬間＝預告
                    M["on"] = nv
                    if M["on"]:
                        M["last_open"] = now
                    # v19：有融合模型＝權重由音訊執行緒逐 hop 算（要當下
                    # 的 MFCC），這裡只維護嘴形 EMA 供它取用
                    if a.vowels and _VF_W is not None:
                        _rv = _ring_vec(
                            np.array([[p.x, p.y] for p in f]),
                            frame.shape[1] / frame.shape[0])
                        M["rv"] = (_rv if M.get("rv") is None else
                                   M["rv"] + 0.35 * (_rv - M["rv"]))
                    elif a.vowels and (_VM_W is not None
                                       or _VR_A is not None):
                        # v17.1 整圈嘴（Harry 08-13）：內外唇 40 點 → 80 維
                        # ——母音靠圓唇度/嘴角形狀/唇曲率分，不只高寬
                        # （欸↔咿在 2D 重疊 0.37 的解）。
                        _rv = _ring_vec(
                            np.array([[p.x, p.y] for p in f]),
                            frame.shape[1] / frame.shape[0])
                        M["rv"] = (_rv if M.get("rv") is None else
                                   M["rv"] + 0.35 * (_rv - M["rv"]))
                        if _VM_W is not None:
                            # v18：個人模型的類別機率＝混層權重（五行純
                            # numpy，~30μs；render 迴圈仍零 ML）
                            _l = _VM_W @ ((M["rv"] - _VM_M) / _VM_S) + _VM_B
                            _e = np.exp(_l - _l.max())
                            M["vw"] = _sub(_e / _e.sum())
                        else:
                            _dl = np.abs((_VR_A - M["rv"]) / _VR_S).mean(1)
                            _w = 1.0 / (_dl * _dl + 1e-6)
                            M["vw"] = _sub(_w / _w.sum())
                    elif a.vowels and _VL_A is not None:
                        # v17 連續母音場（2D 退路）：嘴形 (高,寬) 對五錨點
                        # 正規化 L1 距離 → 平方反比權重（在錨點上＝近一熱、
                        # 錨點間＝連續混合）。無分類、無遲滯、無 dwell——
                        # 離散選層範式 08-13 判死（鏡子永遠遲到半顆母音）。
                        _wd = (abs(f[61].x - f[291].x)
                               / (abs(f[234].x - f[454].x) + 1e-9))
                        _hw = np.array([val, _wd])
                        M["hw"] = (_hw if M.get("hw") is None else
                                   M["hw"] + 0.35 * (_hw - M["hw"]))
                        _dl = np.abs((_VL_A - M["hw"]) / _VL_S).sum(1)
                        _w = 1.0 / (_dl * _dl + 1e-6)
                        M["vw"] = _sub(_w / _w.sum())
                    if a.view or a.frame_b64 > 0:
                        # v14 監看：內唇線＋數值（主執行緒 imshow，這裡只畫）
                        h2 = int(frame.shape[0] * 480 / frame.shape[1])
                        fr2 = cv2.resize(frame, (480, h2))
                        # v29（Harry：「中間牙齒一條線很奇怪，不能十字嗎、
                        # 不能嘴唇圈嗎」）：畫**整圈嘴唇＋十字**——縱線＝
                        # 開口高度、橫線＝嘴寬，兩者都是門控真的在用的量
                        # （v28 起嘟嘴保持看的就是嘴寬）。單畫一條縱線是
                        # v14 只看開闔時代的遺留。
                        def _pt(i):
                            return (int(f[i].x * 480), int(f[i].y * h2))
                        col = (0, 255, 0) if M["on"] else (0, 0, 255)
                        for _seg in (LIP_RING[:20], LIP_RING[20:]):
                            cv2.polylines(fr2,
                                          [np.array([_pt(i) for i in _seg],
                                                    np.int32)],
                                          True, col, 1, cv2.LINE_AA)
                        cv2.line(fr2, _pt(13), _pt(14), col, 2)    # 開口
                        cv2.line(fr2, _pt(61), _pt(291), col, 2)   # 嘴寬
                        _wdv = M.get("wd", -1.0)
                        cv2.putText(fr2,
                                    f"FACE {'Y' if M.get('face') else 'N'} "
                                    + (f"sing {M['sp']:.2f} "
                                       if "sp" in M else
                                       f"jaw {M.get('jaw', 0):.2f} puck "
                                       f"{M['puc']:.2f} "
                                       if "jaw" in M else
                                       f"open {val:.3f} wide {_wdv:.3f} ") +
                                    f"{'ON' if M['on'] else 'off'}"
                                    f"  gate "
                                    f"{'OK' if M.get('eff', M['ok']) else 'X'}",
                                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55, (255, 255, 255), 2)
                        if a.vowels and M.get("vw") is not None:
                            # v17/v20：在用的層權重即時疊字（母音名＋%）
                            cv2.putText(fr2, " ".join(
                                f"{nm_}{int(round(w * 100)):02d}"
                                for nm_, w in zip(_SEL_NAMES, M["vw"])),
                                (10, 48), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 255), 1)
                        M["frame"] = fr2
            # review A7：M["ok"] 每圈都要重算（鏡頭一卡不凍死）。
            # v14.1（Harry 驗收裁決）：臉不在＝他不在＝門**關**——原「臉丟
            # 3s fail-open」讓空房間被回授/雜音觸發。鏡頭真死（心跳斷）
            # 另由看門狗裸奔＋警告，兩個情境分開。
            M["ok"] = now - M["last_open"] < 0.8
            if not okf:
                time.sleep(0.01)
        cap.release()
    except Exception as e:
        M["ok"] = True
        M["dead"] = str(e)          # 審查★5：原本只印一行就沉默裸奔整場
        import traceback
        traceback.print_exc()
        print(f"⚠ [mouth] GATE DEAD ({e}) = running ungated",
              flush=True)


if a.mouth and not a.file:
    threading.Thread(target=_mouth_worker, daemon=True).start()


class VoicePlay:
    """一個聲部的取樣播放：目前音、換音 crossfade、起釋音包絡。"""

    def __init__(self, nm, atk=None, rel=None, count=True, bank=None):
        self.nm = nm
        self.count = count               # v38：False＝團員，不進法醫計數
        self.atk = atk if atk is not None else a.attack * self.ATK_MULT.get(nm, 1.0)
        self.rel = rel if rel is not None else a.release
        self.bank = BANKS[nm] if bank is None else bank   # v38：團員可換歌手
        self.cur = None                  # (midi, buf, pos)
        self.old = None                  # 換音淡出中的上一個音
        self.env = 0.0
        self.tgt = 0.0
        self.gl = 0.0                    # v15 porta：起始半音偏移
        self.glt = 0                     # 滑音總長（樣本）
        self.glr = 0                     # 滑音剩餘（樣本）
        self.xfn = int(0.03 * SR)
        self.xin = 0                     # 換音淡入殘餘（review A2）
        self.vw = _VW_MID                # v17：上一 hop 的層權重（混層內插用）

    def set_note(self, m):
        if m is None:
            self.tgt = 0.0
            return
        self.tgt = 1.0
        # review A1：clip 要在比較之前——音域外的目標音否則每 hop 重觸發
        # 一次（bass >68 曾進 28.6Hz 重啟迴圈＝梳狀嗡鳴，最大爆音源）
        lo_b, hi_b = min(self.bank), max(self.bank)
        if a.fold:
            # 音域外**折八度**而不是 clip：clip 會把音高類別也改掉（實測
            # 19-26% 的 pad 音因此變成非和弦音，sop pad 在他唱低音時卡在
            # F3 不動、tenor 曾 83→68）。折返保留音級，只換八度。
            while m < lo_b:
                m += 12
            while m > hi_b:
                m -= 12
        m = int(np.clip(m, lo_b, hi_b))
        if self.cur is not None and self.cur[0] == m:
            return
        if self.cur is not None:
            self.old = [self.cur, self.xfn]        # [(midi,buf,pos), 剩餘]
            self.xin = self.xfn                    # review A2：新音要淡入
            self.xlin = False
            if (a.porta > 0 and abs(m - self.cur[0]) <= 2
                    and st.get("porta_ok", True)):
                # v15：前音停留夠久（≥0.3s）才滑＝表情滑音；快樂句乾淨切
                self.gl = float(self.cur[0] - m)
                self.glt = self.glr = max(1, int(a.porta * SR))
                if self.count:      # v38：只有領唱記數。團員也走這條的話
                    #   滑音計數會被人數放大（--per-part 4 實測 53→209），
                    #   而那個數字是滑音觸發率法醫的尺（v26 那條 46.6%→
                    #   68.9% 就是它量的）＝儀器會被悄悄污染。
                    st["porta_n"] = st.get("porta_n", 0) + 1   # v25 儀器
                self.xlin = True   # R4A#12：porta＝同基頻起步＝相干，
                #   等功率縫會凸（σ2.19dB 實測）→ 線性縫
        # v17：五層堆成 (5,L)（同 midi 各層 loop 等長＝樣本可對齊；長度
        # 只依 f0，見 _render_layer）。混層在 render 逐樣本做，這裡不選層
        # ＝「長音中不換層」的 v16 邊界隨範式一起消失。全聲部跟場（v16.3b
        # alto-only 隔離是死範式下的實驗，不留）。音域 clip 仍用 self.bank
        # （層庫同 lo/hi）。
        buf = (np.stack([LBANKS[self.nm][li][m] for li in _LSEL])
               if a.vowels else self.bank[m])
        self.cur = [m, buf, 0.0]

    ATK_MULT = {"bass": 0.6, "alto": 1.0, "sop": 1.6}   # v5 錯落起音：
    #   三聲部起音時值各異＝進場是「綻開」不是一堵牆（v6② stagger 簡式）

    def render(self, n, rate):
        out = np.zeros(n, dtype="float32")
        if a.vowels:
            # v17：本 hop 的逐樣本層權重＝上一 hop 權重 → 最新權重線性內插
            # （殺 30Hz 權重更新的 zipper 階躍）；cur/old 共用同一組＝換音
            # crossfade 兩端聽到同一張嘴。鏡頭沒給值＝釘 mid 一熱。
            w1 = M.get("vw")
            if w1 is None:
                w1 = _VW_MID
            g5 = np.linspace(self.vw, w1, n, dtype="float32")   # (n,5)
            self.vw = w1
        else:
            g5 = None
        if self.cur is not None and (self.env > 1e-4 or self.tgt > 0):
            if self.glr > 0:
                # v15 線性定長滑音：porta 秒後**精確抵達**（原指數尾＝
                # 永遠在路上＝正常唱速全變滑音的主因之一）
                # v25（審查 S1）：原本整塊套同一個 rate＝**滑音是樓梯不是
                # 斜坡**——porta 0.08 / hop 0.035 時第一塊 35ms 完全停在
                # 舊音高，之後三階、每階最大 ~90 cents。改成塊內逐樣本
                # 內插（_read 接受每樣本 rate 陣列）。
                r0 = self.glr / self.glt
                r1 = max(0, self.glr - n) / self.glt
                g = 2 ** (self.gl * np.linspace(r0, r1, n, endpoint=False,
                                                dtype="float64") / 12.0)
                self.glr = max(0, self.glr - n)
            else:
                g = 1.0
            w = self._read(self.cur, n, rate * g, g5)
            if self.xin > 0:
                # review A2：互補 crossfade——新音線性淡入、舊音線性淡出，
                # 換音瞬間總幅 ≈ 常數（原本新音全幅＋舊音全幅起跳＝+6dB 過衝）
                k = min(n, self.xin)
                w = w.copy()
                ramp = np.linspace(1 - self.xin / self.xfn,
                                   1 - (self.xin - k) / self.xfn,
                                   k, dtype="float32")
                w[:k] *= ramp if getattr(self, "xlin", False) \
                    else np.sqrt(ramp)
                self.xin -= k
            out += w

        if self.old is not None:
            w = self._read(self.old[0], n, rate, g5)
            k = min(n, self.old[1])
            framp = np.linspace(self.old[1] / self.xfn,
                                (self.old[1] - k) / self.xfn,
                                k).astype("float32")
            fade = framp if getattr(self, "xlin", False) else np.sqrt(framp)
            out[:k] += w[:k] * fade    # 等功率；porta 相干時線性
            self.old[1] -= k
            if self.old[1] <= 0:
                self.old = None
        # 起音/釋音包絡（塊內線性）
        aatk = n / (self.atk * SR)
        arel = n / (self.rel * SR)
        e0 = self.env
        e1 = (min(1.0, e0 + aatk) if self.tgt > 0 else max(0.0, e0 - arel))
        self.env = e1
        return out * np.linspace(e0, e1, n, dtype="float32")

    def _read(self, slot, n, rate, g5=None):
        """rate 可為純量或**每樣本陣列**（v25 滑音：塊內逐樣本變速）。"""
        _m, buf, pos = slot
        L = buf.shape[-1]
        if np.ndim(rate) == 0:
            idx = (pos + np.arange(n, dtype="float64") * rate) % L
            end = pos + n * rate
        else:
            # 位移＝速率的累積和（idx[0]=pos，之後每樣本各自的 rate）
            c = np.cumsum(rate, dtype="float64")
            idx = (pos + c - rate[0]) % L
            end = pos + c[-1]
        i0 = idx.astype(np.int64)
        fr = (idx - i0).astype("float32")
        i1 = (i0 + 1) % L
        slot[2] = float(end % L)
        if buf.ndim == 1:
            return buf[i0] * (1 - fr) + buf[i1] * fr
        # v17：五層同相位對齊讀出 (5,n) → 逐樣本權重混合（層等長＝同一個
        # 游標；同 seed 同 f0 渲＝跨層混音相干性待耳裁）
        y5 = buf[:, i0] * (1 - fr) + buf[:, i1] * fr
        return (y5 * g5.T).sum(axis=0)


players = [VoicePlay(nm) for nm, *_x in VOICES]
# v7 站位（成分四 lite）：快層 bass 中偏左/alto 右/sop 左；墊拉到外圈
# 站位：依聲部數展開（原本寫死三個）——tenor 插在 bass 右側
_PANF = {"bass": -0.15, "tenor": 0.15, "alto": 0.35, "sop": -0.35}
_PANP = {"bass": -0.6, "tenor": 0.3, "alto": 0.6, "sop": 0.15}
PAN_F = [_PANF[nm] for nm, *_x in VOICES]
PAN_P = [_PANP[nm] for nm, *_x in VOICES]


def _pan(p_):
    th = (p_ + 1) * np.pi / 4
    return np.float32(np.cos(th)), np.float32(np.sin(th))
# v6 慢層：同音庫、長起釋音（0.8s 綻開/1.2s 收）＝和弦墊，活得比你的句子久
_PADATK = {"bass": 0.8, "tenor": 0.85, "alto": 0.9, "sop": 1.0}
pads = ([VoicePlay(nm, atk=_PADATK[nm], rel=0.9)
         for nm, *_x in VOICES] if a.pad > 0 else [])
# v3 人味：每聲部兩個不共相位的慢 LFO 疊出 ±human cents 的獨立漂移
_rng = np.random.RandomState(20260811)
HUM = [( _rng.uniform(0.06, 0.16), _rng.uniform(0, 6.28),
         _rng.uniform(0.15, 0.3), _rng.uniform(0, 6.28))
       for _ in range(2 * len(VOICES))]
VIBPH = [_rng.uniform(0, 6.28) for _ in range(2 * len(VOICES))]
# v38 每聲部多人（--per-part）。**團員抽籤在 VIBPH 之後**＝上面那些既有
# 的 HUM/VIBPH 值不會因為開了這個旗標而位移（--per-part 1 要 byte-identical，
# 而且 4 人版的第一位聽起來要跟 1 人版是同一個人）。
PP = int(a.per_part)
MEMG = np.float32(1.0 / np.sqrt(PP))     # PP=1 → 恰好 1.0＝乘法是 no-op
MEMS = [[] for _ in VOICES]
if PP > 1:
    _mrng = np.random.RandomState(20260814)
    _real = 0
    for _vi, (_nm, _mdl, _sp, *_x) in enumerate(VOICES):
        # 團員優先吃**真的別的歌手**（同模型的其他 spk），用完才退回
        # 「複製領唱＋失諧」。08-14 Harry：「我比較希望是用不同音色」。
        # v39：名單那欄是 (模型, spk)，所以 key 是三元組
        # v43 --mem-real 0＝團員一律用領唱音庫的複製（v38 行為），不借
        # 別顆模型的歌手。Harry 08-16 已耳裁「真人音色與失諧複製分不出來」，
        # 所以拿掉它在音樂上零損失，但它是 roughness 的來源之一（不同錄音
        # 的同一個音疊加＝必然有微小音高差）。
        # v44（08-17 review #6）：**領唱自己的 (模型, spk) 不能借**——v42
        # tenor 領唱換成 male8 spk7 後，名單上的「別人」跟領唱成了同一人＝
        # 團員是領唱的逐位元複製（npz 實測 cmp 相同）＋失諧＝剛判死的干涉
        # 雜音以最壞形式回歸，而開機 banner 還把它報成 DIFFERENT singer。
        _avail = ([ms for ms in MEM_SPK.get(_nm, [])
                   if (_nm, *ms) in XBANKS and ms != (_mdl, _sp)]
                  if a.mem_real else [])
        for _k in range(PP - 1):
            _dly = int(_mrng.uniform(0, 2 * a.spread_ms * 1e-3) * SR)
            _spk = _avail[_k] if _k < len(_avail) else None
            if _spk is not None:
                _real += 1
            MEMS[_vi].append({
                # 借到別人的音庫＝真的另一個人；借不到就共用 BANKS[nm]
                # （零重渲）＋靠失諧/錯開假裝。count=False＝不污染法醫計數
                "p": VoicePlay(_nm, count=False,
                               bank=(XBANKS[(_nm, *_spk)] if _spk is not None
                                     else None)),
                "spk": _spk,
                # 靜態失諧：真團團員彼此就是差這麼多（F3）
                "det": 2.0 ** (float(_mrng.normal(0.0, a.spread_cents)) / 1200.0),
                "hum": (_mrng.uniform(0.06, 0.16), _mrng.uniform(0, 6.28),
                        _mrng.uniform(0.15, 0.3), _mrng.uniform(0, 6.28)),
                # v43 測試（08-16）：`--vib-sync 1` 讓團員與領唱**共用同一個
                # 顫音相位**。假設：同一個音的 N 個副本各自隨機相位時，瞬時
                # 頻率彼此錯開，在高諧波上落進 15-300Hz 的差頻區＝心理聲學的
                # roughness（粗糙感）——那是「聽得到但頻譜找不到」的東西，
                # 因為它不是新成分、是既有成分的干涉。同相位＝一起上下＝
                # 沒有相對頻差。⚠ 無論走哪條都先抽掉這個亂數，隨機序列才不會
                # 位移（否則 det/hum/pan 全跟著變，就不是單一變數）。
                "vib": (lambda _p: VIBPH[_vi] if a.vib_sync else _p)(
                    _mrng.uniform(0, 6.28)),
                "pan": float(np.clip(
                    PAN_F[_vi] + _mrng.uniform(-a.spread_pan, a.spread_pan),
                    -1.0, 1.0)),
                # 起音錯開＝延遲線。它同時把同一份 loop 讀在不同相位上，
                # 這是去相關的主力——只給失諧的話兩個一模一樣的 loop 只會
                # 產生拍頻，不會產生「兩個人」。
                "dly": _dly,
                "tail": np.zeros(_dly, dtype="float32"),
            })
    print(f"[choir] per-part {PP} → {len(VOICES) * PP} voices "
          f"({_real} of them a DIFFERENT singer, the rest detuned copies; "
          f"detune sd {a.spread_cents:.0f}c, onset 0-{2*a.spread_ms:.0f}ms, "
          f"pan ±{a.spread_pan:.2f}, gain 1/√{PP}={float(MEMG):.3f})",
          flush=True)


def _mdelay(mm, w):
    """團員的固定延遲線（起音錯開＋loop 相位去相關）。"""
    if mm["dly"] == 0:
        return w
    buf = np.concatenate((mm["tail"], w))
    mm["tail"] = buf[len(w):]
    return buf[:len(w)]
RVIR, RVT = None, None
if a.wet > 0:
    import reverb as rv
    from scipy.signal import fftconvolve as _fftc
    RVIR = rv.make_ir(2.4, trim_db=35.0)
    RVT = [np.zeros(len(RVIR) - 1)]
    print(f"[wet] IR {len(RVIR)/SR:.1f}s, send {a.wet}", flush=True)
dmp = ({"mic": [], "out": [], "mok": [], "f0": [], "face": [], "mon": [],
        "note": [], "vw": [], "val": [], "wd": [], "sp": [], "vn": []}
       if a.dump else None)     # v14.2：法醫通道全集（臉/嘴/門/偵測/音）
#                                v17 加 vw＝每 hop 的五層權重（vowels 開才存）
DMPMAX = max(1, int(a.dump_max_min * 60 / a.hop))   # v44 上限（hop 數）


def _write_dump():
    """dump 落檔（live 收場與 --file 共用）。**live 必須在關流之前呼叫**
    （stream context 內）：CoreAudio FinishStoppingStream 會掛死（playbook），
    掛死發生在 __exit__——dump 寫在那之後＝shell 的 8s SIGKILL 讓整份錄音
    陪葬（08-17 review #13；respond2 08-04 同一課，見其 finally 註解）。
    長度取 mic 快照＝worker 還在 append 也安全（各通道切齊）。"""
    if dmp is None or not dmp["mic"]:
        return
    k = len(dmp["mic"])
    sf.write(a.dump + "_mic.wav", np.concatenate(dmp["mic"][:k]), SR)
    sf.write(a.dump + "_out.wav", np.concatenate(dmp["out"][:k]), SR)
    np.savez(a.dump + "_st.npz", mok=np.array(dmp["mok"][:k]),
             f0=np.array(dmp["f0"][:k]), face=np.array(dmp["face"][:k]),
             mon=np.array(dmp["mon"][:k]), note=np.array(dmp["note"][:k]),
             val=np.array(dmp["val"][:k]), wd=np.array(dmp["wd"][:k]),
             sp=np.array(dmp["sp"][:k]), vn=np.array(dmp["vn"][:k]),
             hop=a.hop,              # R5#2：離線法醫流程要能跑
             **({"vw": np.array(dmp["vw"][:k])} if a.vowels else {}))
    print(f"\ndump: {a.dump}_mic/_out.wav + _st.npz", flush=True)
in_q, out_q = [], []
qlock = threading.Lock()


LM = None
if a.predict and os.path.exists(a.predict):
    import json as _json
    _lm = _json.load(open(a.predict))
    LM = {"tri": {tuple(map(int, k.split(","))): v
                  for k, v in _lm["tri"].items()},
          "bi": {int(k): v for k, v in _lm["bi"].items()},
          "uni": _lm["uni"]}
    print(f"[habit] melody_lm loaded ({len(LM['tri'])} trigrams)", flush=True)


def lm_top3(iv1, iv2):
    """最近兩個音程 → 模型 top-3 下一音程（backoff）。"""
    sc = {}
    t = LM["tri"].get((iv1, iv2))
    b = LM["bi"].get(iv2)
    for v in range(-12, 13):
        sc[v] = ((t.get(str(v), 0.0) * 100 if t else 0.0)
                 + (b.get(str(v), 0.0) * 10 if b else 0.0)
                 + LM["uni"].get(str(v), 0.0) * 0.01)
    return sorted(sc, key=sc.get, reverse=True)[:3]


_tick = [np.zeros(hopN * 2, dtype="float32")]


_OLV = [-120.0] * 8          # v33 最近 8 hop 的輸出位準（聲學延遲 ~5 hop）


def process_hop(chunk):
    """一個 hop 的完整管線（偵測→決策→渲染→濕）→ 輸出樣本。
    live worker 與 --file 台架共用＝離線驗證的結論對 live 成立。"""
    _tick[0] = np.concatenate([_tick[0][hopN:], chunk])
    f0 = f0_autocorr(_tick[0][-hopN * 2:])
    f0_raw = f0                          # R4A#2：法醫記門控**前**的偵測
    _lvl = 20 * np.log10(float(np.sqrt((chunk ** 2).mean())) + 1e-12)
    if f0 > 0 and _lvl < a.min_level:
        f0 = 0.0                         # v33 絕對音量門：房間噪音不算唱
    if f0 > 0 and a.bleed < 0:
        # v33 漏音門：天使 ~164ms 前的輸出 + 耦合 = 此刻麥克風裡的天使量
        if _lvl < _OLV[-5] + a.bleed + a.bleed_margin:
            f0 = 0.0
    mok = M["ok"]
    if a.mic_gate:
        # v43 麥克風門：權威從鏡頭改到 mic（08-18）。地板取「最近 8 hop
        # 輸出的**最大**電平」而非 v33 漏音門的單點 _OLV[-5]——F26 量到
        # 回授峰在 +279ms ≈ 8 hop（hop 35ms），單點賭延遲估計，窗最大值
        # 只多擋不漏擋。開/關都用 hop 計數不用牆鐘＝ --file 台架跑多快
        # 結論都成立（本檔開頭的「離線驗證對 live 成立」承諾靠這個）。
        _fl = a.mic_open
        if a.bleed < 0:
            _fl = max(_fl, max(_OLV) + a.bleed + a.bleed_margin)
        if _lvl > _fl:
            M["marm"] = M.get("marm", 0) + 1
            if M["marm"] >= a.mic_frames:
                M["mrun"] = 0
        else:
            M["marm"] = 0
            M["mrun"] = M.get("mrun", 10 ** 6) + 1
        mok = M.get("mrun", 10 ** 6) * a.hop < a.mic_hold
        if a.gate_log and mok != M.get("eff"):
            print(f"[gate] mic {'open' if mok else 'close'}"
                  f"  lvl {_lvl:.1f}  floor {_fl:.1f}", flush=True)
    elif a.mouth and not a.file:
        # R3#2 看門狗：鏡頭 worker 心跳斷（cap.read 阻塞/開機中/掛掉）
        # ＝fail-open＝裸奔有聲，不是無聲（原病：開場 2-5s 全啞、鏡頭
        # 卡死全場啞）
        # 審查：`M.get("hb", 0.0)` 讓**開場必定判定鏡頭死**＝門控裸奔
        # 3-6 秒（鏡頭挑選 + mediapipe 初始化期間），演出上台前的空檔會唱。
        # 改成「從未有過心跳＝還沒準備好＝關門」，有過才進看門狗邏輯。
        if "hb" not in M:
            mok = False
            if not st.get("bootwarn"):
                st["bootwarn"] = True
                print("[mouth] warming up: gate closed until camera is live",
                      flush=True)
            alive = True
        else:
            alive = time.time() - M["hb"] < 0.85   # R5#1：貼齊嘴部
        #   寬限 0.8s＝看門狗與門控間的無聲縫隙縮到 ~50ms
        if "hb" in M and not alive:
            mok = True
            if a.vowels:
                # 審查 A3：鏡頭死＝嘴形資料過期，母音場回 mid（render 的
                # 逐 hop 內插會平滑滑過去），不凍在最後一張嘴
                M["vw"] = _VW_MID
            if not st.get("camwarn"):
                st["camwarn"] = True
                # 08-17 review #12：這條路原本靜默裸奔——只印一行小字、不設
                # M["dead"]＝狀態列不掛 ⚠GATE DEAD、shell 大警告條抓不到，
                # 而 USB 鏡頭 stall 是**最可能**的鏡頭故障。與 crash 路徑
                # （_mouth_worker 的 except）同格式＝同一種可見性。
                M["dead"] = "camera heartbeat lost"
                print("⚠ [mouth] GATE DEAD (camera heartbeat lost) = running "
                      "ungated (feedback guard off)", flush=True)
        elif st.get("camwarn"):
            st["camwarn"] = False
            if M.get("dead") == "camera heartbeat lost":
                del M["dead"]                # 心跳回來＝解除（crash 路不解除）
            print("[mouth] camera back = gate online again", flush=True)
    M["eff"] = mok                       # R4A#2：生效門值（狀態列/view）
    if a.vowels and _VF_W is not None:
        _ABUF[0] = np.concatenate([_ABUF[0], chunk])[-_FN:]   # 每 hop 續接
    if not mok:
        # review A4/A6：v9「新音高放行」判死——bass unison（他自己的音）
        # 與 pad 和弦音（他最常唱的三/五度）都會被判「不 novel」＝反而製造
        # 0.5s 週期頓挫與「反應時間看撞不撞和弦」的不一致。回到 v2 語意：
        # 嘴閉＝一律不算你；「時好時壞」改從鏡頭端修（更快輪詢/閾值/凍結修復）
        f0 = 0.0
    if a.vowels and _VF_W is not None:
        # v19 融合場：嘴形（鏡頭 EMA）＋當下 MFCC → 機率＝混層權重。
        # **位置很重要**：在門控之後——嘴閉時 f0 已歸零＝不更新，不然
        # 天使回授會自己去改母音（v16.2 的死因，這次從結構上擋掉）。
        # 不唱時權重停在最後一個母音（比追垃圾好），render 逐 hop 內插
        # 負責平滑；鏡頭斷（camwarn）時同樣不更新＝看門狗設的 mid 站得住。
        if f0 > 0 and M.get("rv") is not None and not st.get("camwarn"):
            _P = np.abs(np.fft.rfft(_ABUF[0] * _FWIN)) ** 2
            _c = _FDCT @ np.log(_P @ _FFB.T + 1e-10)
            _x = (np.concatenate([M["rv"], _c]) - _VF_M) / _VF_S
            _l = _VF_W @ _x + _VF_B
            _e = np.exp(_l - _l.max())
            _vw = _sub(_e / _e.sum())
            if a.vw_tau > 0 and M.get("vw") is not None:
                # v20：權重本身再平滑（--vw-tau）＝音色飄的直接旋鈕
                _k = 1.0 - float(np.exp(-a.hop / a.vw_tau))
                _vw = (M["vw"] + _k * (_vw - M["vw"])).astype("float32")
            M["vw"] = _vw
    if f0 > 0:
        mf = 69 + 12 * np.log2(f0 / 440.0)
        m = int(round(mf))
        # v26：v24 的「音高遲滯死區」**整套移除**——三份 fresh 審查各自
        # 判死：①生效機制是意外的（夾住的幀被下游合理性門丟掉＝候選票
        # 不清，事後可湊票跳到半秒前擦過的音）②hys>0.5 時逃逸條件與合理
        # 性門不相交＝**回不去剛離開的音（永久卡）**，實測 5s 不動
        # ③hys 0.55 靜默漏音（每趟漏 3 音）④把換音推到 0.3s 後＝滑音觸發
        # 率 46.6%→68.9%（貼「有 autotune 就是不行」那條線）。
        # 取而代之：①下方 need/搶拍的 young 條件（剛換過去的音要更多票）
        # ②本區的 clamp——死區**做對的版本**：夾住的幀明講「他還在原來
        # 那個音」（走下面 m == st["note"] 那條＝清候選票），不是靠合理性
        # 門把幀丟掉；且死區上限鎖 0.5 半音 ⇒ 逃逸門檻 ≤1.0＝鄰音中心，
        # 永遠可達＝**卡音與靜默漏音在結構上不可能**（S2/S3 根治）。
        # 年輕音加票擋得住顫音級的快抖，但他的來回每邊停 200-400ms＝
        # 持續證據，票數擋不住，需要這條音高記憶。
        clamp = (a.deadzone > 0 and st["note"] is not None
                 and m != st["note"]
                 and abs(mf - st["note"]) < 0.5 + a.deadzone)
        if clamp:
            m = st["note"]
        # v11 已知音符拒斥＝只管「真停頓後的進場」。v10 版（隨時擋 >6
        # 半音外的在播音）被 08-12 dump 法醫定罪：他跳五度/八度進場貼到
        # 還在響的 pad/快層音＝被吃掉，pad 響多久斷多久（2.45s 中斷實錄
        # ＝pad 壽命 2.4s 分毫不差）。跳進是音樂不是回授——拒斥收縮到
        # 回授捕獲唯一的真實場景：他停 ≥0.5s、嘴還在寬限內開著、天使殘
        # 響仍在，此時貼著任何在播音的偵測一律存疑（不分遠近）。
        # ⚠ 本區塊**不是死碼**：上面算的 mf/m 是下游狀態機的輸入，只有
        # 拒斥邏輯被移除（R3）。照註解刪整塊＝NameError。
        # R3 判決：已知音符拒斥層**整層移除**——實測複音回授被自相關鎖到
        # 「和弦共同週期」幽靈音（例：在播 48/64/67 → 偵測 38.5），離任何
        # 在播音都 >0.4 半音＝這層在真實編制下命中率 ≈ 0（假保護），唯一
        # 非冗餘窗只有嘴閉後 0.35-0.8s 的 0.45 秒，卻帶著 fail-open 抽吸
        # 迴歸（R3#1）。真防線＝嘴部門控；鏡頭死＝誠實裸奔（下方看門狗）。
        pass
    if f0 > 0:
        st["quiet"] = 0
        # R4B 合理性門：離半音格 >0.35 的偵測＝換音過渡的滑行幀，不算票
        # ——假鄰音（全音換音 4/6 出現 70-100ms 假中繼音）與假八度 commit
        # 的根治；kt/cents 不吃過渡污染、顫音 ±60c 峰不再擦邊
        plaus = abs(mf - m) <= 0.35 or clamp
        if plaus and not clamp:
            # clamp 幀＝他的音在兩格之間，cents/調性不吃這種污染（審查 B1
            # 記過「死區凍結 cents」——這裡是刻意的：過渡幀本來就不投票）
            kt.push(mf)
            # 和弦鎖定：他對量化音的 cents 偏移 → EMA → 播放速率
            st["cents"] += 0.25 * ((mf - m) * 100 - st["cents"])
            st["lastm"] = m
        if not plaus:
            pass                         # 過渡幀：不投票、不清票
        elif m == st["note"]:
            st["cand"], st["cc"] = None, 0
        elif m == st["cand"]:
            st["cc"] += 1
            tno = st.get("t", 0.0)
            # R2A#6：不應期涵蓋兩票路徑；顫音對偶（A↔B 來回）要 4 票
            need = 2
            if m == st.get("pn") and tno - st["tsw"] < a.rebound:
                need = 4                 # v15：只有超快回跳才算顫音抖動
            elif st["note"] is not None and abs(m - st["note"]) > 7:
                need = 3                 # R5#3：大跳多一票＝假八度瞬態
                #   （R4B 實測 105ms 假 52）死、真八度只慢 ~35ms
            if a.young > 0 and tno - st["tsw"] < a.young:
                need = max(need, 2 + a.young_need)   # v26 年輕音加阻力
            if st["cc"] >= need and tno - st["tsw"] > 0.10:
                if st["note"] is not None:
                    st["hist"] = (st["hist"] + [m - st["note"]])[-4:]
                st["porta_ok"] = tno - st["tsw"] >= 0.3   # v15
                st["pn"] = st["note"]
                st["note"] = m
                st["tsw"] = tno
                st["cand"], st["cc"] = None, 0
        else:
            st["cand"], st["cc"] = m, 1
            breath = (not a.file and a.mouth
                      and time.time() - M["t_on"] < 1.0)
            hab = False
            if (LM is not None and st["note"] is not None
                    and len(st["hist"]) >= 2):
                hab = (m - st["note"]) in lm_top3(st["hist"][-2],
                                                  st["hist"][-1])
            cool = st.get("t", 0.0) - st["tsw"] > 0.10
            # review B4：級進本身就是強證據，不再綁調內（KeyTracker 冷啟動
            # 前 root=0＝非 C 調的歌全被誤擋）；調內判斷只留給呼吸進場，
            # 且 key 未定時放行
            keyok = (kt.h.sum() < kt.MIN
                     or (m - kt.root()) % 12 in MAJ)
            rebound = (m == st.get("pn")
                       and st.get("t", 0.0) - st["tsw"] < a.rebound)
            young = (a.young > 0 and st["note"] is not None
                     and st.get("t", 0.0) - st["tsw"] < a.young)
            if (a.fast and cool and not rebound   # v15：只擋超快回跳
                    and not young                 # v26：年輕音不給一票通道
                    and ((st["note"] is not None
                          and abs(m - st["note"]) <= 2)
                         # 審查 A3：hab 原本**繞過假八度防護**——melody_lm
                         # 的 293 個情境有 50% top-3 含 >7 半音，一幀 105ms
                         # 的假八度撞上習慣模型就 35ms commit。搶拍只給
                         # 五度內的跳，大跳一律回慢路徑吃 R5#3 的三票。
                         or (hab and abs(m - st["note"]) <= 7)
                         or (breath and st["note"] is None and keyok))):
                # v6 搶拍：一票 commit（~35ms）；150ms 不應期防顫音風暴
                if st["note"] is not None:
                    st["hist"] = (st["hist"] + [m - st["note"]])[-4:]
                st["porta_ok"] = (st.get("t", 0.0) - st["tsw"]) >= 0.3  # v15
                st["pn"] = st["note"]
                st["note"] = m
                st["tsw"] = st.get("t", 0.0)   # review A3/B11
                st["cand"], st["cc"] = None, 0
    else:
        st["quiet"] += 1
        st["cents"] *= 0.9               # review A13：無聲時音準偏移歸零，
        #                                  不讓上一個音的 cents 掛在墊上
        if st["quiet"] >= max(1, int(0.25 / a.hop)):
            st["note"] = None            # 停 ~0.25s＝放
            st["cand"], st["cc"] = None, 0   # review B7：殘留候選一併清
    root = kt.root()
    n = st["note"]
    for vi, (p, (_nm, _md, _sp, _g, _rng, sh, th)) in enumerate(
            zip(players, VOICES)):
        if n is None:
            p.set_note(None)
        elif a.vl >= 2 and vi > 0:
            # v36 合唱配置（--vl 2）：舊版三個上聲部**共用同一組候選、又都
            # 用「離自己上一個音最近」**＝一起單向棘輪爬到最高候選後鎖死，
            # 實測上三聲部 95% 唱同一顆音（+19 半音），四聲部等於三份齊唱。
            # 新版每個聲部各自：①候選＝和弦音在**自己音域內**的所有八度
            # ②評分＝聲部進行(|Δ|) ＋ 音域向心力 ＋ 同音懲罰 ＋ 交叉懲罰
            # ③由低到高依序指派（下面已定的音當作不可交叉的地板）。
            lo_, hi_ = _rng
            ctr = (lo_ + hi_) / 2.0
            offs = (0, dia_step(n, root, 2), dia_step(n, root, 4))
            cands = [n + o + 12 * k for o in offs for k in range(-2, 4)]
            cands = [c for c in cands if lo_ <= c <= hi_]
            if not cands:
                cands = [int(np.clip(n + sh, lo_, hi_))]
            prev = p.cur[0] if p.cur is not None else ctr
            floor_ = st.get("vlfloor", None)

            def _score(c):
                v = abs(c - prev) + 0.6 * abs(c - ctr)
                if floor_ is not None:
                    if c == floor_:
                        v += 8.0          # 同音（齊唱）＝要有代價
                    elif c < floor_:
                        v += 14.0         # 聲部交叉
                return v
            pick = min(cands, key=_score)
            st["vlfloor"] = pick
            p.set_note(pick)
        elif a.vl and vi > 0:
            # v4 voice-leading：bass 恆根音；上聲部從三/五/八度系
            # 候選中選離自己上一個音最近的＝小步進行（撞同音＝double，
            # 合唱團本來就有）
            third = dia_step(n, root, 2)
            fifth = dia_step(n, root, 4)
            cands = [n + third, n + fifth, n + 12,
                     n + 12 + third, n + 12 + fifth]
            prev = (p.cur[0] if p.cur is not None else
                    n + sh + (dia_third(n + sh, root) if th else 0))
            p.set_note(min(cands, key=lambda c: abs(c - prev)))
        else:
            t = n + sh + (dia_third(n + sh, root) if th else 0)
            p.set_note(t)
        if vi == 0:
            # 08-17 review #5：vlfloor 記 bass **實際發聲**的音（set_note 之後
            # 的 p.cur），不是折疊前的 n+sh——set_note 會把低於音庫下限的目標
            # 上折八度（bass -8 之後 n≤43 就會折），記折疊前的值＝地板低報
            # 12 半音＝tenor 的同音(+8)/交叉(+14)懲罰對最低那些音整段失效、
            # tenor 可能被配到 bass 實際音高之下（sh=0 時代不可能發生的迴歸）。
            st["vlfloor"] = (p.cur[0]
                             if (n is not None and p.cur is not None) else None)
    if PP > 1:
        # 團員**跟著自己聲部的領唱**，不自己決定音。voice-leading 用 p.cur
        # 當「上一個音」評分（同音罰 8／交叉罰 14），四個團員各自跑一次
        # 會各自棘輪到不同的音＝一個聲部散成四聲部。真合唱團的分部也是
        # 一條線多個人唱，不是每個人自己挑。
        # 用 p.tgt 判有沒有音：p.cur 在放掉之後仍留著上一個音。
        for vi, p in enumerate(players):
            m_ = p.cur[0] if (p.cur is not None and p.tgt > 0) else None
            for mm in MEMS[vi]:
                mm["p"].set_note(m_)      # 同音＝set_note 直接 return，不重觸發
    if pads:
        tmono = st.get("t", 0.0)
        if n is not None and st["pada"] != n and \
                tmono - st["padt"] >= a.pad:
            # v6 慢層：錨音換和弦（駐留限制＝和聲節奏）；三度/五度看調
            st["pada"], st["padt"] = n, tmono
            r_ = kt.root()
            # 審查：pads 依聲部數建（v31 起可能 4 個）但和弦只有 3 音，
            # zip 會**靜默截斷**＝最後一個聲部的墊從頭到尾是啞的。和弦
            # 依 pads 數展開（第 4 音補高八度根音）。
            chord = [n - 12, n + dia_step(n, r_, 2), n + dia_step(n, r_, 4),
                     n][:len(pads)]
            for pp, cnote in zip(pads, chord):
                pp.set_note(cnote)
        elif n is None and st["quiet"] >= max(1, int(a.pad_hold / a.hop)):
            st["pada"] = None
            for pp in pads:
                pp.set_note(None)
    rate = 2 ** (a.lock * st["cents"] / 1200.0)
    # 審查 B：法醫權重在 render **前**取樣＝記到的就是本 hop 用的場
    vw_h = (np.asarray(M.get("vw", _VW_MID)).copy()
            if a.vowels and dmp is not None else None)
    y = np.zeros((hopN, NCH), dtype="float32")    # NCH=2（無 --out-map）＝舊形狀
    tnow = st["t"] = st.get("t", 0.0) + a.hop     # 檔案模式也要走假時鐘
    for vi, p in enumerate(players):
        if a.human > 0:
            f1, p1, f2, p2 = HUM[vi]
            c = a.human * (0.6 * np.sin(6.28 * f1 * tnow + p1)
                           + 0.4 * np.sin(6.28 * f2 * tnow + p2))
            r = rate * 2 ** (c / 1200.0)
        else:
            r = rate
        if a.vib > 0:
            # v37 顫音（審查：他的 3-8Hz 音高能量天使只還原 2.2%＝**天使
            # 完全沒有顫音**，這是「修過音」最強的單一指紋，勝過換音瞬間）。
            # 音齡 >--vib-delay 才漸入＝換音當下乾淨、長音才活過來（真唱者
            # 也是這樣）。逐樣本（借 v25 的 rate 陣列）＝5.5Hz 不會被 35ms
            # 的 hop 切成階梯。
            age = tnow - st["tsw"]
            if age > a.vib_delay:
                dep = a.vib * min(1.0, (age - a.vib_delay) / 0.3)
                ph = VIBPH[vi]
                t_ = tnow + np.arange(hopN, dtype="float64") / SR
                r = r * 2 ** (dep * np.sin(6.28 * a.vib_rate * t_ + ph)
                              / 1200.0)
        w = p.render(hopN, r) * MIXG[VOICES[vi][0]] * MEMG
        if OMAP is not None:
            y[:, OMAP[vi]] += w              # 路由：整聲部進自己那道（不 pan）
        else:
            gl, gr = _pan(PAN_F[vi])
            y[:, 0] += w * gl
            y[:, 1] += w * gr
        for mm in MEMS[vi]:
            # 團員：跟領唱同一個音、同一份音庫，但每個人有自己的失諧／
            # 微漂／顫音相位／站位／起音延遲。刻意重寫一次而不是把領唱那
            # 段抽成函式——抽出來就動到了已認證的路徑，PP=1 的 byte-identical
            # 保證會變成「要重新證明」。
            rm = rate * mm["det"]
            if a.human > 0:
                f1, p1, f2, p2 = mm["hum"]
                cm = a.human * (0.6 * np.sin(6.28 * f1 * tnow + p1)
                                + 0.4 * np.sin(6.28 * f2 * tnow + p2))
                rm = rm * 2 ** (cm / 1200.0)
            if a.vib > 0:
                age_m = tnow - st["tsw"]
                if age_m > a.vib_delay:
                    dep_m = a.vib * min(1.0, (age_m - a.vib_delay) / 0.3)
                    t_m = tnow + np.arange(hopN, dtype="float64") / SR
                    rm = rm * 2 ** (dep_m * np.sin(6.28 * a.vib_rate * t_m
                                                   + mm["vib"]) / 1200.0)
            wm = _mdelay(mm, mm["p"].render(hopN, rm)
                         * MIXG[VOICES[vi][0]] * MEMG)
            if OMAP is not None:
                y[:, OMAP[vi]] += wm         # 團員跟領唱同一道
            else:
                glm, grm = _pan(mm["pan"])
                y[:, 0] += wm * glm
                y[:, 1] += wm * grm
    for vi, pp in enumerate(pads):
        f1, p1, f2, p2 = HUM[len(VOICES) + vi]
        c = (a.human or 0) * (0.6 * np.sin(6.28 * f1 * tnow + p1)
                              + 0.4 * np.sin(6.28 * f2 * tnow + p2))
        w = (pp.render(hopN, rate * 2 ** (c / 1200.0)) * a.pad_gain
             * MIXG[VOICES[vi][0]])
        if OMAP is not None:
            y[:, OMAP[vi]] += w              # pad 也跟該聲部同一道
        else:
            gl, gr = _pan(PAN_P[vi])
            y[:, 0] += w * gl
            y[:, 1] += w * gr
    if RVIR is not None:
        # 殘響走 mono 匯流排（擴散場）、回灌全部聲道＝空間包住站位。
        # OMAP 下用 sum（每個樣本只在一道＝sum 才是完整混音）；舊路徑
        # mean 不動（byte-identical）。
        _bus = y.sum(axis=1) if OMAP is not None else y.mean(axis=1)
        wf = _fftc(_bus.astype(float), RVIR)
        wet_ = wf[:hopN]
        tl = RVT[0]
        wet_ += tl[:hopN]
        nt = wf[hopN:]
        rest = tl[hopN:]
        nt[:len(rest)] += rest
        RVT[0] = nt
        y = (y + a.wet * wet_[:, None]).astype("float32")
    y = np.clip(y * a.gain, -1, 1).astype("float32")
    if OMAP is not None:
        # 08-18 live 實測（回授漲到斷電）抓到的路由回歸：array-rms 會隨聲道
        # 數稀釋（4ch 比立體聲低 3dB）＝回授地板讀低、mic 門更容易被回授
        # 撐開。換算回「立體聲等效」尺度（總能量 ÷ 2 道）＝-13.5 的 bleed
        # 校準口徑不變。⚠ 校準本身是舊喇叭擺位量的，4 喇叭散開後 k 要重掃。
        _OLV.append(20 * np.log10(
            float(np.sqrt((y ** 2).sum() / (y.shape[0] * 2))) + 1e-12))
    else:
        _OLV.append(20 * np.log10(float(np.sqrt((y ** 2).mean())) + 1e-12))
    del _OLV[0]
    if dmp is not None and len(dmp["mic"]) < DMPMAX:
        dmp["mic"].append(chunk.copy())
        dmp["out"].append(y.copy())
        dmp["mok"].append(1.0 if mok else 0.0)   # R4A#2：生效門值
        dmp["face"].append(1.0 if M.get("face") else 0.0)
        dmp["mon"].append(1.0 if M["on"] else 0.0)
        dmp["val"].append(float(M.get("val", -1.0)))  # v27 開口值
        dmp["wd"].append(float(M.get("wd", -1.0)))    # v28 嘴寬
        dmp["sp"].append(float(M.get("sp", -1.0)))    # v34 在唱機率
        dmp["f0"].append(f0_raw)                 # R4A#2：門控前偵測
        dmp["note"].append(-1 if st["note"] is None else st["note"])
        dmp["vn"].append([p.cur[0] if p.cur else -1 for p in players])
        if a.vowels:
            dmp["vw"].append(vw_h)
    elif dmp is not None and not st.get("dumpfull"):
        st["dumpfull"] = True                    # v44：滿了停錄、明講一次
        print(f"⚠ dump buffer full (--dump-max-min {a.dump_max_min:g}) = "
              f"later audio not recorded (performance unaffected)", flush=True)
    return y


def worker():
    res = np.zeros(0, dtype="float32")
    while not st["die"]:
        with qlock:
            pend = np.concatenate(in_q) if in_q else None
            in_q.clear()
        if pend is None:
            time.sleep(0.002)
            continue
        res = np.concatenate([res, pend])
        while len(res) >= hopN:
            chunk, res = res[:hopN], res[hopN:]
            try:
                y = process_hop(chunk)
            except Exception as e:
                # 08-17 review #11：原本裸奔——process_hop 一個例外就殺掉唯一
                # 產音的執行緒，callback 照跑＝整場靜音、xrun 凍結、儀表全綠
                # ＝live 樂器最糟的死法。錯一個 hop 補一塊靜音照常前進；
                # ⚠ 行進大警告條（shell warn pattern 認 ⚠ 開頭）。
                st["hopfail"] = st.get("hopfail", 0) + 1
                if st["hopfail"] <= 3 or st["hopfail"] % 200 == 0:
                    import traceback
                    traceback.print_exc()
                    print(f"⚠ [engine] process_hop failed #{st['hopfail']} "
                          f"({e!r}) — inserting silence for this hop",
                          flush=True)
                y = np.zeros((hopN, NCH), dtype="float32")
            with qlock:
                out_q.append(y)
                while sum(len(q) for q in out_q) > a.maxlag * hopN:
                    out_q.pop(0)                 # 積壓上限＝--maxlag 塊；
                    #   review A10 誠實註：丟最舊＝持續音挖掉一塊（會有 35ms
                    #   跳格），是「延遲不沉澱」的代價，靠 A9 的淡出遮爆音
                    # v25（審查 S2）：延遲棘輪——worker 一次卡頓造成的積壓
                    #   **永遠不會被追回**（cb 每次只取剛好填滿的量），最壞
                    #   常駐 +105ms 而 xrun 完全照不到（callback 都填滿了）。
                    #   上限改旗標（預設仍 3＝舊行為），並在丟塊時記一筆＋
                    #   讓下一塊淡入（原本硬接）。
                    st["drop"] = st.get("drop", 0) + 1
                    UF[0] = True


if a.file:
    x, sr = sf.read(a.file[0], dtype="float32", always_2d=True)
    assert sr == SR, (a.file[0], sr)
    x = x[:, 0]
    outs = [process_hop(np.ascontiguousarray(x[i:i + hopN]))
            for i in range(0, len(x) - hopN + 1, hopN)]
    yy = np.concatenate(outs)
    sf.write(a.file[1], yy, SR)
    _write_dump()                                    # review A15：別默丟 dump
    print(f"{len(x)/SR:.1f}s -> {a.file[1]}  rms "
          f"{float(np.sqrt((yy**2).mean())):.4f}  "
          f"glide {st.get('porta_n', 0)}", flush=True)
    raise SystemExit

OB = [np.zeros((0, NCH), dtype="float32")]
UF = [False]                              # 上一個 callback underrun 過
LB = [np.zeros(NCH, dtype="float32")]     # 最後輸出樣本（underrun 淡出用）


def cb(indata, outdata, frames, tinfo, status):
    if status:
        st["xrun"] += 1
    with qlock:
        in_q.append(indata[:, 0].copy())
        grabbed = []
        need = frames - len(OB[0])
        while out_q and need > 0:
            b = out_q.pop(0)
            grabbed.append(b)
            need -= len(b)
    # review A8：concatenate 移出鎖外（RT 執行緒持鎖配置記憶體＝優先權反轉）
    if grabbed:
        OB[0] = np.concatenate([OB[0]] + grabbed) if len(OB[0]) else (
            grabbed[0] if len(grabbed) == 1 else np.concatenate(grabbed))
    buf = OB[0]
    if len(buf) >= frames:
        o = buf[:frames]
        if UF[0]:
            # review A9：underrun 復原第一塊淡入（不然又是一個階躍）
            o = o.copy()
            k = min(96, frames)
            o[:k] *= np.linspace(0, 1, k, dtype="float32")[:, None]
            UF[0] = False
        outdata[:, :NCH] = o
        LB[0] = np.asarray(o[-1]).copy()
        OB[0] = buf[frames:]
    else:
        # review A9：underrun 不再硬切零——從最後樣本短淡出
        outdata.fill(0)
        nb_ = len(buf)
        if nb_:
            outdata[:nb_, :NCH] = buf
            last = np.asarray(buf[-1])
        else:
            last = LB[0]
        k = min(96, frames - nb_)
        if k > 0:
            outdata[nb_:nb_ + k, :NCH] = (last[None, :]
                                          * np.linspace(1, 0, k,
                                                        dtype="float32")[:, None])
        UF[0] = True
        LB[0] = np.zeros(NCH, dtype="float32")   # R2A#5：連續 underrun＝靜音，
        OB[0] = np.zeros((0, NCH), dtype="float32")   # 不是 86Hz 脈衝串


import sounddevice as sd  # noqa: E402

_WT = threading.Thread(target=worker, daemon=True)
_WT.start()
try:
    VIEW = a.view and a.mouth and not a.file
    FB64 = a.frame_b64 > 0 and a.mouth and not a.file
    if VIEW or FB64:
        import cv2 as _cv                 # imshow 必須在主執行緒（macOS）
    if FB64:
        import base64 as _b64
    with sd.Stream(samplerate=SR, blocksize=a.sblock, channels=(1, NCH),
                   device=(a.in_name, a.out_name), dtype="float32",
                   latency="low", callback=cb):
        # ready 在**開流之後**才印（08-17 review）：它是 respond_shell 切換
        # 退場舊引擎、UI 顯示 SING 的依據——原本在開流前印，音訊裝置還沒到
        # 手 UI 就說「合唱團跟著你」，切換也會提早殺掉舊引擎。
        print(f"ready (sample bank: detection window {a.hop*1000:.0f}ms x2, "
              f"note change fast ~{a.hop*1000:.0f}ms / slow ~{a.hop*2000:.0f}ms, "
              f"lock {a.lock}; Ctrl-C to stop)", flush=True)
        _ts = 0.0
        _tf = 0.0
        try:
            while True:
                time.sleep(0.1 if (VIEW or FB64) else 2)   # R5：view 30Hz 實測
                #   +12 xrun/分鐘（imshow 吃 RT）；10fps 監看夠用、xrun 回本底
                if VIEW and M.get("frame") is not None:
                    _cv.imshow("mouth gate", M["frame"])
                    _cv.waitKey(1)
                if (FB64 and M.get("frame") is not None
                        and time.time() - _tf >= 1.0 / a.frame_b64):
                    # 給 respond_shell 的影像通道：單行 FRAME <b64>，shell 特判
                    # 不進 log。q60/480 寬 ≈ 20KB/幀，5fps ≈ 100KB/s＝pipe 無感。
                    _tf = time.time()
                    okj, _jb = _cv.imencode(".jpg", M["frame"],
                                            [int(_cv.IMWRITE_JPEG_QUALITY), 60])
                    if okj:
                        print("FRAME " + _b64.b64encode(_jb).decode(), flush=True)
                if time.time() - _ts < 2:
                    continue
                _ts = time.time()
                # 08-17 review #11：worker 活性偵測——它是唯一產音的執行緒，
                # 死了 callback 照樣填 underrun 淡出＝無聲但儀表全綠。
                if not _WT.is_alive() and not st.get("wdead"):
                    st["wdead"] = True
                    print("⚠ [engine] audio worker thread died — OUTPUT IS "
                          "SILENT (restart the engine)", flush=True)
                nt = st["note"]
                print(f"note {nt if nt is not None else '—'}  "
                      f"cents {st['cents']:+5.1f}  "
                      f"mouth {'open' if M.get('eff', M['ok']) else 'closed'}  "
                      + f"xrun {st['xrun']}"
                      # v25：積壓（塊）與丟塊數＝延遲棘輪的儀器（原本全盲）
                      + f"  backlog {sum(len(q) for q in out_q) // hopN}"
                      + (f"/dropped {st['drop']}" if st.get("drop") else "")
                      # v40：門為什麼開？sp＝模型機率、mo＝下巴運動量（否決票）。
                      # 沒有這兩個數字，誤觸只能用猜的（08-15 繞了一整天）
                      + (f"  sp {M.get('sp', 0.0):.2f} mo {M.get('mo', 0.0):.4f}"
                         if a.sing_gate and M.get("sp") is not None else "")
                      # v16/v17：附加欄接在**行尾**——既有欄位格式一字不動
                      # （respond_shell 的 regex 靠它）。場＝五層權重百分比
                      + (("  field " + "/".join(
                          f"{nm_}{int(round(w * 100))}"
                          for nm_, w in zip(_SEL_NAMES, M.get("vw", _VW_MID))))
                         if a.vowels else "")
                      # 審查：警告要接**行尾**——原本插在中間會撞壞 shell 的
                      # bank_st 正則＝門控一死，警告與唯一的即時讀數同時消失
                      + ("  ⚠GATE DEAD" if M.get("dead") else "")
                      + ("  ⚠WORKER DEAD" if st.get("wdead") else ""),
                      flush=True)
        except KeyboardInterrupt:
            pass
        finally:
            # 08-17 review #13：dump 在**關流之前**落檔（stream context 內）。
            # 舊碼把它放在 with 的 __exit__ 之後＝CoreAudio 關流掛死（playbook
            # 已知）時 shell 的 8s SIGKILL 讓整份錄音陪葬——而「耳朵是唯一的
            # 儀器、事後讀 dump」整套方法論都掛在這份檔案上。
            st["die"] = True
            time.sleep(0.2)              # 讓 worker 吐完手上的 hop
            _write_dump()
except KeyboardInterrupt:
    pass                                 # 關流期間再按 Ctrl-C：dump 已寫完
print("bye")
