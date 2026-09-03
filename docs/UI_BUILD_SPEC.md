# Solo Choir — UI Build Spec (handoff for Claude Code)

設計定稿 = `solo_choir_ui_v10.html`(冷色淺主題、SATB 玻璃頭排列、Record/Play 置中、錄製→轉譜→分聲部播放、Ensemble/Spread、輸入電平/延遲/音準三項回饋,**+ GLTFLoader `.glb` 頭模管線含 fallback**)。
本文件是把這個**視覺/互動定稿**接上**已驗證引擎**的規格。

---

## 0. 核心原則
- **引擎別動。** Beatrice 移植、stateful `soxr.ResampleStream`、`SoloChoir.py` 的 diatonic 邏輯都已驗證可用。Claude Code 只在引擎**外圍**加「控制 + 遙測 + 打包」層,**不改音訊 DSP**。
- 原型裡所有「動的東西」目前都是**假的(demo)**,要逐項換成引擎的真實資料(見 §4)。

---

## 1. 形態:獨立桌面 app(非 web)
**決定:用 `pywebview` 把 v10 的 HTML/three.js 包成原生 macOS 視窗,打包成 `.app`。**

- UI 完全沿用 v10(WKWebView 支援 WebGL / three.js)。
- Python 引擎與 UI 同進程:GUI 跑主執行緒,音訊引擎跑背景執行緒。
- 打包:`py2app`(macOS 原生)或 `pyinstaller`。
- 替代方案僅供參考:Electron(最肥)、Tauri(輕但需 Rust + Python sidecar)、PyQt/QML(等於放棄 WebGL 頭)。**預設走 pywebview。**
- ⚠️ 授權:Beatrice 二進位禁止散佈。本機自用/論文展示可打包;**不要對外發佈含該 binary 的 app**。

---

## 2. 架構 / 橋接
兩種橋接擇一(pywebview 兩者都支援):

**A. pywebview js_api(最簡單,單一進程)**
- Python 用 `window.expose()` 提供控制函式;UI 變更時呼叫。
- 引擎每幀用 `window.evaluate_js("onTelemetry({...})")` 把資料推進 UI。

**B. localhost WebSocket(解耦,之後較好擴充)**
- harness 開 `ws://localhost:8770`。
- 引擎**推**遙測 ~30–60fps;UI **送**控制。

**遙測(引擎 → UI):**
```
{ f0: Hz, midi: float, cents: float, level: 0..1(RMS), rtf: float, latency_ms: float, voiced: bool }
```
**控制(UI → 引擎):**
```
{ convert: bool, key: 0..11, scale: "major"|"minor",
  intervals: [-7,-2,2,4] 中目前啟用者, harmonize: bool,
  pitch: st, gain: dB, gate: dB }
```

---

## 3. 3D 頭模
**`solo_choir_ui_v10.html` 已實作這條流程**,本節描述它、以及進 Claude Code 後要做的事。

**v10 已具備:**
- `GLTFLoader` 載入 `.glb` → 取場景第一個 mesh 的 geometry → `normalizeHeadGeo()`(置中、縮放到 `MODEL_TARGET_H`、可用 `MODEL_ROT_Y`/`MODEL_TILT` 校正朝向)→ 套上既有**玻璃 shader**(fresnel + additive + 由下往上淡出)。
- **退場保險**:`GLTFLoader` 不存在 / 載入失敗 / 9s timeout → 自動 fallback 回程序化頭,畫面不會壞。
- **狀態徽章**(右下角):綠 = `model` 載入成功;橘 = `procedural` fallback。方便驗證。
- 載入後才 `init()` 建場景與 `tick()`,事件處理對空 `pickables` 安全。

**進 Claude Code / pywebview 要改:**
- **改成本地檔**:`const MODEL_URL = "./assets/head.glb";`(本地檔→CORS 問題消失,也最穩)。
- 朝向不對就調 `MODEL_ROT_Y`,大小用 `MODEL_TARGET_H`。
- 現成頭(目前的 LeePerrySmith)只是**驗證流程**;之後換自己的掃描頭只需換 `MODEL_URL`,管線不動。

**待辦 / 注意:**
- **效能**:真實頭模建議**降面數 / 做 LOD**(WKWebView 內 5 顆高模 + 幽靈頭);ghost 已用低模球體。
- **每聲部變化**:目前 5 顆共用同一 geometry + 不同顏色;之後可對應不同音色給不同頭。
- **可選(切題加分)**:用自己的頭部掃描(iPhone LiDAR / photogrammetry)做「You」那顆,呼應「身體作為介面」。
- 對 model 版本已關掉 sphere inner lattice、降低 wireframe 透明度(高模上 wireframe 會太密)。

---

## 4. 原型是假的 → 要換成真的
| 原型現況(mock) | 換成 |
|---|---|
| demo 旋律迴圈(`phrase`) | 引擎即時 **f0** |
| 讀數 Sung/Choir | 真 f0 + `diatonicShift`(已與 `SoloChoir.py` 一致) |
| 輸入電平表(模擬) | 真實麥克風 **RMS** |
| Tuning cents 指針(模擬) | 真 f0 對最近音階音的 **cents** 偏差 |
| `LIVE · 12 MS` | 引擎真實 **RTF / latency** |
| Record → 樂譜(假音符) | 真錄音 → **Basic Pitch** 轉譜 |
| Play / 分聲部 | 真播放 take(melody / choir / both) |
| Ensemble 幽靈頭、Spread | **Phase 2**(目前純視覺) |
| Calibrate(動畫) | 真校準(取 RMS 基準設 gate/gain) |

---

## 5. UI 控制 → 引擎參數對照
- **Convert** → 啟/停 `beatrice_solo_choir_live.py`
- **Key + Scale** → `--key` / major·minor
- **每顆頭開關**(Bass −7 / Tenor −2 / Alto +2 / Sop +4)→ 啟用哪些 interval pass;**You = 來源**
- **Harmonize** → 和聲總開關
- **Pitch 滑桿** → 引擎 transpose(formant-preserving)
- **Output gain / Noise gate** → 引擎參數

---

## 6. 開工順序(對著 6/29 Tech Form)
1. **pywebview 殼**:把 v10 開進原生視窗,確認 WebGL/three.js 正常。
2. **橋接層**:harness 加 §2 的控制/遙測(不碰 DSP)。
3. **接活訊號**:讀數 + 頭 + Convert + key/scale/intervals(拿掉 demo 迴圈)。
4. **接回饋**:input level + latency + tuning。
5. **Record → Basic Pitch → 樂譜 → Play。**
6. **3D 頭模**載入(§3),先單顆驗證再套全部。
7. → 到這裡就有可展示 / 可錄影的 MVP。
8. **打包 `.app`。**

> Ensemble / 多音色 / score 進階 / 骨傳導硬體 = **Tech Form 之後**。

---

## 7. Phase 2(之後)
- **真・多聲部**:每聲部跑 N 次推論再混。
- **多音色**:不同聲音模型 **或** 同一 Beatrice 模型的不同 **speaker embedding**(模型 `.bin` 內已含 `speaker_embeddings`——先跟 Claude Code 確認能否每聲部即時選用)。
- **Ensemble 厚度**:每聲部多 pass + 微量音高/時間離散再混(= Synth V 那種人味的即時版)。
- **Score 模式**強化、**骨傳導 / 穿戴輸出**(Dayton BCE-1、PAM8403、Pico W I/O)。

---

## 8. 雜務提醒
- [ ] 把 `~/SoloChoir_engine_model_backup/` 搬離筆電(外接/私有雲,**勿公開**——授權)。
- [ ] `BEATRICE_SOLO_CHOIR.md` / README 裡指向 `server/voice_changer/SoloChoir.py` 的路徑修正。
- [ ] MVP 一跑起來就**錄螢幕 + 系統實拍**——viva / 文件最關鍵的素材。
