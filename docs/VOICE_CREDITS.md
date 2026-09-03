# VOICE CREDITS — 誰的聲音在裡面唱

建立 2026-08-19。**這份只記「出貨路徑上真的會發聲的模型」**：viva、展場、作品集會被聽到的那些。
訓練配方與實驗紀錄在 `TRAINING_NOTES.md`；本檔只回答一個問題：**這個聲音是誰的、我憑什麼用它。**

> 紀律出處：`TRAINING_NOTES.md §8`「Record donor consent for any non-own voices.」
> 繳交文件的 Ethical Considerations 段聲明「credited in the repository」＝指這一份。

---

## 1. 朋友本人錄的聲音（唯一的非語料人聲）

| 欄位 | 內容 |
|---|---|
| **模型** | `260724_ddsp_svc/exp/combsub-girl/model_30000.pt`（DDSP-SVC CombSub，30000 步） |
| **用在哪** | **Respond 模式的上聲部**（`harmony/respond2.py:66` `VOICES["upper"]`）。混進送出去的回應聲（`respond2.py:502`），不是備用聲部 |
| **聲音是誰的** | Harry 的朋友，本人當面錄音。**依本人意願匿名，對外一律寫 “a friend”**（Harry 2026-08-19 確認） |
| **錄音來源鏈** | `260620/test/women/girl_clip1–4.m4a`（iOS 語音備忘錄，2026-06-19 建，110 秒）→ `260618/girl_clip 5,7,8,11.wav` → 訓練集 `260724_ddsp_svc/data_girl/train/audio/` 245 切片 |
| **授權依據** | **本人同意**。Harry 2026-08-10 確認已取得 |
| **同意紀錄存放** | **Harry 手機裡的訊息對話**（Harry 2026-08-19 確認）。本 repo 不放個資檔。被問到時可出示該對話 |
| **範圍** | 學術用途（MA 作品、viva、展場、作品集）。商業使用未涵蓋，要另外問 |
| **模型檔** | git-ignored，不隨 repo 散布（HARD RULE 3） |

✅ **這條已落檔完成（2026-08-19）**：本人同意、同意紀錄可出示、對外匿名、致謝在此。
⚠ 仍要記得的兩件：①這是唯一可辨識到個人的聲音，依系上 Ethics checklist 第 12 項（Visual or Vocal Methods）屬個人資料；②同意紀錄只存在手機對話裡＝**換機或清訊息就沒了**，建議另外截圖存一份到電腦。

---

## 2. 公開歌唱語料訓練的聲音

全部來自 **M4Singer**（Zhang et al., NeurIPS 2022 Datasets & Benchmarks）。授權 **CC BY-NC-SA**＝非商業、須署名、相同方式分享。授權查證：vault `Solo Choir/查證 2026-08-04 開源歌唱語料授權`。

| 模式 | 聲部 | 模型 | 原唱者（語料內編號） |
|---|---|---|---|
| Respond | 低聲部 | `exp/combsub-m4-bass1/model_11000.pt` | Bass-1 |
| Respond | 高聲部 | `exp/combsub-m4-sop3/model_10000.pt` | Soprano-3 |
| Live（取樣合唱團） | bass | `exp/reflow-bass1/model_32000.pt` | Bass-1 |
| Live | alto | `exp/reflow-alto3/model_40000.pt` spk2 | Alto-5／6／7 合訓，出貨用 Alto-6 |
| Live | sop | `exp/reflow-sop3/model_20000.pt` | Soprano |
| Live | tenor 領唱嘴 | `exp/reflow-male8/…` spk7 | Tenor-5（08-16 v42 換上） |

**訓了但沒出貨**：`reflow-male8`／`reflow-fem8` 各 8 位歌手（M4Singer，本機 40000 步）。08-16 盲聽判「真人音色不加分」⇒ 凍結配置 `--mem-real 0` 不載入。除了上表的 tenor 領唱嘴之外，這兩顆模型的其他歌手在演出中不發聲。

---

## 3. Harry 自己的聲音

| 模式 | 用在哪 | 模型 |
|---|---|---|
| Live · neural | 低聲部 | `model/VC/dist/model_dir/1/model/paraphernalia_data_new25_2k`（2026-06-20 錄的 F2–C5 資料，約 21 分鐘，2000 步） |
| Respond／Live | 乾聲直通 | 不經模型，麥克風原聲 |

---

## 4. Beatrice 引擎附帶的模型

| 用在哪 | 模型 | 來源與授權 |
|---|---|---|
| Live · neural 上聲部 | `paraphernalia_data_satb2` | Beatrice 系模型。底層語料為 **JVS corpus + JVS-MuSiC**（Shinnosuke Takamichi et al.）＝**非商業／學術限定**。⚠ **待確認**：satb2 是官方附帶還是自訓衍生，確認後補這一格（`TRAINING_NOTES.md §8` 只記到 JVS base model） |

引擎二進位檔與模型檔一律 git-ignored，不隨 repo 散布。

---

## 補完清單

- [x] 朋友姓名 → **匿名，對外寫 “a friend”**（2026-08-19）
- [x] 同意紀錄存放 → **Harry 手機的訊息對話**（2026-08-19）
- [ ] satb2 的確切來源（官方附帶或自訓衍生）— 屬引用準確度，不是個資合規
