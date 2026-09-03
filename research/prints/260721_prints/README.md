# 260721_prints — WO-2／WO-3 改版＋黃銅通道統一縮孔 後的全批重印

2026-07-21 自 Fusion `solo_choir_frame_a` 匯出，**12 件、4 盤**。
**本批取代 260719_prints 全部。** 設計變更的內容、數字與驗證見同層 `../260719_prints/WO2_WO3_STATUS_260721.md`。

## 件號 ↔ body（延續 260719 README 的對照表）

| 件號 | 檔 | 來源 body | 本批狀態 |
|---|---|---|---|
| WO-2 | `band_clamp.stl` | `band_clamp` | **改**：髮箍改 C 口嵌入、⌀3.1 隧道、中柱道 4.3→3.1 |
| WO-3 | `door_col_A.stl` `door_col_R.stl` | `door_col_A/R` | **整件重建**：去縫去耳，M3 止付各兩顆，軌孔 3.8→3.2 |
| WO-3 | `door_clip_A1/A2/R.stl` | 同名 body | 未變更 |
| WO-3 | `door_panel_flat.stl` | `door_panel_flat` | 未變更 |
| WO-1 | `pivot_mount.stl` | `pivot_mount` | **改**：直孔 3.8→3.1 |
| WO-4 | `wo4_cup.stl` `wo4_cup_R.stl` | 元件 `wo4_bce` | **取代舊的 `bce_mount_L/R`**；黃銅孔 3.6→3.2 |
| WO-5 | `wo5_shell_v10.stl` `wo5_lid_v10.stl` | 同名 body | **改**：J 鉤座 3.3→3.1（改在草圖源頭） |

> **版本選擇（2026-07-21 Harry 確認）**：
> 1. **`bce_mount_L/R` 不印了**，由 `wo4_cup`/`wo4_cup_R` 取代。
> 2. **新杯不搬上骨架定位**——CAD 內維持世界軸正交姿態，位置在實體組裝時處理，不再是待辦。（`+X` 唇側圓角仍未收，見 `WO4_REBUILD_STATUS_260720.md`。）
> 3. wo5 用 **v10 配對**（`wo5_shell_v10` + `wo5_lid_v10`），不是 260719 那組較小的舊版。
>
> **`jhook_west` 是西側 J 鉤、必須跟殼一起印**（第一版 P1 漏掉、只有一個鉤，Harry 在 gcode 預覽抓到）：殼 body 內含**東側**鉤（座在 x −284.5…−235），`jhook_west` 是**西側**鉤（座在 x −392…−342.5），兩者 `measureMinimumDistance = 0.0mm` 相接＝同一個實體件。P1 盤因此**同時載入兩個 body 並以群組落床／置中**，不可各自 drop（會破壞相對位置）。

## 擺向

`pack.py` 只做 Z 降到 0 ＋ XY shelf 排版（必要時繞 Z 轉 90°），**唯一的例外是 door_col**：

- **`door_col_A` / `door_col_R`：`rotate_x(+30°)` 立正**——這兩件在 CAD 裡整體傾 30°，轉正後筒身軸垂直床面（260711 README A 表的規定）。重建後是純圓柱，立起來幾乎沒懸空；代價是黃銅軌孔沿 Z、印完要 ream。
- 其餘件維持 CAD 原姿態落床（與 260719 相同）。
- `door_panel_flat` 繞 Z 轉 **45°**——原件 184mm > 180 床寬，斜放後 165.5×165.5 才進得去。

## 盤面

| 盤 | 內容 | 時間 | PLA | profile |
|---|---|---|---|---|
| `P1_shell` | `wo5_shell_v10` ＋ `jhook_west`（獨佔一盤，兩 body 同組） | **4h 31m 41s** | 97.96g | `miniis_shell_gridbrim.ini`（grid 支撐＋5mm brim） |
| `P2_frame9` | band_clamp · door_col_A · door_col_R · pivot_mount · door_clip_A1/A2/R · wo4_cup · wo4_cup_R（9 件） | **3h 20m 12s** | 38.06g | `miniis_profile.ini` |
| `P3_door_panel` | door_panel @45° | **57m 47s** | 27.80g | `miniis_profile.ini` |
| `P4_lid` | `wo5_lid_v10` | **1h 04m 05s** | 28.42g | `miniis_profile.ini` |

**合計 9h 54m、192.24g**，每盤皆 < lab 5h 上限（最長 P1 4h32m）。
plate STL 在 `plates/`，gcode 與 ini 在 `gcode/`。改件後重跑 `python3 pack.py all` 再切即可。

對照 260719：P3 的時間與用料**與上一批完全相同**（27.80g／57m47s）——該件未變更，可當作整條工具鏈（匯出→打包→切片）沒有走樣的驗證。P1／P4 變重是因為改用 v10 殼與 v10 蓋（比舊版大）。

## 追加盤 P5 — voice unit v1 面板 ×3（2026-07-21 晚）

第一個 voice unit 的 DML 面板，**⌀3 C 形夾直接卡在骨架側軌**（x = ±110 那條，`door_col_A/R` 也夾在同一條上；門柱佔 z −72 以下，z −20…−65 這段是空的）。

| 項目 | 值 | 理由 |
|---|---|---|
| 板 | 60 × 60 × 2.5 | DML 平板；三顆 exciter（DAEX25／FAUOSWUK 35／Dioche 50）都貼得下 |
| exciter 定位凹點 | **板正中 (30,30)**，深 0.3 | Harry 指定。⚠ DML 慣例是偏心（0.4L×0.4W）以避開對稱模態，方板尤其容易簡併——但**exciter 是用黏的、不開孔**，凹點只是參考，實測覺得空就撕下來往 (24,24) 挪 |
| C 形夾 | ⌀3.1 座、口 2.6、壁 2mm、**寬 12mm** ×2、間距 30mm | 座徑與口寬沿用 WO-2 C 口驗過的數字（⌀3 桿要撐 0.4 才進得去） |
| 細頸 | 6 × 1.5mm | **①聲學隔離**：不讓板的振動灌進黃銅骨架，否則多個聲源會透過骨架串音、打掉 F11 的空間分離 **②吃曲率**：讓兩個夾各自撓一點，適應彎軌 |
| 桿軸高度 | z = 3.55（環底與板底齊平） | **列印必要**：原本軸在板厚中心，環會上下各凸出 2.3mm，平放時整片板被墊高懸空、整面要支撐（支撐痕會留在發聲面上） |

- 單件 **9814.8 mm³ ≈ 12.2 g**；`vu1_panel.stl`、`plates/plate_P5_vu1x3.stl`
- **用 `gcode/P5_vu1x3_nosupport.gcode`（1h 19m 07s／33.30 g／零支撐）**。有支撐版 `P5_vu1x3.gcode`（1h 27m／34.16 g／65 段）留著對照——支撐會長進 ⌀3.1 孔與喉口，難清又傷配合面；橫孔頂部的垂邊由既定的 ream 步驟處理。
- **這盤要回答的問題**：兩個分開的胸前聲源（門板 ＋ 這片 unit）在身上到底分不分得開。印三片是為了能同時試不同位置／不同 exciter。

## 建議排機（2 台，lab 5h 上限）

- **A 機**：P1 殼（4h32）
- **B 機**：P2 框架9（3h20）→ P4 蓋（1h04）＝ ~4h24
- P3 門板（58m）插在任一台空檔

## 出檔前的自我檢查（因為漏過一次）

匯出／排盤前掃一次「**列印件是否與其他可見 body 距離為 0**」——相接就代表它們是同一個實體件、必須一起印。本批掃描結果：`pivot_mount` 碰到 `wo5_PREVIEW_shell`（展示用副本）、`wo5_shell_v10` 碰到 6 個電子件 mock-up（TP4056／PAM×2／MT3608／LiPo／Pico），都不是列印件；唯一真的漏掉的就是 `jhook_west`。
另外注意：**Fusion 的 STL 匯出會跳過隱藏 body**，`execute()` 仍回傳 True 但不產生檔案（`door_panel_flat` 就中過）——出檔後一定要對檔案清單與檔案大小。

## 印後驗收

- [ ] **⌀3 黃銅棒**：`band_clamp` 髮箍隧道與兩條中柱道、`pivot_mount` 直孔、`door_col_A/R` 軌孔、`wo4_cup` 黃銅孔、J 鉤座——全部**縮孔後應為「順、但不鬆」**（CAD 餘隙：band_clamp／pivot／J 鉤 0.05mm 單邊；door_col／wo4_cup 0.1mm 單邊）。**印後一律先用 ⌀3 鑽頭或線材本體 ream 一次**。
- [ ] **WO-2 C 口**：髮箍從**頭皮側（−Y）水平推入**卡進去，喉口 2.6mm 比 ⌀3 窄 0.4；裝上後髮箍**不外露**（上方 2.5mm 壁蓋住）。
- [ ] **止付固定實測（本批重點，尚未驗過）**：`door_col_A/R` 各兩顆、`wo4_cup` 各一顆 M3 止付，**PLA 直攻、牙深 4.1mm**——鎖緊看會不會滑牙／撐裂。滑牙 → 下一版考慮埋銅螺母或加厚牙孔壁。
- [ ] `door_clip` 扣 ⌀8 筒身：門閂側單指可開、鉸鏈側緊（本批 clip 未變更，但 door_col 已重建，配合要重驗）。
- [ ] wo5：滑蓋全程平行推到底；J 鉤掛 ⌀3 桿掛持力（座已縮到 ⌀3.1，應比上批緊）。

*記錄：Claude Code 會話 2026-07-21。*
