# WO4 BCE mount 重建現況（2026-07-20 收工）

Fusion 文件 `solo_choir_frame_a`，元件 `wo4_bce`。**全部世界軸對齊、參數化、不對骨架斜角**（Harry 定調：WO4 就是兩個方塊，別搞複雜）。

## 完成的兩件（並排在骨架右邊 +X 空區）
| body | 世界 X 範圍 | vol |
|---|---|---|
| `wo4_cup`（L） | 341.6–352.9mm | 3931.3mm³ |
| `wo4_cup_R`（R，鏡像已倒角的 L） | 360.0–371.3mm | 3931.4mm³（差 0.1 純鏡像數值） |

每件內容：
- **杯**：外 X11.3×Y25×Z17.3；壁 2.7；exciter 腔 8.6×22.3×12.93（兩相鄰牆、開 +X/+Y）；底 2.6、頂蓋 1.77
- **collar**：10×14×12，**貼齊杯 −X 側**（Harry 要的邊）
- **新鎖孔**（取代夾環）：⌀3.6 黃銅通孔(沿 Y) ＋ ⌀2.8 M3 止付(頂面向下、頂黃銅)。**鎖骨架＝黃銅孔+止付，不需另開孔**（同 R，Harry 確認）
- **上下唇**：+X 開口面，凸出 0.5mm × 厚 1.0mm × 沿整寬（擋 exciter）
- **Harry 的 `OffsetFaces1`**：磨平杯內原本的凸出（−0.56mm）。**別刪別動**
- **R0.5 圓角**：主體凸外緣 12 條 ＋ 開口 rim 2 條

## 明天可接（都是刻意延後的）
1. **+X 唇側完整圓角**——rim 邊與唇邊共用尖角頂點，「只倒 rim 留唇尖」倒不乾淨（ASM_BL_NO_VTX_GEOM）。今天 Harry 選維持現狀。要做就得連唇一起倒（唇變圓角導入）或唇用小半徑 R0.2。
2. ~~**搬上骨架定位**（今天刻意沒對斜角）~~ → **2026-07-21 取消，不做**。Harry 決定 CAD 內維持世界軸正交姿態，位置在實體組裝時處理；`bce_mount_L/R` 一併不再列印，由 `wo4_cup`/`wo4_cup_R` 取代（已在 `260721_prints` 出檔切片）。
3. ~~**WO-3** 套同一雙中柱固定法~~ → **2026-07-21 完成**，但改成整件重建而非就地改；見 `WO2_WO3_STATUS_260721.md`。注意 WO-3 = 門那組 `door_col_A/R`，件號對照見 `README.md`。

## Gotchas（給明天的自己）
- **截圖工具渲不出 `wo4_cup`**（多次刪特徵後顯示 mesh 沒重算）——BRep 完全有效，用 `pointContainment` 探測驗證即可；Harry 的即時 UI 正常。
- 參數：`bce_wall/floor/cap/cav_w/cav_d/cav_h/outer_w/outer_d`、`bce_col_w/d/h/embed`、`bce_brass_dia/grub_dia`、`bce_lip_out/thk`。注意：**collar 位置與孔位是草圖字面座標移的、沒接參數驅動**，微調要改草圖或重建。
- 原件 `bce_mount_L/R`（斜貼骨架的舊版）、`bce_flat`(scratch de-skew 件，已隱藏)、`bce_new`(舊 scratch) **都沒刪**，留給 Harry 處置。

*記錄：Claude Code 會話 2026-07-20。*
