# 260719_prints — gcode（2026-07-20 切片）

12 個可印件排成 **4 盤**，每盤 ≤ lab 5h 上限。切片 profile 直接沿用
`260717_prints/gcode/P1_PLA_wo5_shell.gcode` 內嵌設定（Prusa **MINI**，180×180 床，
`0.20mm STRUCTURAL @MINIIS 0.4`，Generic PLA，40% grid，organic 支撐、buildplate-only）。
**未含變體**（wo5_shell_nohook / wo5_lid_nocradle 未印）。

| 檔 | 內容 | 時間 | PLA |
|---|---|---|---|
| `P1_PLA_wo5_shell.gcode` | wo5_shell（獨佔一盤）**support 已修：organic→grid＋5mm brim** | 3h15m | 66.6g |
| `P2_PLA_frame9.gcode` | band_clamp · bce_mount_L · bce_mount_R · door_clip_A1 · door_clip_A2 · door_clip_R · door_col_A · door_col_R · pivot_mount（9 件同盤） | 3h25m | 35.7g |
| `P3_PLA_door_panel_rot45.gcode` | door_panel（PLA 對孔試印；**Z 轉 45°斜放**才塞得進 180 床——原件 184mm 長 > 床寬） | 58m | 27.8g |
| `P4_PLA_wo5_lid.gcode` | wo5_lid | 42m | 15.4g |

**總計 145.5g，約 8h19m 列印。**

### P1 殼 support 修正（2026-07-20）
原 organic 版 **63% 都是 support**（替 pivot window／出線口那些高處小懸空從床上蓋細高塔到
z31，又在 8711mm² 盒底下軟腳）→ 會脫料印壞。原因＝盒底整片懸空 5.5mm（鉤子從盒底往下
垂去勾 yoke，盒身靠鉤尖撐著），這是幾何本質、翻面更糟（開口朝下盒底變懸空屋頂會垮）。
**改法（`miniis_shell_gridbrim.ini`）**：`support_material_style` organic→**grid**（懸空僅
5.5mm 高，短柱穩、好剝）＋`brim_width` 0→**5mm**（救鉤尖第一層附著）。`buildplate_only`
保留（內腔不塞 support）。結果 support 42248→5000mm（63%→17%），只撐 z0–5.6 的盒底＋鉤子；
出線口走 bridge、pivot window 頂邊剩幾條 overhang perimeter（餘隙孔，可接受）。時間 4h41→3h15。
若要 pivot window 頂邊更完美：GUI 加一個局部 support enforcer，或關 buildplate_only（會多內腔支撐）。

## 建議排機（2 台）
- **A 機**：P1 殼（4h41）→ P3 門板（58m）＝ ~5h39
- **B 機**：P2 框架9（3h25）→ P4 蓋（42m）＝ ~4h07

3 台則各挑一盤、P2+P4 併一台，最慢 ~4h41（殼）收工。

## 重切
`pack.py`（session scratchpad）＝把 9 個框架件逐件落床（Z→0）+ shelf 排版成單一
`plate_frame9.stl`；door_panel 繞 Z 轉 45°。改件後重跑該腳本再切即可。
（PrusaSlicer 2.9.6 CLI 的 `--merge` 內建排版在此 GUI build 會 segfault，故自排。）
