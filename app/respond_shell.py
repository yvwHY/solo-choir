"""respond_shell — pywebview 殼：應答式 live（respond2）的展場/測試介面。

08-02 §I 血訓的正解：校準與狀態只在 console＝站著唱的人看不到。這個殼把
respond2 的 stdout 變成大字狀態（能不能唱、校準指示、句數、等待、警告），
引擎照 CLAUDE.md 架構跑**子行程**（respond2 活在 ddsp venv，殼活在
vcclient-dev；音訊完全在子行程裡，橋只搬文字）。不改 respond2 的行為——
解析的就是人看的那些列印（多的只有「收句→渲染…」一行標記）。

Run (conda env vcclient-dev):  python app/respond_shell.py
"""
from __future__ import annotations
import json
import re
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path

import webview

REPO_ROOT = Path(__file__).resolve().parent.parent
HARMONY = REPO_ROOT / "harmony"
UI_FILE = REPO_ROOT / "ui" / "respond_live.html"
# respond2 的家＝ddsp venv（vcclient-dev 缺 parselmouth，worklog 08-04 §J）
import sys as _sys  # noqa: E402
_sys.path.insert(0, str(REPO_ROOT))
import config  # noqa: E402
ENGINE_PY = config.PY_RESPOND
# bank_live 的家＝6x venv（地圖 runbook 的認證跑法）
BANK_PY = config.PY_ENGINE
# solo_min（神經即時模式，08-14）＝**6x venv**，跟 bank_live 同一個。
# 原本用 conda vcclient-dev，08-14 改掉：嘴部門控要 cv2+mediapipe，那兩個
# 只在 6x venv 裡；往 vcclient-dev 裝會拖著 numpy 版本走（1.23.5→2.x），
# viva 前不動認證環境。實測 6x venv 跑 Beatrice **逐位相同**（同一段輸入
# rms 0.05540 兩邊一致、RTF 0.116 vs 0.122），兩邊都是 Python 3.10.20。
SOLO_PY = BANK_PY
SERVER = REPO_ROOT / "server"
# 08-14 耳裁定案的兩顆嘴：高聲部＝B 線自訓 Soprano-3（bt 3000 步，Harry
# 08-14 判「收在 3000」），低聲部＝六月就在用的 tenor。
SOLO_SOP = config.BEATRICE_MODELS / "paraphernalia_data_00003000"
SOLO_BASS = config.BEATRICE_MODELS / "paraphernalia_data_new25_2k"
SOLO_ALT = config.BEATRICE_MODELS / "paraphernalia_data_satb2"

# stdout → UI 事件。只認人看的那些行；認不得的行照樣進 log 面板。
_PATTERNS = [
    ("cal_quiet", re.compile(r"^1\) Calibration: stay silent")),
    ("cal_sing",  re.compile(r"^2\) Calibration: sing")),
    ("sep",       re.compile(r"separation ([\d.]+) dB")),
    # 校準**成功**時才印的那行（失敗走另一條訊息）。抓下來給後續切換用
    # `--gate` 帶回去＝演出中切進應答模式不必再校準一次（校準要他先安靜
    # 3 秒再唱 5 秒，那在台上做不到）。
    # 08-17 review #16：容許科學記號——安靜鏈路的 gate 可能是 1e-05 級，引擎
    # 改印 %.6g（0.0424 或 1.2e-05 都完整抓到；舊 %.4f 會把小 gate 印成
    # 0.0000，重用時 --gate 0.0＝lv>0 恆真＝樂句永遠不結束）。
    ("calgate",   re.compile(r"→ gate ([\d.eE+-]+) \(")),
    ("key",       re.compile(r"^\[key\] (.+)")),
    ("octave",    re.compile(r"^\[auto-octave\] (.+)")),
    ("live",      re.compile(r"^respond2 live\.")),
    ("listen",    re.compile(r"^listening +([\d.]+)s .*phrase (\d+).*io (\d+)/(\d+)")),
    ("render",    re.compile(r"^  captured → rendering")),
    ("phrase",    re.compile(
        r"^  phrase (\d+) .*response +([\d.]+)s .*wait +([\d.]+)s"
        r".*mute window +([\d.]+)s")),
    ("warn",      re.compile(r"^\s*⚠ (.+)")),
    # 實體鍵的狀態（連上/斷線/退回 UDP）：不上狀態列就等於沒有——站著唱的人
    # 看不到 10.5px 的 log 面板（08-12 審查 A3/B7）。失敗行由引擎加 ⚠ 前綴，
    # 會先被上面的 warn 抓走＝大警告條。
    ("tap",       re.compile(r"^\[tap\] (.+)")),
    ("dump",      re.compile(r"^session dump: (.+?) \(")),
    ("done",      re.compile(r"^total (\d+) phrases \| (.+)")),
    # 即時串流模式（spike_stream，08-05 §M）
    ("stream_on", re.compile(r"^stream on ")),
    ("stream_st", re.compile(
        r"^infer p50 (\d+)ms.*under (\d+) flags (\d+)")),
    # 即時模式 v2＝bank_live（08-12 Harry 裁決換掉 spike_stream）
    # 門控裸奔＝both 模式回授風險，必須上大警告條不是 10.5px log（審查 A4）
    ("warn",      re.compile(r"^⚠? ?\[mouth\] GATE DEAD")),
    # 神經即時模式（solo_min，08-14）。solo_min 那幾行印在 stderr，但
    # Popen 用 stderr=STDOUT 合併，所以照樣進得來＝不必動 solo_min。
    ("solo_on",   re.compile(r"^\[solo_min\] running")),
    ("solo_st",   re.compile(r"^\[solo_min\] xruns (\d+)")),
    ("bank_on",   re.compile(r"^ready \(sample bank")),
    # backlog/dropped 為選配群組：dropped 是「掉音訊」唯一照得到的儀器
    # （F25），原本 regex 不抓＝duo 飽和時 35ms 的洞聽得到、看不到（08-17
    # review #4）。行尾可能還掛 ⚠GATE DEAD / ⚠WORKER DEAD，UI 端另 sniff。
    ("bank_st",   re.compile(
        r"^note (\S+) +cents +([+\-][\d.]+) +mouth +(open|closed) +xrun (\d+)"
        r"(?: +backlog (\d+)(?:/dropped (\d+))?)?")),
]


TAP_PORT = 8766     # tap_listen.TapListener 的埠（respond2 serial 沒插時退回聽這裡）


class Bridge:
    def __init__(self):
        self._win = None
        self._proc: subprocess.Popen | None = None
        self._proc2: subprocess.Popen | None = None   # both 模式的 bank_live
        self._both_pending: dict | None = None        # respond2 校準完才起 bank
        self._tap_seq = 0        # 同韌體協定：序號遞增、收端去重
        self._tap_on = False     # 視窗有 focus 才發 heartbeat（失焦＝鍵斷線）
        self._stopping = False   # SIGINT 已送出（Stop 連點防護）
        self._stop_t = 0.0       # 上次送 SIGINT 的時刻（防護時窗用）
        self._lock = threading.Lock()   # _both_pending 的原子拿取（審查 S2）
        self._mode = None        # 現在在唱的模式（鍵盤切換用）
        self._opts: dict = {}    # Start 時那組裝置/參數，切換時沿用
        self._switch: dict | None = None   # 進行中的切換 {mode, old[], new}
        self._gate: str | None = None      # 本場校準成功的斷句門檻（切換沿用）
        self._octave: int | None = None    # 本場鎖到的八度（respawn 用 --octave 釘住）
        self._key_shift: int | None = None  # 最新的調移（respawn 用 --key-seed 種回）
        self._retired: list = []           # 已 SIGINT 退場、還沒確認死透的引擎
        self._retired_procs: set = set()   # 曾被刻意退場的引擎（exit 事件降級用）
        threading.Thread(target=self._hb_loop, daemon=True).start()
        threading.Thread(target=self._ble_loop, daemon=True).start()

    def _ble_loop(self):
        """BLE 實體鍵接收（08-18 Harry「那就藍牙」）。

        Pico 韌體＝firmware/pico_led_button/main_ble.py：免配對自訂通知服務
        （BLE HID 鍵盤死案＝官方 rp2 韌體沒編配對，見 GRAVEYARD）。收到
        TAP → self.tap()＝跟 UI space **同一條路、同一個流水號**（不會跟
        space 互吞，序號互吞那課見 tap_listen 08-18 註）。斷線自動重掃重連。
        不擋路：bleak 沒裝／藍牙沒權限／沒這顆鍵＝執行緒安靜退場，app 照常。
        """
        try:
            import asyncio

            from bleak import BleakClient, BleakScanner
        except ImportError:
            return
        SVC = "0f9a0001-1e0e-4c7a-9a4e-534f4c4f4348"
        TAP_C = "0f9a0002-1e0e-4c7a-9a4e-534f4c4f4348"

        async def run():
            while True:
                try:
                    dev = await BleakScanner.find_device_by_filter(
                        lambda d, a: SVC in (a.service_uuids or []),
                        timeout=10.0)
                    if dev is None:
                        await asyncio.sleep(3)
                        continue
                    lost = asyncio.Event()
                    async with BleakClient(
                            dev, disconnected_callback=lambda c: lost.set()
                    ) as cl:
                        self._push("tap", {"m": [
                            "connected (BLE key) — button or SPACE ends "
                            "the phrase"]})

                        def _cb(_h, data):
                            if bytes(data).startswith(b"TAP"):
                                self.tap()
                        await cl.start_notify(TAP_C, _cb)
                        await lost.wait()
                    self._push("tap", {"m": ["disconnected (BLE key) — "
                                             "SPACE still works"]})
                except Exception:  # noqa: BLE001 — 藍牙關掉/權限/裝置消失
                    await asyncio.sleep(5)

        try:
            asyncio.new_event_loop().run_until_complete(run())
        except Exception:  # noqa: BLE001 — 整層兜底：BLE 死不拖 app
            pass

    def _tap_send(self, msg):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.sendto(msg, ("127.0.0.1", TAP_PORT))
            s.close()
        except OSError:
            pass                 # 不擋路：發不出去＝當沒這顆鍵

    def _hb_loop(self):
        while True:
            if self._tap_on and self.running():
                self._tap_send(b"HB 0")
            time.sleep(2.0)

    def tap(self):
        """UI space 鍵＝實體鍵：發 TAP（冗餘 3 封，收端以序號去重）。"""
        self._tap_seq += 1
        for _ in range(3):
            self._tap_send(b"TAP %d" % self._tap_seq)
        return True

    def tap_presence(self, on):
        """UI focus/blur → 有無 heartbeat（引擎 6s 沒 HB 自動回純能量斷句）。"""
        self._tap_on = bool(on)
        return True

    def _attach(self, window):
        self._win = window

    def _push(self, kind, payload):
        try:
            if self._win:
                self._win.evaluate_js(
                    "window.respTele&&window.respTele(%s)"
                    % json.dumps({"kind": kind, **payload}, ensure_ascii=False))
        except Exception:  # noqa: BLE001 — 視窗關閉中
            pass

    def _reader(self, proc, who):
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            # 08-17：**退場中／切換中的舊引擎只餵 log，不再驅動 UI**。
            # exit 降級之後，舊引擎在 ≤8s 收場窗裡的 FRAME/狀態行原本照樣
            # 推畫面＝切走 bank 後鏡頭殘影凍住（Harry 實測抓到）；同理，
            # 切換一開始（他點下拉的那一刻）舊引擎就該從畫面上退場——聲音
            # 繼續唱（無縫），但 UI 跟著**他的意圖**走，不是跟著舊引擎走。
            sw0 = self._switch
            if (proc in self._retired_procs
                    or (sw0 and proc in sw0.get("old", []))):
                if not line.startswith("FRAME "):
                    self._push("log", {"line": line})
                continue
            if line.startswith("FRAME "):
                # 08-18 Harry：「UI 鏡頭顯示都拿掉」＝影像不再推 UI。凍結
                # 配置已不帶 --frame-b64，這裡是兜底：任何 FRAME 行（舊版
                # 引擎/手帶旗標）直接丟掉，不進 log、不掃 _PATTERNS。
                continue
            ev = {"line": line}
            kind = "log"
            for name, pat in _PATTERNS:
                m = pat.search(line)
                if m:
                    kind, ev["m"] = name, list(m.groups())
                    break
            if kind == "calgate":
                # 校準成功的門檻記下來（只記成功那條；失敗那條走別的訊息）。
                # 08-17 review #15：只收**現任或切換中**引擎的值——退場中的
                # 舊引擎晚到的行不得覆寫（它可能還活 ≤8s）。
                sw0 = self._switch
                if proc is self._proc or (sw0 and sw0.get("new") is proc):
                    self._gate = ev["m"][0]
            elif kind == "octave":
                # 08-17 review #16：記住鎖到的八度＝之後帶 --gate 重生的引擎
                # 用 --octave 釘住，不再退回「最初 6 tick」估計器（08-02 實測
                # 同素材鎖出 +0 與 +12 兩種＝live 鎖錯就整場都錯）。
                mo = re.search(r"(?:locked|pinned to) ([+-]?\d+)", line)
                if mo and (proc is self._proc
                           or (self._switch or {}).get("new") is proc):
                    self._octave = int(mo.group(1))
            elif kind == "key":
                # 同上：最新調移（seeded/rotation 行的最後一個帶號整數）
                mk = re.findall(r"[+-]\d+", line)
                if mk and (proc is self._proc
                           or (self._switch or {}).get("new") is proc):
                    self._key_shift = int(mk[-1])
            self._push(kind, ev)
            # 切換中的新引擎報到＝退場舊引擎（要在 both_pending 之前處理，
            # 這樣切到 both 時 _proc2 才會掛在新的那顆 respond2 底下）
            self._switch_ready(proc, kind)
            if kind == "live" and proc is self._proc:
                # both 模式：respond2 校準完成開聽了，bank 才進場——早進會把
                # 和聲餵進 respond2 的安靜/唱歌校準窗（plan 08-12 Step 3）。
                # 競態（審查 S2）：先原子拿走 pending 再 spawn；spawn 期間
                # Stop 進來的話，spawn 完立刻補刀。
                # `proc is self._proc`（08-17 review #15）：退場中的舊 respond2
                # 晚到的 live 行不得代領 pending、把 bank 掛錯位子。
                with self._lock:
                    o, self._both_pending = self._both_pending, None
                if o is not None and not self._stopping:
                    self._proc2 = self._spawn_bank(o, boot_kind="log")
                    if self._stopping and self._proc2.poll() is None:
                        self._retire([self._proc2])
        code = proc.wait()
        with self._lock:
            sw = self._switch
            # 新引擎還沒 ready 就死了＝切換作廢、**舊的留著繼續唱**。
            # （不作廢的話 _switch 會永遠卡著，之後每一次按鍵都被當成
            #   「切換進行中」而靜靜忽略＝鍵盤整場失效。）
            dead_new = bool(sw and sw["new"] is proc)
            if dead_new:
                self._switch = None
        if dead_new:
            # 08-17「一動作一動」之後：舊引擎在點擊時已退場＝失敗不是
            # 「staying on」而是全停，switch_fail 讓 UI 還原成 Start。
            self._push("warn", {"line": f"⚠ switch to {sw['mode']} died "
                                        f"({code}) — engines stopped, press "
                                        f"Start", "m": [sw["mode"]]})
            self._push("switch_fail", {"line": "press Start",
                                       "m": [sw["mode"]]})
            with self._lock:
                self._mode = None
        if proc is self._proc:
            self._both_pending = None   # 主引擎死＝both 排程作廢（審查 A3）
        # 審查 S1：exit 帶「誰死的」與「還有沒有活的」，UI 據此決定要不要
        # 把按鈕還原成 Start（另一顆還在唱時還原＝假 ENDED＋孤兒）。
        # 08-17 review #19：**刻意退場**的引擎不發 exit——原本每次成功切換
        # 都會觸發 UI 的「another engine is still running」黃色假警報、佔住
        # warn bar 直到下次 Start（跟 GATE DEAD 同一條 bar＝訓練人忽略它）。
        if proc in self._retired_procs:
            self._push("log", {"line": f"{who} retired ({code})"})
        else:
            self._push("exit", {"line": f"{who} exited ({code})", "code": code,
                                "who": who, "still": self.running()})

    def _spawn_bank(self, o, boot_kind="boot"):
        """bank_live 子行程＝**viva 凍結配置**（08-14 補齊；原本只帶
        --attack 0.06）。

        修的是什麼：08-13 一整天做的東西**全部是預設關的旗標**，所以
        「app 跑的是 v26 之前的行為」——排練用 CLI、上台開 app＝兩台不同
        的機器，這是當時唯一的排練/演出不一致點。每個旗標的來歷：
          （--sing-gate 1 曾在此：27 維嘴/下巴模型，跨場次 acc 0.934、
                         閉嘴與微張誤觸 0%（v34.1）。08-18 隨鏡頭退役移除
                         ——門權威已是 mic 電平，見 --mic-gate / --mouth 0）
          --vl 2         預設 1＝上三部 79-100% 唱同一顆音（四聲部其實是
                         三份齊唱）。改 2 後齊唱 0%、完整三和弦 43-50%，
                         並把各聲部拉出模型破音區（tenor 66-87%→4-9%）
          --tenor 1      多開 tenor 聲部（v31；v42 起領唱嘴＝reflow-male8
                         spk7，**不再是同一顆 bass 嘴**——bass1 在 65-68
                         是壞音）
          --trim bass=+7,tenor=+6  耳朵層平衡（08-16 定案，原為 tenor=+3）。
                         Harry：「--trim 就共鳴問題，bass tenor 可以再大聲
                         一些」。**A 加權**量到 alto 最響，bass 差 4.7dB、
                         tenor 差 4.0dB（等響值），他要更突出 → 各加 2dB。
                         ⚠ 一定要用 A 加權不能用平坦 rms：bass 降八度後
                         兩者差 **13.3dB**（tenor 8.7 / alto 5.3 / sop 1.3）
                         ——平坦 rms 會把 bass 判成「夠大聲」而耳朵聽不到，
                         那正是 --balance 判死的原因（08-13）
          --vib 25       預設 0＝完全沒有顫音，是「修過音」最強的指紋。
                         3-8Hz 佔比 11.6%→19.9%（Harry 本人 22.7%）
          --min-level -44  預設 -100＝音量門關閉。閉嘴 -54dBFS 仍有 43%
                         幀報音高（自相關對音量無感）。**08-19 Harry「音量
                         門檻都拉更高」：-50→-44（+6 dB）**
          --deadzone 0.35  預設 0＝來回跳 32%。0.35：來回 19%、漏真音
                         0/41、commit 延遲 0ms（上限鎖 0.45，見碼內）
          --maxlag 2     輸出延遲棘輪的上限（預設 3）
          --per-part 4   預設 1＝四聲部只有四條線。08-16 Harry 三段盲聽
                         （A=4 條線／B=16 人失諧複製／C=16 人含真人音色）
                         裁「B、C 分不太出來，也比較像一群人」＝**16 人要，
                         真人音色不加分**。C 沒贏是結構性的：MEM_SPK 每聲部
                         只有 1 位存活歌手（12 位耳裁只活 4 位），所以
                         per-part 4 的組成是 領唱＋1 位真人＋2 個失諧複製
                         ＝16 個聲音裡只有 4 個是真人。CPU 已驗：離線台架
                         median 2.7ms／35ms 預算、零超支（F25）
          --mouth 0      **08-18 Harry：「UI 鏡頭顯示都拿掉」＝鏡頭整條
                         退役**。門權威同日已改 mic（見 --mic-gate），畫面
                         一拿掉鏡頭就零功能＝連 mediapipe 執行緒都不起，
                         省 CPU 也少一個故障模式（[mouth] GATE DEAD 類警告
                         從此不會來自鏡頭）。--sing-gate 1 與 --frame-b64 5
                         一併移除（沒有鏡頭就是死旗標）。要回鏡頭版：拿掉
                         --mouth 0、帶回那兩支旗標，UI 端 mouthCam 在
                         08-18 之前版本的 respond_live.html。
          --mem-real 0   **08-16：這是「一下一下的雜音」的元兇。**（08-17 起
                         引擎預設也翻成 0；這裡仍明寫＝凍結配置不靠預設。）
                         原預設 1＝團員可借別顆模型的真人歌手（v39c）。兩份不同錄音
                         的同一個音疊在一起，必然有微小音高差＝聽得到、
                         但頻譜找不到（它不是多出來的成分，是干涉）。
                         定位方式＝把七層處理全關成裸版再逐項加回：
                           per-part 1 + 顫音        → 乾淨
                           per-part 4 + 顫音        → 有
                           per-part 4 全副本相同    → 乾淨
                           per-part 4 只拿掉真人    → **乾淨**
                         音樂上零損失：Harry 同日耳裁 T2(失諧複製) vs
                         T3(含真人音色)「分不太出來」，16 人感靠失諧維持。
                         ⚠ 五支儀器全部照不到它（dropped/xrun/高頻能量/
                         非諧波能量/roughness），判準只有耳朵。
          --pad 0        預設 1.5＝慢層和弦墊開著。08-16 Harry 聽四聲部隔離
                         時抓到「除了本身的聲部音，好像還有一個底層旋律」
                         ——就是它（錨音駐留 1.5s 才換和弦＝自己走出一條沒
                         人唱的線）。量測：pad 在他**唱的時候**只貢獻 +0.8dB、
                         在他**不唱時**貢獻 +5.2dB（不唱時仍 >−40dB 的比例
                         97.3%→53.9%）＝它主要在他停下來時自己撐著。Harry
                         裁「先關掉」。⚠ 關 pad **不會**解決「停唱後 3 秒才
                         安靜」：兩者的衰減曲線幾乎一樣（3s 後 −0.8 vs −1.1dB），
                         那條尾巴另有來源（且離線 --file 無鏡頭＝門控全開，
                         這個數字要 live 才量得準）
        **--vowels 刻意不帶**（＝預設 0，全「啊」）：08-13 下午 Harry 走完
        多母音四版後的最終耳裁是「改回都是啊、多母音整條線不上 viva」。
        當晚那份紀錄裡列的 `--vowels 1 --layers 2,3,4` 自己標著「母音開關
        ＝耳裁項」＝**尚未定案**，所以這裡採用已經定案的那個。要翻案就
        改這裡並在 STATE 記一筆。

        不帶 --gain：過耳版預設 1.0，UI gain 欄是 respond 語意，0.5 硬套會
        偏離認證。
        boot_kind="log"＝both 模式：boot 事件會把 UI 蓋成 LOADING、而 respond2
        的心跳在他開唱時不印＝LOADING 掛滿第一句（審查 A5），所以只進 log。
          --mic-gate 1 --mic-open -18 --bleed -13.5
                         **08-18 Harry：「live 只用麥克風就可以觸發」**＝門
                         的權威從鏡頭改成 mic 電平（bank_live v43）。開門＝
                         hop RMS 贏過 max(-18dBFS, 最近8hop輸出-13.5+6dB)
                         （-26→-24＝08-18 稍後 Harry 耳測 OK 後「可以更嚴格
                         點」；他唱的電平 ≥-22、山谷在 -24~-22＝-24 貼谷底）
                         **08-19 Harry「都拉更高」：-24→-18（+6 dB）。
                         退回＝把這裡與 _spawn_solo 的 --mic-open 改回 -24**
                         連續 2 hop；關門＝低於地板 0.8s。首版帶 -45/-17.5
                         （08-10 CALIB k）上機即翻車：Harry 實測「一直觸發」
                         ＝首場 dump（app_bank_0818_134915）量到門開 88.6%。
                         病因兩條：①房間+回授底噪坐在 -34~-28dBFS，-45 門
                         檻形同虛設；②本機當日實測耦合中位 -13.4dB（p90
                         -7.9），比 CALIB 的 -17.5 漏更多。新值＝拿該場錄音
                         掃參數格選的：門開 15.9%、唱到的 hop 96% 有開、
                         回授誤開 9%（舊值 88/100/88）。（--sing-gate 與
                         --frame-b64 曾短暫留著當純 UI 畫面；08-18 稍後
                         Harry 裁「UI 鏡頭顯示都拿掉」＝隨 --mouth 0 移除，
                         見上。）誠實邊界（F26）：能量域把回授誤判成在
                         唱的死穴仍在＝重開了 08-16「門控不可靠」的判決，
                         生死交 Harry 耳測。
        已知邊界（審查 B10）：bank 固定寫 ch0/1，respond 用 --out-map 分路時
        會疊到 D 聲道——分路演出要用 both 模式前先議聲道配置。"""
        o = o or {}
        cmd = [str(BANK_PY), "bank_live.py",
               "--tenor", "1", "--vl", "2", "--trim", "bass=+7,tenor=+6",
               "--vib", "25", "--deadzone", "0.35", "--maxlag", "2",
               "--min-level", "-44", "--attack", "0.06",
               "--per-part", "4", "--mem-real", "0", "--pad", "0",
               "--view", "0", "--mouth", "0",
               # bank 音量（08-18 Harry：「duo 裡 bank 蓋過 neural」）：UI 的
               # bank 欄，預設 1.0＝過耳認證原值（原本刻意不帶 --gain，帶
               # 1.0 逐位元等價）。⚠ 引擎在 Start/切換時讀旗標＝調完要重按
               # Start 或切換模式才生效，不是即時鈕。空/非數字照 1.0。
               "--gain", str(o["bank_gain"]
                             if o.get("bank_gain") not in (None, "")
                             else 1.0),
               "--mic-gate", "1", "--mic-open", "-18", "--bleed", "-13.5",
               "--dump", "scratchpad/app_bank_" + time.strftime("%m%d_%H%M%S")]
        # 分聲道（08-18 Harry：「neural/live/live+neural 都要能分聲道」）。
        # UI 的 map B·T·A·S ＝ bank_live VOICES 順序（bass,tenor,alto,sop）
        # 原樣傳。四格沒選滿＝bankMapValue 回空＝不帶旗標＝原立體聲 pan。
        _bm = str(o.get("bank_map") or "").strip()
        if _bm:
            cmd += ["--out-map", _bm]
        if o.get("in_name"):
            cmd += ["--in-name", str(o["in_name"])]
        if o.get("out_name"):
            cmd += ["--out-name", str(o["out_name"])]
        p = subprocess.Popen(
            cmd, cwd=str(HARMONY), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        threading.Thread(target=self._reader, args=(p, "bank"),
                         daemon=True).start()
        self._push(boot_kind, {"line": " ".join(cmd[1:])})
        return p

    def _spawn_solo(self, o, boot_kind="boot"):
        """solo_min 子行程（神經即時＝量化天使版，08-18 Harry「更新進 neural」）。

        08-18 晚場整輪耳裁收斂的凍結配置（工程細節見 worklog 08-18 晚場）：
          量化天使（--quantize）：天使唱「他所站音符」的調內絕對目標＝
            和弦有準的錨（舊版＝他的音準複本，他飄全體跟著飄，Harry 裁
            「和諧度不如 ddsp 離線版」的根子）。含純律＋共享調音中心
            （τ≈2.5s 貼他的音準中心）。音符只從站穩 30ms 的音高提交＝
            路過音進不來（「降音天使不動」「極速下滑」兩實案的根治）。
          編制：alto（satb2 spk0 女聲）四度上＋tenor（new25_2k＝他自己的
            嗓）四度下＋乾聲＝三部（08-18 連環裁決；四聲部 sop/bass 舊制
            退役，回舊制＝git 08-18 之前的本函式）。
          --vib 14@5.2,12@4.6：各聲部自有顫音（速率不同＝互相獨立）、
            起音 300ms 漸開＋深度呼吸＝不是合成正弦。
          --wet-send 0.28：合唱濕層（同一個廳＝合唱膠水；乾聲不濕）。
          --voice-delay 25/40ms：起音錯開＝合唱跟主唱的那點拖。
          --blocksize 960（20ms）：兩顆**不同**模型同跑會 cache thrash
            （08-14 耳判 480=滋滋波波）；單模型才可 480。
        門控：mic 電平門（見下方 --mic-gate 註記），不開鏡頭。"""
        o = o or {}
        # key：diatonic 要調名。UI 的 key 欄填 C..B（含 #/b）就用它；
        # auto/空/整數＝先用 C 起跑（logged）。小調要 --minor，尚未拉上 UI。
        # ⚠ 08-18 音程＝Harry 當日連環裁決的終點（八度→三度→四度對稱）；
        # 要改音程只動 --steps/--steps2（quantize 的音符/純律邏輯自動跟）。
        _k = str(o.get("key") or "").strip()
        if not re.fullmatch(r"[A-Ga-g][#b]?", _k):
            _k = "C"
        cmd = [str(SOLO_PY), "solo_min.py",
               "--model", str(SOLO_ALT), "--speaker", "0",
               "--mode", "diatonic", "--key", _k,
               # 08-18 Harry「有高樓但缺屋頂」：上聲部錨 +3（四度）→ +7
               # （八度區）＝free 的和弦音改在你上方八度那帶撿＝真正蓋頂；
               # 太高/太尖就退 +5（六度）。tenor 錨不動。
               "--steps", "7",
               # 08-18 Harry「第三個天使補中間」：中層＝同顆 satb2 女聲、錨
               # +3（四度區＝原上聲部的位置，正好當樓層）；--model2 不帶＝
               # 繼承 --model。三顆嘴兩模型 @960＝08-14 驗過的負載同級。
               "--steps2", "3",
               "--model3", str(SOLO_BASS), "--steps3", "-3",
               "--voices", "3",
               # 08-18 Harry「四度組成不適用大多數歌」→ free 和弦推測＋
               # voice-lead 獨立聲線（**08-18 晚 Harry 耳裁「可以 過」＝認證**，
               # 含四部制＋轉音收斂那組值）：天使配和弦音
               # 而非固定音程；steps ±3 降級為音域錨。台架已驗：和弦音配置
               # 正確、共同音留住、聲線 ≤5 半音、量化 held 是和弦狀態唯一
               # 驅動源（舊判音鏈在 quantize 下已斷＝無雙重驅動）。
               # 回固定四度＝刪掉這兩支旗標。
               "--free", "--voice-lead",
               # 08-18 Harry「轉音有時糊在一起」：換音收利落——滑速 0.4→
               # 0.7（三度 ~43ms 完成）、>2.5st 直接落點、殘響 t60 1.2→0.9、
               # 起音錯開 25/33/40→20/28/36ms。長音判準（穩定/鎖住）不動。
               "--quantize", "1", "--q-glide", "0.7", "--q-snap", "2.5",
               "--vib", "14@5.2,13@4.9,12@4.6",
               "--wet-send", "0.28", "--wet-t60", "0.9",
               "--voice-delay", "0.02,0.028,0.036",
               # 神經聲部音量：UI 的 neural 欄生效；預設 1.0＝08-18 的
               # +5dB 帳（(choir/√2)/you＝(1/1.414)/0.4＝1.77＝+5dB，
               # Harry 裁「兩聲部比我大聲 5dB」）。duo 模式預設再砍半。
               # 08-17 review #17：「非空才收」判法不變。
               "--choir-gain", str(o["ngain"]
                                   if o.get("ngain") not in (None, "")
                                   else 1.0),
               "--gate-floor", "0.02",
               "--blocksize", "960", "--cushion-ms", "20",
               # 門控（08-18 Harry：「neural 也改成不用鏡頭觸發，如同 bank」
               # ＝跟進 bank_live v43 的裁決）：權威從鏡頭改成 mic 電平，
               # solo_min **完全不開鏡頭**（cv2/mediapipe 不載）。開門＝
               # hop RMS 贏過 max(-26dBFS, 最近280ms輸出峰-13.5+6dB) 連續
               # 70ms；關門＝低於地板 0.8s。-26/-13.5 直接沿用 bank 今天在
               # 本機掃出來的值（同一支 mic 同一個房間；bank 首版帶預設
               # -45 上機即翻車「一直觸發」——別走回頭路）。要回鏡頭版：
               # 這行換回 "--mouth-gate 1 --gate-fail open"（08-14 版）。
               # 誠實邊界同 bank（F26）：能量域可能把回授誤判成在唱，生死
               # 交耳測；鏡頭版「講話誤觸 11%」變成「講話一定觸發」——
               # mic 門分不出唱與講，對觀眾講話前先按 Stop 或把 neural 拉 0。
               "--mic-gate", "1", "--mic-open", "-18", "--bleed", "-13.5"]
        # dry 直通：**要留著**（Harry 08-14：拿掉之後「聽不出來和聲，只剩
        # 兩個乾聲」——和聲需要一個參照物才成立）。回音的真兇不是直通本身，
        # 是 `--dry-delay-ms 40`：那個延遲是為了讓直通對齊**轉換後的和聲**，
        # 前提是他戴耳機（空氣裡沒有他）。用喇叭演出時他本人就在現場，直通
        # 該對齊的是**他的真實聲音**⇒ 延遲 0，否則就是 slap-back。
        # 代價：和聲比他慢約 40ms——那本來就是合唱團跟著主唱的樣子。
        # you 欄 0＝完全不直通（戴耳機或走 PA 分軌時才這樣設）。
        _you = float(o.get("you_gain") if o.get("you_gain") not in (None, "")
                     else 0.3)   # 08-18：Harry app 實測「user 聲音有點大」
                                 # 0.4→0.3（天使相對乾聲 +5→+7.4dB）
        cmd += ["--dry-delay-ms", "0"]
        cmd += ["--no-dry"] if _you <= 0 else ["--you-gain", str(_you)]
        # 分聲道（08-18 Harry：「neural/live/live+neural 都要能分聲道」）。
        # UI 的 map D·U·L·S 在 neural 的語意：U=屋頂(+7)、S=中層(+3)、
        # L=tenor(−3)＝solo_min --out-map 的 v1,v2,v3 順序；D=乾聲 --dry-ch。
        # S 留空＝跟 U 同道（respond 同慣例）。沒選滿＝mapValue 回空＝不帶
        # 旗標＝原混音路徑（引擎端逐位元不變）。
        _om = str(o.get("out_map") or "").strip()
        if _om:
            _p = _om.split(",")
            _d, _u, _l = _p[0], _p[1], _p[2]
            _s = _p[3] if len(_p) > 3 else _u
            cmd += ["--out-map", f"{_u},{_s},{_l}", "--dry-ch", _d]
        if o.get("in_name"):
            cmd += ["--in-name", str(o["in_name"])]
        if o.get("out_name"):
            cmd += ["--out-name", str(o["out_name"])]
        p = subprocess.Popen(
            cmd, cwd=str(SERVER), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        threading.Thread(target=self._reader, args=(p, "solo"),
                         daemon=True).start()
        self._push(boot_kind, {"line": " ".join(cmd[1:])})
        return p

    def _spawn_respond(self, o, boot_kind="boot"):
        """respond2 子行程（應答模式的主引擎；both 模式也是它打頭陣）。

        `--gate`：**演出中切進應答模式時，沿用本場已經校準好的門檻。**
        不這樣做的話每次切進來都要重跑校準（安靜 3 秒＋唱 5 秒）——那在台
        上做不到，而且校準期間他必須配合機器而不是唱他的曲子＝那不叫無縫。
        只有本場校準**成功**過才會有值（失敗那條印的是別的訊息，不會被抓
        到）；沒有值就照舊校準。前提是 mic/房間沒變，同一場內成立。
        """
        cmd = [str(ENGINE_PY), "respond2.py", "--live",
               # 08-19 Harry「音量門檻都拉更高」：校準門檻 x2（+6 dB）。
               # 引擎預設 1.0＝原行為，退回就是把這兩個字拿掉。
               "--gate-boost", "2.0",
               "--key", str(o.get("key") or "auto"),
               "--gain", str(o.get("gain") or 0.5),
               "--stab", str(o.get("stab") or 1),
               "--max-min", str(o.get("max_min") or 40)]
        if o.get("gap"):
            # 08-14 Harry：「SING 自動切到 HOLD 的等待時間太快了，不夠唱」。
            # 引擎預設 0.35s（他換氣實測 0.3-0.5s）＝換氣稍長就被當成唱完。
            # 拉到 UI 上讓他自己調，不要寫死一個我猜的值。
            cmd += ["--gap", str(o["gap"])]
        if o.get("gate"):
            cmd += ["--gate", str(o["gate"])]
        elif self._gate:
            cmd += ["--gate", self._gate]
            self._push("log", {"line": f"[switch] reusing calibrated gate "
                                       f"{self._gate} — no recalibration"})
            # 08-17 review #16：帶 --gate 重生＝跳過校準＝原本退回「最初 6
            # 個有聲 tick」的八度估計器（08-02 同素材鎖出 +0 與 +12 兩種；
            # live 鎖錯就整場都錯）、--key auto 也失去校準種子從 0 起跑。
            # 把本場已鎖到的值種回去：--octave 釘八度、--key-seed 種調（引擎
            # 仍逐句修）。只掛在 gate 重用這條——全新校準的 Start 自己會鎖。
            if self._octave is not None:
                cmd += ["--octave", str(self._octave)]
            if (self._key_shift is not None
                    and str(o.get("key") or "auto").lower() == "auto"):
                cmd += ["--key-seed", str(self._key_shift)]
        if o.get("in_name"):
            cmd += ["--in-name", str(o["in_name"])]
        if o.get("out_name"):
            cmd += ["--out-name", str(o["out_name"])]
        if o.get("out_map"):                 # 'D,U,L' → respond2 --out-map（分路）
            cmd += ["--out-map", str(o["out_map"])]
        if o.get("reverb") not in (None, ""):    # 悠遠：殘響送出量（預設 0.20）
            cmd += ["--reverb", str(o["reverb"])]
        if o.get("tail_s") not in (None, ""):    # 悠長：唱完天使自己延幾秒
            cmd += ["--tail-s", str(o["tail_s"])]
        p = subprocess.Popen(
            cmd, cwd=str(HARMONY), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        threading.Thread(target=self._reader, args=(p, "respond"),
                         daemon=True).start()
        self._push(boot_kind, {"line": " ".join(cmd[1:])})
        return p

    def _spawn_mode(self, mode, o, boot_kind="boot"):
        """依模式生引擎，回 (主引擎, [同時起的其他引擎])。

        **Start 與鍵盤切換共用這一條**——08-14 加切換時刻意先抽出來：
        `_spawn_bank` 旗標寫死那件事的教訓就是「同一件事有兩條路徑＝遲早
        變成兩台不同的機器」。
        「主引擎」＝報 ready 的那顆，切換以它為準退場舊的。"""
        if mode == "stream":
            # 即時模式 v2＝bank_live（08-12 Harry 裁決換掉 spike_stream；
            # 舊路線要回來：cmd 換 [ENGINE_PY, "spike_stream.py", "--gain",
            # "0.6", "--block", "0.20", "--crossfade", "0.08", "--extra",
            # "0.7"]＝08-05 §M v14 原樣）。
            return self._spawn_bank(o, boot_kind), []
        if mode == "solo":
            # 神經即時（08-14 Harry 裁決接入：不想讓觀眾看到換 app）
            return self._spawn_solo(o, boot_kind), []
        if mode == "duo":
            # 08-14 Harry：「live neural + live 試試看」＝取樣器四聲部＋神經
            # 兩顆嘴＋他的乾聲同時響。鏡頭兩邊都不開了（08-18：門權威改
            # mic 電平；同日稍後 Harry「UI 鏡頭顯示都拿掉」＝bank 連純畫面
            # 的鏡頭也退役）。主引擎取 bank（它的 ready 比較晚到：要載音
            # 庫），切換退場時兩顆都已在唱。
            # duo 起點＝把神經那半再砍一半（Harry 08-14 實測它比取樣器大
            # 太多）。UI 的 neural 欄仍然生效——這裡只是把它的**預設**在
            # duo 模式下減半，他填了值就照他的。
            od = dict(o)
            # 08-17 review #17：不能用 `or`——UI 送來的是字串，float 之後 0
            # 是 falsy，「把 neural 調到 0」會被 _spawn_solo 的預設復活成 0.5
            # 且不砍半＝靜音請求反轉成雙倍音量（隔壁 you_gain 的慣用法才對）
            od["ngain"] = float(o["ngain"] if o.get("ngain") not in (None, "")
                                else 0.5) * 0.5
            partner = self._spawn_solo(od, "log")
            # 08-18 Harry duo 平衡終判「0.5/0.5」：bank 欄留空時 duo 用 0.5
            # （UI neural 0.5×砍半=0.25 那半由上面既有機制達成）；bank 單獨
            # 跑（mode 2）留空仍是認證的 1.0（_spawn_bank 的 fallback）。
            # 他手動填值＝兩種模式都照他的。
            ob = dict(o)
            if ob.get("bank_gain") in (None, ""):
                ob["bank_gain"] = 0.5
            return self._spawn_bank(ob, boot_kind), [partner]
        return self._spawn_respond(o, boot_kind), []

    def start(self, opts: dict):
        """opts: {mode, gain, stab, key, gap, in_name, out_name, out_map,
        max_min, reverb, tail_s, ngain, bank_gain, you_gain} — 全部有預設；
        誰讀哪幾欄看 _spawn_*（respond/bank/solo 各取所需，其餘欄位忽略）。"""
        if self.running():
            return False
        self._stopping = False
        self._both_pending = None   # 上一場殘留的排程作廢（審查 A3：respond2
        #   校準期就掛→換純應答重開，舊 pending 會讓 bank 憑空冒出）
        o = opts or {}
        # 換了輸入裝置＝上一場校準出來的門檻對這支 mic 不成立，作廢重校。
        # 不作廢的話它會被靜靜沿用整場，而症狀（斷不出句／整首變一句）跟
        # 「換了 mic」看起來毫無關聯。
        if o.get("in_name") != (self._opts or {}).get("in_name"):
            self._gate = None
        self._opts = dict(o)        # 切換時沿用同一組裝置/參數
        self._mode = o.get("mode") or "respond"
        if self._mode == "both":
            # 第三模式（08-12 Harry 裁決）：respond2 先走，_reader 等到
            # 「respond2 live.」（校準完成）才起 bank_live。
            self._both_pending = o
        self._proc, extra = self._spawn_mode(self._mode, o)
        self._proc2 = extra[0] if extra else None
        return True

    def _retire(self, procs):
        """引擎退場（SIGINT）＋確保它終會死透（08-17 review #15）。

        原本切換退場只送一發裸 SIGINT、引用直接丟進區域變數＝引擎 hang 在
        CoreAudio 關流（playbook 已知）時**永遠**抱著 mic/輸出/鏡頭不放，
        stop()/_on_closed 都看不到它——逐字重演 _on_closed 的 docstring 說
        要防的「下一次開演 space 鍵直接是死的」。這裡：記進 _retired registry
        （stop()/_on_closed 會一併收）、標記 _retired_procs（exit 事件降級）、
        並起一條 8s reaper 兜底 SIGKILL（預算與 stop() 相同＝dump 寫得完）。"""
        alive = [p for p in procs if p and p.poll() is None]
        if not alive:
            return
        with self._lock:
            self._retired = [q for q in self._retired
                             if q.poll() is None] + alive
            self._retired_procs.update(alive)
        for p in alive:
            p.send_signal(signal.SIGINT)     # 自己會把 dump 寫完再退

        def _r():
            t0 = time.time()
            while any(p.poll() is None for p in alive) and time.time() - t0 < 8:
                time.sleep(0.25)
            for p in alive:
                if p.poll() is None:
                    self._push("warn", {"line": "⚠ retired engine hung in "
                                        "stream close → killed (dump was "
                                        "written before the close)",
                                        "m": ["kill"]})
                    p.kill()
        threading.Thread(target=_r, daemon=True).start()

    # 模式 → 「這顆引擎真的在唱了」的事件種類。切換就是等這個事件才殺舊的。
    # both 仍可用但 UI 已拿掉（08-14 Harry：「respond+live 模式拿掉」），
    # 保留機制不刪碼＝要回來只是 UI 加一個 option。
    _READY = {"respond": "live", "both": "live",
              "stream": "bank_on", "solo": "solo_on", "duo": "bank_on"}

    def switch_mode(self, mode, opts=None):
        """鍵盤 1-4／滑鼠下拉切換：**點下去舊引擎立刻停**，新引擎載入期間
        安靜（UI 撐 SWITCHING），ready 後開唱。

        08-17 Harry 裁決「一動作一動」：舊設計（先起新的、等 ready 才殺舊
        的＝聲音無縫）在切換期間舊引擎**還在跟著他反應**——樂器不服從動作。
        廢除重疊；切換失敗＝什麼都不唱、回 Start（switch_fail 事件負責把
        UI 還原）。"""
        if mode not in self._READY:
            return False
        if not self.running() or self._stopping:
            return False            # 還沒開演就切＝沒有意義，要按 Start
        with self._lock:
            if self._switch is not None:
                return False        # 切換進行中：連按忽略（不排隊＝不會積
                #                     一串切換在後面自己跑）
            if mode == self._mode:
                return True
            old = [p for p in (self._proc, self._proc2)
                   if p and p.poll() is None]
            self._switch = {"mode": mode, "old": old, "new": None, "extra": []}
        # 切換時吃 UI **當下**的欄位值（neural / gap / you…）。只用 Start 那
        # 一組的話，他調完 neural 再按鍵會發現沒反應——而畫面上那個數字明明
        # 已經改了＝顯示與行為不一致。裝置欄位仍以 Start 那組為準（演出中
        # 換音訊裝置不在這條路徑的守備範圍）。
        o = dict(self._opts or {})
        for k, v in (opts or {}).items():
            if k not in ("mode", "in_name", "out_name", "out_map"):
                o[k] = v
        o["mode"] = mode
        try:
            p, extra = self._spawn_mode(mode, o, boot_kind="log")  # boot 會蓋 LOADING
        except Exception as e:      # noqa: BLE001 — 生不出來＝舊的留著唱
            with self._lock:
                self._switch = None
            self._push("warn", {"line": f"⚠ switch to {mode} failed ({e})",
                                "m": [str(e)]})
            return False
        with self._lock:
            cancelled = self._switch is None    # stop() 在 spawn 期間進來了
            if not cancelled:
                self._switch["new"] = p
                self._switch["extra"] = extra
                old_now = list(self._switch["old"])
                if mode == "both":
                    self._both_pending = o  # ready 之後 bank 才接著上
        if cancelled:
            self._retire([p] + extra)       # 連組合模式的夥伴一起補刀
            #   （_retire 自己拿鎖，所以要在鎖外呼叫）
            return False
        # 08-17「一動作一動」：舊引擎在**點擊當下**退場（寫完 dump 就走），
        # 不再唱到新引擎 ready。載入期間的安靜由 UI 的 SWITCHING 撐著。
        self._retire(old_now)
        self._push("switching", {"line": f"switching to {mode}…",
                                 "m": [mode]})
        threading.Thread(target=self._switch_watchdog, args=(p, mode),
                         daemon=True).start()
        return True

    SWITCH_TIMEOUT = 30.0

    def _switch_watchdog(self, proc, mode):
        """新引擎**活著但一直不報 ready** 的看門狗。

        沒有這個的話：引擎卡住（音庫渲到一半、CoreAudio 開流卡死、鏡頭搶
        不到）⇒ `_switch` 永遠留著 ⇒ 之後每一次按鍵都被當成「切換進行中」
        而靜靜忽略＝**鍵盤整場失效，而且台上沒有任何症狀告訴你為什麼**。
        行程死掉那條路 _reader 有處理，卡住這條只能靠時間。

        08-17 review #3：計時改「**距最後一行輸出** 30s」不是「距啟動 30s」
        ——冷 cache 的 bank 重渲要好幾分鐘，但引擎每 8 音印一行進度＝一直
        有活動；真卡死（沒輸出）維持 30s 就收。t_act 由 _switch_ready 每行
        蓋章。"""
        t0 = time.time()
        while True:
            time.sleep(0.5)
            with self._lock:
                sw = self._switch
                if sw is None or sw["new"] is not proc:
                    return              # 已經 ready 或已被取消
                last = max(t0, sw.get("t_act", t0))
            if time.time() - last >= self.SWITCH_TIMEOUT:
                break
        with self._lock:
            sw = self._switch
            if not sw or sw["new"] is not proc:
                return
            self._switch = None
            victims = [sw["new"]] + sw["extra"]
        self._retire(victims)           # 卡住的新引擎要收掉（含 8s 兜底
        #                                 SIGKILL），否則抱著裝置/鏡頭不放
        self._push("warn", {"line": f"⚠ switch to {mode} timed out "
                                    f"({self.SWITCH_TIMEOUT:g}s with no "
                                    f"output) — engines stopped, press Start",
                            "m": [mode]})
        self._push("switch_fail", {"line": "press Start", "m": [mode]})
        with self._lock:
            self._mode = None

    def _switch_ready(self, proc, kind):
        """新引擎報到＝退場舊引擎。回 True 表示這一拍已經處理掉切換。"""
        with self._lock:
            sw = self._switch
            if sw and sw["new"] is proc:
                sw["t_act"] = time.time()   # 看門狗的活動蓋章（冷 cache 渲庫
                #                             有進度行＝不會被 30s 誤殺）
            if not sw or sw["new"] is not proc or kind != self._READY[sw["mode"]]:
                return False
            self._switch = None
            self._mode = sw["mode"]
            # 組合模式的夥伴要接手到 _proc2，否則它不在 stop()/running() 的
            # 視野裡＝孤兒抱著音訊裝置與鏡頭不放
            self._proc, self._proc2 = proc, (sw["extra"] or [None])[0]
            old = sw["old"]
        self._retire(old)                   # 08-17 起舊引擎在點擊時就退場，
        #                                     這裡只是兜底（_retire 只碰還活
        #                                     著的）；hang 住由 reaper 收
        self._push("switched", {"line": f"now {self._mode}", "m": [self._mode]})
        return True

    def devices(self):
        """列出音訊裝置給 UI 下拉選單（08-05）。用引擎的 venv 問 sounddevice
        ＝跟 respond2 開流看到的是同一張表。"""
        try:
            r = subprocess.run(
                [str(ENGINE_PY), "-c",
                 "import sounddevice as sd, json;"
                 "d = sd.query_devices();"
                 "print(json.dumps({"
                 "'in': [x['name'] for x in d if x['max_input_channels'] > 0],"
                 "'out': [{'name': x['name'], 'ch': x['max_output_channels']}"
                 " for x in d if x['max_output_channels'] > 0]"
                 "}, ensure_ascii=False))"],
                capture_output=True, text=True, timeout=15)
            return json.loads(r.stdout.strip().splitlines()[-1])
        except Exception as e:  # noqa: BLE001 — UI 顯示錯誤即可
            return {"in": [], "out": [], "err": str(e)}

    def stop(self):
        """SIGINT → dump 先落檔（08-04 修過）。CoreAudio 關流掛死（playbook）
        ＝等 8s 還不退就 SIGKILL——dump 已安全，不再需要 lldb 救。

        連點防護（08-12 審查 A7）：第二個 SIGINT 會打進正在寫 dump 的
        sf.write ⇒ 檔案截斷。已經送過就只是等。"""
        self._both_pending = None
        with self._lock:
            sw, self._switch = self._switch, None   # 切換作廢（新的一起殺）
            retired = list(self._retired)   # 08-17 review #15：退場中但可能
            #   還沒死透的引擎也要收——否則它 hang 住＝孤兒抱著裝置
        procs = [p for p in ([self._proc, self._proc2] + retired
                             + ([sw["new"]] + sw["extra"] if sw else []))
                 if p and p.poll() is None]
        self._last_stop_procs = procs       # _on_closed 要等同一份名單
        if not procs:
            return True
        # 連點防護＝時窗而非永久閂：both 模式 stop/spawn 競態若漏出孤兒
        # bank，第二次 Stop（時窗後）還收得到它，不會永遠空轉。
        # 時窗＝reaper 預算 8s＋4s margin（審查 B8：綁死關係，改預算要一起改，
        # 時窗 < 預算＝第二個 SIGINT 打進正在寫 dump 的 sf.write＝檔案截斷）。
        if self._stopping and time.time() - self._stop_t < 8 + 4:
            return True
        self._stopping = True
        self._stop_t = time.time()
        for p in procs:
            p.send_signal(signal.SIGINT)

        def _reap():
            t0 = time.time()
            while any(p.poll() is None for p in procs) and time.time() - t0 < 8:
                time.sleep(0.25)
            for p in procs:
                if p.poll() is None:
                    self._push("warn", {"line": ("⚠ stream close hung (known playbook issue) → forcing shutdown; "
                         "the dump was already written before the close"), "m": ["kill"]})
                    p.kill()
        threading.Thread(target=_reap, daemon=True).start()
        return True

    def running(self):
        sw = self._switch
        return bool((self._proc and self._proc.poll() is None)
                    or (self._proc2 and self._proc2.poll() is None)
                    # 切換中的新引擎也算「還在跑」：漏算的話 UI 會在重疊
                    # 期間把按鈕還原成 Start，而且 Stop 收不到它＝孤兒抱著
                    # 音訊裝置不放（審查 S1 那類的假 ENDED）
                    or (sw and any(q and q.poll() is None
                                   for q in [sw["new"]] + sw["extra"])))

    def _on_closed(self):
        """關窗＝直譯器馬上就要退出，stop() 那條 daemon reaper 會被一起砍掉
        ⇒ 8s 後的 SIGKILL 永遠不會送出 ⇒ 卡在 CoreAudio 關流的引擎變孤兒、
        抱著 :8766 與音訊裝置不放，下一次開演 space 鍵直接是死的（08-12 審查
        S1/A5）。所以這裡同步收屍：SIGINT → 有界等待 → kill。"""
        self.stop()
        # 08-17 review #15：等 stop() 實際瞄準的那份名單（含切換中/退場中
        # 的引擎）——原本只快照 (_proc,_proc2)，關窗時正在切換的引擎會被
        # 漏掉＝SIGINT 到一半直譯器退出、reaper 陪葬＝孤兒。
        procs = [p for p in getattr(self, "_last_stop_procs", []) if p]
        # 8s＝跟 stop() 的 reaper 同一個預算：渲染中關窗時 SIGINT 要等
        # torch 回到 Python、dump 的 sf.write 可能上百 MB——3s 會把正在
        # 寫檔的引擎 SIGKILL＝整場錄音截斷（08-12 驗收 N1）。
        # 兩顆共用同一個 deadline（審查 B7：逐顆各 8s＝最壞 16s 卡 GUI 主
        # 執行緒，期間所有 evaluate_js 全阻塞）。
        deadline = time.time() + 8
        for p in procs:
            try:
                p.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                p.kill()


def main():
    if not UI_FILE.exists():
        raise SystemExit(f"UI file not found: {UI_FILE}")
    if not ENGINE_PY.exists():
        raise SystemExit(f"engine python not found: {ENGINE_PY}")
    bridge = Bridge()
    window = webview.create_window(
        "Solo Choir · 應答", url=str(UI_FILE), width=760, height=640,
        min_size=(560, 480), background_color="#10141a", js_api=bridge)
    bridge._attach(window)
    window.events.closed += bridge._on_closed
    webview.start()


if __name__ == "__main__":
    main()
