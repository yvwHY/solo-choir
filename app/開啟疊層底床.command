#!/bin/zsh
# loop_deck 啟動器（跟「開啟應答app」同款：留終端機視窗＝看得到訊息、按得到鍵）
REPO=${0:A:h}/..
PY=${SOLO_CHOIR_PY_ENGINE:-${SOLO_CHOIR_MODELS:-$REPO/models}/ddsp-svc-6x/venv/bin/python}
cd $REPO || exit 1

# 只認「真的用 venv 的 python 在跑 loop_deck」的行程。
# 不能只比對 "loop_deck.py"：任何指令列裡出現這個字串的 shell（包括別人
# 用來啟動它的那一行）都會被算成殘留，開場就卡在 read 等 Enter。
PAT="venv/bin/python.*loop_deck\.py"
echo "── 清場 ──"
LEFT=$(pgrep -fl "$PAT")
if [ -n "$LEFT" ]; then
  echo "$LEFT"
  echo "有殘留。按 Enter 送 kill -INT，或 Ctrl-C 取消。"; read
  pkill -INT -f "$PAT"; sleep 1
  pgrep -f "$PAT" >/dev/null && echo "⚠ 關不掉：kill -9 $(pgrep -f "$PAT")"
else
  echo "乾淨。"
fi
echo "── 裝置 ──"
LIST=$($PY app/loop_deck.py --list)
echo "$LIST"
echo ""

# 輸出裝置會消失：聚合裝置的成員被拔掉之後，它還在清單上但 out 變成 0。
# 直接指名就會開不起來。所以按順序挑第一個「真的有輸出通道」的。
PICK=""
# 名字裡有 bh 的一律不自動選：那些會把底床送回 BlackHole＝自己錄自己。
# 真的要用就自己 export OUT=speaker+bh。
for WANT in "$OUT" speaker_set headphone_set MacBook; do
  [ -z "$WANT" ] && continue
  if echo "$LIST" | awk -F'|' '{gsub(/ /,"",$4); if ($4 ~ /^out[1-9]/) print $2}' \
       | grep -qi -- "$WANT"; then
    PICK="$WANT"; break
  fi
  echo "  跳過 $WANT（沒有輸出通道）"
done
if [ -z "$PICK" ]; then
  echo "⛔ 清單裡沒有任何能出聲的裝置。先去「音訊 MIDI 設定」把喇叭接回來。"
  echo "按 Enter 關掉。"; read; exit 1
fi

echo "── 開 ──  輸出＝$PICK（要指定就先 export OUT=別的名字）"
exec $PY app/loop_deck.py --out-dev "$PICK"
