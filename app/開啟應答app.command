#!/bin/zsh
# 雙擊就開應答 app。桌面上那顆只是捷徑，真正的內容在這裡（在 repo 裡＝進 git）。
#
# 為什麼是 .command 不是 .app：
#   1) 終端機視窗要留著。runbook 的救場路徑是 Ctrl-C，啟動必看的
#      `[gate] sing model loaded` 也印在這裡。
#   2) 麥克風／攝影機權限跟著 Terminal.app 走。做成新的 .app 會被 macOS
#      當成陌生程式，展場當天會跳權限視窗。

cd "${0:A:h}/.." || exit 1

PATTERN='bank_live|solo_min|respond_shell'
# 直譯器：跟 config.py 同一組環境變數，沒設就用本機預設。
PY=${SOLO_CHOIR_PY_APP:-/opt/anaconda3/envs/vcclient-dev/bin/python}

# runbook 自檢第 1 步：清場。有殘留就先關，不然新舊引擎搶麥克風。
leftovers=(${(f)"$(pgrep -f $PATTERN)"})
if (( ${#leftovers} )); then
  print "⚠ 還有舊的引擎在跑："
  ps -o pid=,command= -p $leftovers[@]
  print ""
  print "按 Enter 關掉它們並繼續開新的，不要的話按 Ctrl-C 離開。"
  read -r _
  kill -INT $leftovers[@] 2>/dev/null
  for i in {1..20}; do
    pgrep -f $PATTERN >/dev/null || break
    sleep 0.5
  done
  stuck=(${(f)"$(pgrep -f $PATTERN)"})
  if (( ${#stuck} )); then
    print "⚠ 關不掉，還剩下："
    ps -o pid=,command= -p $stuck[@]
    print ""
    print "在這個視窗貼上這行強制關掉，再重開一次："
    print "    kill -9 $stuck"
    exit 1
  fi
  print "舊的關好了。"
fi

print "開應答 app… 視窗出來後按 Start。"
print "全螢幕：⌃⌘F　　救場：回到這個視窗按 Ctrl-C"
print ""
exec $PY app/respond_shell.py
