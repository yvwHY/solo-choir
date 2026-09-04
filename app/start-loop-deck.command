#!/bin/zsh
# loop_deck launcher. Same shape as the Respond launcher: the terminal window
# stays open, so its messages are visible and its key prompts can be answered.
REPO=${0:A:h}/..
PY=${SOLO_CHOIR_PY_ENGINE:-${SOLO_CHOIR_MODELS:-$REPO/models}/ddsp-svc-6x/venv/bin/python}
cd $REPO || exit 1

# Match only a process that is really running loop_deck under the venv python.
# Matching "loop_deck.py" alone is not enough: any shell whose command line
# contains that string, including the one launching it, would count as a
# leftover and the script would stall at the read below.
PAT="venv/bin/python.*loop_deck\.py"
echo "-- clearing --"
LEFT=$(pgrep -fl "$PAT")
if [ -n "$LEFT" ]; then
  echo "$LEFT"
  echo "Leftovers found. Press Enter to send kill -INT, or Ctrl-C to cancel."; read
  pkill -INT -f "$PAT"; sleep 1
  pgrep -f "$PAT" >/dev/null && echo "! Would not close: kill -9 $(pgrep -f "$PAT")"
else
  echo "Clear."
fi
echo "-- devices --"
LIST=$($PY app/loop_deck.py --list)
echo "$LIST"
echo ""

# Output devices can vanish: when a member of an aggregate device is unplugged
# the aggregate stays in the list but its output channel count drops to zero,
# and naming it directly then fails to open. So pick, in order, the first
# device that really has output channels.
PICK=""
# Never auto-select a device whose name contains bh: those route the bed back
# into BlackHole, which records itself. To use one, export OUT=speaker+bh.
for WANT in "$OUT" speaker_set headphone_set MacBook; do
  [ -z "$WANT" ] && continue
  if echo "$LIST" | awk -F'|' '{gsub(/ /,"",$4); if ($4 ~ /^out[1-9]/) print $2}' \
       | grep -qi -- "$WANT"; then
    PICK="$WANT"; break
  fi
  echo "  skipping $WANT (no output channels)"
done
if [ -z "$PICK" ]; then
  echo "x No device in the list can make sound. Reconnect the speakers in Audio MIDI Setup first."
  echo "Press Enter to close."; read; exit 1
fi

echo "-- opening --  output = $PICK (to choose another, export OUT=name first)"
exec $PY app/loop_deck.py --out-dev "$PICK"
