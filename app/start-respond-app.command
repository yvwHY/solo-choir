#!/bin/zsh
# Double-click to open the Respond app. The icon on the desktop is only a
# shortcut; the real file lives here, inside the repository, so it is in git.
#
# Why this is a .command and not a .app:
#   1) The terminal window has to stay open. Ctrl-C in this window is the
#      recovery path in the runbook, and the start-up line you must see,
#      `[gate] sing model loaded`, is printed here.
#   2) Microphone and camera permission follow Terminal.app. A new .app would
#      be treated by macOS as an unknown program and would raise a permission
#      dialog on the day of the show.

cd "${0:A:h}/.." || exit 1

PATTERN='bank_live|solo_min|respond_shell'
# Interpreter: the same environment variables config.py uses, with a local
# default when nothing is set.
PY=${SOLO_CHOIR_PY_APP:-/opt/anaconda3/envs/vcclient-dev/bin/python}

# Step 1 of the runbook self-check: clear the field. Anything left running
# would fight the new engine for the microphone.
leftovers=(${(f)"$(pgrep -f $PATTERN)"})
if (( ${#leftovers} )); then
  print "! An old engine is still running:"
  ps -o pid=,command= -p $leftovers[@]
  print ""
  print "Press Enter to stop them and start fresh, or Ctrl-C to leave."
  read -r _
  kill -INT $leftovers[@] 2>/dev/null
  for i in {1..20}; do
    pgrep -f $PATTERN >/dev/null || break
    sleep 0.5
  done
  stuck=(${(f)"$(pgrep -f $PATTERN)"})
  if (( ${#stuck} )); then
    print "! Could not stop these:"
    ps -o pid=,command= -p $stuck[@]
    print ""
    print "Paste this line in this window to force them closed, then run again:"
    print "    kill -9 $stuck"
    exit 1
  fi
  print "Old engines stopped."
fi

print "Opening the Respond app. Press Start when the window appears."
print "Full screen: Ctrl-Cmd-F     Recovery: come back to this window and press Ctrl-C"
print ""
exec $PY app/respond_shell.py
