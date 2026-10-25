#!/usr/bin/env bash
# Small adb/uiautomator helpers for the E2E scripts. Source with: source ui_automation.sh <serial>
SERIAL=${1:-${ANDROID_SERIAL:-}}
ADB="${ADB:-$HOME/Android/Sdk/platform-tools/adb}"
adbs() { "$ADB" -s "$SERIAL" "$@"; }
ui_dump() { adbs shell uiautomator dump /sdcard/faith_ui.xml >/dev/null 2>&1; adbs shell cat /sdcard/faith_ui.xml; }
ui_texts() { ui_dump | grep -o 'text="[^"]\+"' | sed 's/^text="//; s/"$//'; }
ui_has() { ui_dump | grep -q "text=\"$1\""; }
# Taps the n-th element whose text, content-desc or resource id equals $1.
ui_tap() {
  local query="$1" nth="${2:-1}" bounds
  bounds=$(ui_dump | python3 -c '
import re, sys
xml, q, n = sys.stdin.read(), sys.argv[1], int(sys.argv[2])
hits = []
for node in re.findall(r"<node [^>]*>", xml):
    a = dict(re.findall(r"([\w-]+)=\"([^\"]*)\"", node))
    if q in (a.get("text"), a.get("content-desc")) or a.get("resource-id", "").endswith(":id/" + q):
        hits.append(a["bounds"])
print(hits[n - 1] if len(hits) >= n else "")' "$query" "$nth")
  [ -z "$bounds" ] && return 1
  set -- $(echo "$bounds" | tr '[],' '   ')
  adbs shell input tap $(( ($1 + $3) / 2 )) $(( ($2 + $4) / 2 ))
}
ui_tap_class() {
  local cls="$1" nth="${2:-1}" bounds
  bounds=$(ui_dump | python3 -c '
import re, sys
xml, c, n = sys.stdin.read(), sys.argv[1], int(sys.argv[2])
hits = [dict(re.findall(r"([\w-]+)=\"([^\"]*)\"", nd))["bounds"] for nd in re.findall(r"<node [^>]*>", xml) if "class=\"" + c + "\"" in nd]
print(hits[n - 1] if len(hits) >= n else "")' "$cls" "$nth")
  [ -z "$bounds" ] && return 1
  set -- $(echo "$bounds" | tr '[],' '   ')
  adbs shell input tap $(( ($1 + $3) / 2 )) $(( ($2 + $4) / 2 ))
}
ui_type() { adbs shell input text "$1"; }
ui_shot() { adbs exec-out screencap -p > "$1"; }
