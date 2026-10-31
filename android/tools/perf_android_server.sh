#!/usr/bin/env bash
# Measures sync timings against the Faith DB Server app on a device (SyncPerformanceTest).
#
# The real data is never touched: for the run the server's data folder is switched to an empty
# folder, and afterwards back to the original one (the server restarts on the change, and
# "DB 다시 읽기" makes every kiosk download the original data again).
#
# Preconditions: the server app is running. Optionally Faith ID debug kiosks run, connected to that
# server, on the same device and/or others (kiosk-serials, comma-separated); their timings are
# measured too.
# Usage: tools/perf_android_server.sh <adb-serial> <server-url> <label> <admin-password> [kiosk-serials]
#   e.g. tools/perf_android_server.sh 192.168.45.107:40451 http://192.168.45.107:18080 android-before
set -uo pipefail
SERIAL="$1"
SERVER_URL="$2"
LABEL="${3:-android}"
ADMIN_PASSWORD="${4:?server admin password required}"
IFS=, read -ra KIOSK_SERIALS <<< "${5:-$1}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
source "$HERE/ui_automation.sh" "$SERIAL"

SERVER=com.example.facedbserver
adbk() { local serial="$1"; shift; "$ADB" -s "$serial" "$@"; }
KIOSK_RECEIVER=com.example.facetestbed/com.allstar.faithid.kiosk.debug.DebugObservationReceiver
PERF_DIR="/sdcard/faceDB-perf-$LABEL"
OUT="$ROOT/integration-tests/build/perf"
LOG="$OUT/$LABEL.log"
mkdir -p "$OUT"

clear_field() { adbs shell input keyevent KEYCODE_MOVE_END; for _ in $(seq 40); do adbs shell input keyevent KEYCODE_DEL; done; }
server_front() {
  adbs shell am start -n "$SERVER/com.allstar.faithid.serverapp.ui.MainActivity" >/dev/null 2>&1; sleep 3
  if ui_has "로그인"; then
    ui_tap loginIdInput; clear_field; ui_type admin
    ui_tap loginPasswordInput; clear_field; ui_type "$ADMIN_PASSWORD"; ui_tap loginButton; sleep 3
  fi
}
# Settings, status and DB tasks live on the server app's 개발자 화면 (left menu: 상태 / 설정 / 데이터 / …).
dev_open() { ui_tap devScreenButton; sleep 1; ui_tap "${1:-상태}"; sleep 1; }
dev_close() { ui_tap "서버 앱으로"; sleep 1; }
status_line() { ui_texts | grep -m1 "실행 중 ·\|중지됨"; }
wait_text() { local deadline=$(( $(date +%s) + $1 )); while [ "$(date +%s)" -lt "$deadline" ]; do ui_texts | grep -q "$2" && return 0; sleep 1; done; return 1; }
current_data_dir() { ui_texts | sed -n 's/^데이터 폴더: //p' | head -1; }
# Changes the server's data folder in its settings; a running server restarts on that folder.
set_data_dir() {
  server_front; dev_open "상태"
  local current; current=$(current_data_dir)
  [ "$current" = "$1" ] && { dev_close; return 0; }
  ui_tap "설정"; sleep 1
  ui_tap devDataDirInput || { echo "[perf] data folder field not found"; dev_close; return 1; }
  clear_field; ui_type "$1"; sleep 1
  adbs shell input keyevent KEYCODE_BACK; sleep 1 # hide the keyboard so it does not cover the buttons
  ui_has "$1" || { echo "[perf] typed folder did not stick"; ui_tap devRevertButton; dev_close; return 1; }
  ui_tap devSaveButton; sleep 4
  ui_tap "상태"; sleep 1
  { wait_text 30 "데이터 폴더: $1" && wait_text 30 "실행 중 ·"; } || { echo "[perf] server did not switch to $1"; dev_close; return 1; }
  dev_close
}
# Reloads the server database (new epoch) and sets RELOADED_MS to the moment of the confirming tap,
# then brings the kiosks to the front, as in operation.
reload_db() {
  dev_open "데이터"; ui_tap "DB 다시 읽기"; sleep 1; ui_tap "다시 읽기"; RELOADED_MS=$(now_ms); sleep 1; dev_close; ui_tap "키오스크"
  local k; for k in "${KIOSK_SERIALS[@]}"; do
    adbk "$k" shell am start -n com.example.facetestbed/com.allstar.faithid.kiosk.ui.MainActivity >/dev/null 2>&1
  done
}
now_ms() { date +%s%3N; }
offset_ms() { local a d b; a=$(now_ms); d=$(adbk "$1" shell date +%s%3N | tr -d '\r'); b=$(now_ms); echo $(( d - (a + b) / 2 )); }
# Host ms of the first log line of kiosk $1 containing $2 after host ms $3 (waits up to 120 s).
kiosk_log_after() {
  local serial="$1" text="$2" after="$3" off; off=$(offset_ms "$serial")
  for _ in $(seq 120); do
    local hit; hit=$(adbk "$serial" logcat -d -v epoch -s FaithKiosk:I | python3 -c '
import sys
text, after, off = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
for line in sys.stdin:
    if text in line:
        t = int(float(line.split()[0]) * 1000) - off
        if t > after: print(t); break' "$text" "$after" "$off")
    [ -n "$hit" ] && { echo "$hit"; return 0; }
    sleep 1
  done
  return 1
}
kiosk_camera() { # paused true|false; paused also means "no face", so changes apply immediately
  local k; for k in "${KIOSK_SERIALS[@]}"; do
    adbk "$k" shell am broadcast -n "$KIOSK_RECEIVER" -a com.allstar.faithid.kiosk.DEBUG_CAMERA --ez paused "$1" >/dev/null 2>&1
    [ "$1" = true ] && adbk "$k" shell am broadcast -n "$KIOSK_RECEIVER" -a com.allstar.faithid.kiosk.DEBUG_NO_FACE >/dev/null 2>&1
  done
}
# Prints how long after RELOADED_MS each kiosk had received the whole database (prefix $1).
report_full_download() {
  local k t; for k in "${KIOSK_SERIALS[@]}"; do
    local model; model=$(adbk "$k" shell getprop ro.product.model | tr -d '\r')
    if t=$(kiosk_log_after "$k" "서버 DB 전체 수신" "$RELOADED_MS"); then
      echo "$1 $model: $(( t - RELOADED_MS )) ms"
    else
      echo "$1 $model: not received within 120 s"
    fi
  done
}

server_front; dev_open "상태"
ORIGINAL_DIR=$(current_data_dir)
dev_close
[ -n "$ORIGINAL_DIR" ] || { echo "[perf] server is not running"; exit 1; }
echo "[perf] server data folder: $ORIGINAL_DIR; measuring in $PERF_DIR"

# Always back to the original folder, also after a failure or interruption.
GRADLE_PID=""
finish() {
  trap - EXIT INT TERM
  [ -n "$GRADLE_PID" ] && kill "$GRADLE_PID" 2>/dev/null
  if set_data_dir "$ORIGINAL_DIR"; then
    reload_db
    report_full_download "[perf] back on $ORIGINAL_DIR; full download after reload,"
    adbs shell rm -rf "$PERF_DIR"
  else
    echo "[perf] could not switch back: set the data folder to $ORIGINAL_DIR in the server settings"
  fi
  kiosk_camera false
  server_front; dev_open "상태"
  echo "[perf] $(status_line)"
  dev_close
}
trap finish EXIT
trap 'exit 130' INT TERM

adbs shell rm -rf "$PERF_DIR"
set_data_dir "$PERF_DIR" || exit 1
kiosk_camera true

# 1. The measurement, approving its kiosks in the server app.
(cd "$ROOT" && ./gradlew --no-daemon :integration-tests:test --tests '*SyncPerformanceTest*' --rerun \
  -Pfaith.it.perfUrl="$SERVER_URL" -Pfaith.it.adbSerial="$(IFS=,; echo "${KIOSK_SERIALS[*]}")" --console=plain > "$LOG" 2>&1; echo "GRADLE_EXIT=$?" >> "$LOG") &
GRADLE_PID=$!
ui_tap "키오스크"
while kill -0 "$GRADLE_PID" 2>/dev/null; do
  if ui_has "허락 대기"; then ui_tap "허락하기"; fi
  sleep 3
done
GRADLE_PID=""
grep -E "PERF|PASSED|FAILED|GRADLE_EXIT" "$LOG"
[ -f "$OUT/perf-android.json" ] && mv "$OUT/perf-android.json" "$OUT/$LABEL.json"

# 2. The real kiosks download the whole measurement database (reload starts a new epoch), timed
#    from the confirming tap, so the time includes the server reading its database again.
kiosk_camera true
server_front
reload_db
report_full_download "PERF  device: full download after reload," | tee -a "$LOG"
grep -q "GRADLE_EXIT=0" "$LOG"
