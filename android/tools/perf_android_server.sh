#!/usr/bin/env bash
# Measures sync timings against the Faith DB Server app on a device (SyncPerformanceTest).
#
# The real data is never touched: for the run the server's data folder is switched to an empty
# folder, and afterwards back to the original one (the server restarts on the change, and
# "DB 다시 읽기" makes every kiosk download the original data again).
#
# Preconditions: the server app is running. Optionally the Faith ID debug kiosk runs on the same
# device, connected to that server; its timings are measured too.
# Usage: tools/perf_android_server.sh <adb-serial> <server-url> <label> <admin-password>
#   e.g. tools/perf_android_server.sh 192.168.45.107:40451 http://192.168.45.107:18080 android-before
set -uo pipefail
SERIAL="$1"
SERVER_URL="$2"
LABEL="${3:-android}"
ADMIN_PASSWORD="${4:?server admin password required}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
source "$HERE/ui_automation.sh" "$SERIAL"

SERVER=com.example.facedbserver
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
status_line() { ui_texts | grep -m1 "실행 중 ·\|중지됨"; }
wait_text() { local deadline=$(( $(date +%s) + $1 )); while [ "$(date +%s)" -lt "$deadline" ]; do ui_texts | grep -q "$2" && return 0; sleep 1; done; return 1; }
current_data_dir() { status_line | sed -n 's/.*데이터 폴더: //p'; }
# Changes the server's data folder in its settings; a running server restarts on that folder.
set_data_dir() {
  server_front; ui_tap "현황"; sleep 1
  local current; current=$(current_data_dir)
  [ "$current" = "$1" ] && return 0
  ui_tap settingsButton; sleep 2
  ui_tap "$current" || { echo "[perf] data folder field not found"; return 1; }
  clear_field; ui_type "$1"; sleep 1
  adbs shell input keyevent KEYCODE_BACK; sleep 1 # hide the keyboard so it does not cover the buttons
  ui_has "$1" || { echo "[perf] typed folder did not stick"; ui_tap "취소"; return 1; }
  ui_tap "저장"; sleep 4
  wait_text 30 "데이터 폴더: $1" || { echo "[perf] server did not switch to $1"; return 1; }
}
# Reloads the server database (new epoch), then shows the kiosk so it syncs right away.
reload_db() {
  ui_tap "명단"; sleep 1; ui_tap "DB 다시 읽기"; sleep 1; ui_tap "다시 읽기"; sleep 1; ui_tap "현황"
  adbs shell am start -n com.example.facetestbed/com.allstar.faithid.kiosk.ui.MainActivity >/dev/null 2>&1
}
now_ms() { date +%s%3N; }
offset_ms() { local a d b; a=$(now_ms); d=$(adbs shell date +%s%3N | tr -d '\r'); b=$(now_ms); echo $(( d - (a + b) / 2 )); }
# Host ms of the first kiosk log line containing $1 after host ms $2 (waits up to 120 s).
kiosk_log_after() {
  local text="$1" after="$2" off; off=$(offset_ms)
  for _ in $(seq 120); do
    local hit; hit=$(adbs logcat -d -v epoch -s FaithKiosk:I | python3 -c '
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
  adbs shell am broadcast -n "$KIOSK_RECEIVER" -a com.allstar.faithid.kiosk.DEBUG_CAMERA --ez paused "$1" >/dev/null 2>&1
  [ "$1" = true ] && adbs shell am broadcast -n "$KIOSK_RECEIVER" -a com.allstar.faithid.kiosk.DEBUG_NO_FACE >/dev/null 2>&1
}

server_front
ORIGINAL_DIR=$(current_data_dir)
[ -n "$ORIGINAL_DIR" ] || { echo "[perf] server is not running"; exit 1; }
echo "[perf] server data folder: $ORIGINAL_DIR; measuring in $PERF_DIR"

# Always back to the original folder, also after a failure or interruption.
GRADLE_PID=""
finish() {
  trap - EXIT INT TERM
  [ -n "$GRADLE_PID" ] && kill "$GRADLE_PID" 2>/dev/null
  if set_data_dir "$ORIGINAL_DIR"; then
    local t0 t1; t0=$(now_ms)
    reload_db
    if t1=$(kiosk_log_after "서버 DB 전체 수신" "$t0"); then echo "[perf] back on $ORIGINAL_DIR; kiosk re-downloaded it in $(( t1 - t0 )) ms"; else echo "[perf] back on $ORIGINAL_DIR"; fi
    adbs shell rm -rf "$PERF_DIR"
  else
    echo "[perf] could not switch back: set the data folder to $ORIGINAL_DIR in the server settings"
  fi
  kiosk_camera false
  echo "[perf] $(status_line)"
}
trap finish EXIT
trap 'exit 130' INT TERM

adbs shell rm -rf "$PERF_DIR"
set_data_dir "$PERF_DIR" || exit 1
kiosk_camera true

# 1. The measurement, approving its kiosks in the server app.
(cd "$ROOT" && ./gradlew --no-daemon :integration-tests:test --tests '*SyncPerformanceTest*' --rerun \
  -Pfaith.it.perfUrl="$SERVER_URL" -Pfaith.it.adbSerial="$SERIAL" --console=plain > "$LOG" 2>&1; echo "GRADLE_EXIT=$?" >> "$LOG") &
GRADLE_PID=$!
ui_tap "현황"
while kill -0 "$GRADLE_PID" 2>/dev/null; do
  if ui_has "승인 대기"; then ui_tap "대기 중 전체 승인" && sleep 1 && ui_tap "승인"; fi
  sleep 3
done
GRADLE_PID=""
grep -E "PERF|PASSED|FAILED|GRADLE_EXIT" "$LOG"
[ -f "$OUT/perf-android.json" ] && mv "$OUT/perf-android.json" "$OUT/$LABEL.json"

# 2. The real kiosk downloads the whole measurement database (reload starts a new epoch).
kiosk_camera true
server_front
T0=$(now_ms)
reload_db
if T1=$(kiosk_log_after "서버 DB 전체 수신" "$T0"); then
  echo "PERF  device: real kiosk full download after reload  $(( T1 - T0 )) ms (includes the reload itself)" | tee -a "$LOG"
  echo "{\"deviceFullDownloadMs\": $(( T1 - T0 ))}" > "$OUT/$LABEL-device-full.json"
fi
grep -q "GRADLE_EXIT=0" "$LOG"
