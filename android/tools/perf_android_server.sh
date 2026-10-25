#!/usr/bin/env bash
# Measures sync timings against the Faith DB Server app on a device (SyncPerformanceTest), then
# puts the server data back exactly as it was.
#
# Preconditions: the server app is logged in and running. Optionally the Faith ID debug kiosk runs
# on the same device, connected to that server; its timings are measured too.
# Usage: tools/perf_android_server.sh <adb-serial> <server-url> [label] [admin-password]
#   e.g. tools/perf_android_server.sh 192.168.45.107:40451 http://192.168.45.107:18080 android-before
# The server app is force-stopped while its files are copied, so any server version restores cleanly.
set -uo pipefail
SERIAL="$1"
SERVER_URL="$2"
LABEL="${3:-android}"
ADMIN_PASSWORD="${4:-church2026}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
source "$HERE/ui_automation.sh" "$SERIAL"

SERVER=com.example.facedbserver
KIOSK_RECEIVER=com.example.facetestbed/com.allstar.faithid.kiosk.debug.DebugObservationReceiver
DATA=/sdcard/faceDB
BACKUP=/sdcard/faceDB.perf-backup
OUT="$ROOT/integration-tests/build/perf"
LOG="$OUT/$LABEL.log"
mkdir -p "$OUT"

server_front() {
  adbs shell am start -n "$SERVER/com.allstar.faithid.serverapp.ui.MainActivity" >/dev/null; sleep 3
  if ui_has "로그인"; then
    ui_tap loginIdInput; ui_type admin; ui_tap loginPasswordInput; ui_type "$ADMIN_PASSWORD"; ui_tap loginButton; sleep 3
  fi
}
wait_text() { local deadline=$(( $(date +%s) + $1 )); while [ "$(date +%s)" -lt "$deadline" ]; do ui_texts | grep -q "$2" && return 0; sleep 1; done; return 1; }
kill_server() { adbs shell am force-stop "$SERVER"; sleep 2; }
start_server() { server_front; ui_tap "현황"; sleep 1; ui_tap startButton; sleep 3; wait_text 20 "실행 중" || echo "[perf] server did not report running"; }
reload_db() { ui_tap "명단"; sleep 1; ui_tap "DB 다시 읽기"; sleep 1; ui_tap "다시 읽기"; }
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

# 1. Back up the server data with the server process gone.
kill_server
adbs shell "rm -rf $BACKUP && cp -r $DATA $BACKUP" || { echo "[perf] backup failed"; exit 1; }
start_server
echo "[perf] server data backed up to $BACKUP"

# 2. Run the measurement, approving its kiosks in the server app.
(cd "$ROOT" && ./gradlew :integration-tests:test --tests '*SyncPerformanceTest*' --rerun \
  -Pfaith.it.perfUrl="$SERVER_URL" -Pfaith.it.adbSerial="$SERIAL" --console=plain > "$LOG" 2>&1; echo "GRADLE_EXIT=$?" >> "$LOG") &
GRADLE_PID=$!
ui_tap "현황"
while kill -0 "$GRADLE_PID" 2>/dev/null; do
  if ui_has "승인 대기"; then ui_tap "대기 중 전체 승인" && sleep 1 && ui_tap "승인"; fi
  sleep 3
done
grep -E "PERF|PASSED|FAILED|GRADLE_EXIT" "$LOG"
[ -f "$OUT/perf-android.json" ] && mv "$OUT/perf-android.json" "$OUT/$LABEL.json"

# 3. The real kiosk downloads the whole (now large) database: reload starts a new epoch.
adbs shell am broadcast -n "$KIOSK_RECEIVER" -a com.allstar.faithid.kiosk.DEBUG_CAMERA --ez paused true >/dev/null 2>&1
adbs shell am broadcast -n "$KIOSK_RECEIVER" -a com.allstar.faithid.kiosk.DEBUG_NO_FACE >/dev/null 2>&1
server_front
T0=$(now_ms)
reload_db
if T1=$(kiosk_log_after "서버 DB 전체 수신" "$T0"); then
  echo "PERF  device: real kiosk full download after reload  $(( T1 - T0 )) ms (includes the reload itself)" | tee -a "$LOG"
  echo "{\"deviceFullDownloadMs\": $(( T1 - T0 ))}" > "$OUT/$LABEL-device-full.json"
fi

# 4. Restore the data exactly and make every kiosk download it again.
kill_server
adbs shell "rm -rf $DATA && cp -r $BACKUP $DATA && rm -rf $BACKUP" || { echo "[perf] restore failed, backup kept at $BACKUP"; exit 1; }
start_server
T0=$(now_ms)
reload_db
if T1=$(kiosk_log_after "서버 DB 전체 수신" "$T0"); then echo "[perf] restored; kiosk re-downloaded the original data in $(( T1 - T0 )) ms"; fi
adbs shell am broadcast -n "$KIOSK_RECEIVER" -a com.allstar.faithid.kiosk.DEBUG_CAMERA --ez paused false >/dev/null 2>&1
ui_tap "현황"
grep -q "GRADLE_EXIT=0" "$LOG"
