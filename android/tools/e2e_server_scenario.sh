#!/usr/bin/env bash
# Runs RemoteServerScenarioTest (6 simulated kiosks) against the Faith DB Server app on a device.
#
# Preconditions: the server app is installed, logged in and the server is running.
# Usage: tools/e2e_server_scenario.sh <adb-serial> [server-url]
#   server-url defaults to http://127.0.0.1:18080 through adb forward; pass the tablet's Wi-Fi
#   address (http://192.168.x.y:18080) to test over the real network.
set -euo pipefail
SERIAL="$1"
SERVER_URL="${2:-http://127.0.0.1:18080}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
source "$HERE/ui_automation.sh" "$SERIAL"

adbs forward tcp:18080 tcp:18080 >/dev/null
# Bring the server app to the front on its 키오스크 tab, where waiting kiosks are approved.
adbs shell am start -n com.example.facedbserver/com.allstar.faithid.serverapp.ui.MainActivity >/dev/null 2>&1; sleep 3
ui_tap "서버 앱으로" || true; ui_tap "키오스크" || true; sleep 1
LOG="$ROOT/integration-tests/build/remote-scenario.log"
mkdir -p "$(dirname "$LOG")"
rm -f "$ROOT/integration-tests/build/remote-scenario.properties"
(cd "$ROOT" && ./gradlew :integration-tests:test --tests '*RemoteServerScenarioTest*' --rerun \
  -Pfaith.it.serverUrl=$SERVER_URL --console=plain > "$LOG" 2>&1; echo "GRADLE_EXIT=$?" >> "$LOG") &
GRADLE_PID=$!

# Act as the administrator: approve kiosks whenever some are waiting.
while kill -0 "$GRADLE_PID" 2>/dev/null; do
  if ui_has "허락 대기"; then
    ui_tap "허락하기" && echo "[e2e] approved a waiting kiosk"
  fi
  sleep 3
done
grep -E "REMOTE:|PASSED|FAILED|GRADLE_EXIT" "$LOG" || true
grep -q "GRADLE_EXIT=0" "$LOG" || { echo "[e2e] scenario failed, see $LOG"; exit 1; }

# Check-ins: every event of this run must appear exactly once in today's attendance sheet.
PROPS="$ROOT/integration-tests/build/remote-scenario.properties"
RUN_GRADE=$(grep '^grade=' "$PROPS" | cut -d= -f2)
EXPECTED=$(grep '^checkIns=' "$PROPS" | cut -d= -f2)
TODAY=$(adbs shell date +%F | tr -d '\r')
adbs shell cat "/sdcard/faceDB/attendance_$TODAY.csv" > "$ROOT/integration-tests/build/attendance_device.csv"
ACTUAL=$(python3 - "$ROOT/integration-tests/build/attendance_device.csv" "$RUN_GRADE" <<'PY'
import csv, sys
rows = list(csv.reader(open(sys.argv[1], encoding="utf-8-sig")))
print(sum(int(r[6] or 0) for r in rows[2:] if len(r) > 6 and r[2] == sys.argv[2]))
PY
)
echo "[e2e] check-ins expected=$EXPECTED on device sheet=$ACTUAL"
[ "$EXPECTED" = "$ACTUAL" ] || { echo "[e2e] attendance mismatch"; exit 1; }
echo "[e2e] OK"
