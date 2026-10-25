#!/usr/bin/env bash
# The DB server app and the kiosk on one tablet: does the kiosk keep syncing while the server app
# is in front? (issue #19)
#
# The kiosk screen is closed (debug-only DEBUG_FINISH) before the server app comes to the front,
# as Android may do on its own, and the process is then offered to the low-memory cleanup
# (am kill). The server data is not changed: "DB 다시 읽기" only makes every kiosk download the
# database again.
#
# Preconditions: the server app is running, the kiosk (debug build) is approved and connected.
# Usage: tools/e2e_background_sync.sh <adb-serial> [server-admin-password]
set -uo pipefail
SERIAL="$1"
ADMIN_PASSWORD="${2:-church2026}"
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/ui_automation.sh" "$SERIAL"

KIOSK=com.example.facetestbed
SERVER=com.example.facedbserver
PASS=0
FAIL=0
ok() { echo "  PASS: $*"; PASS=$((PASS + 1)); }
bad() { echo "  FAIL: $*"; FAIL=$((FAIL + 1)); }
now_ms() { date +%s%3N; }
offset_ms() { local a d b; a=$(now_ms); d=$(adbs shell date +%s%3N | tr -d '\r'); b=$(now_ms); echo $(( d - (a + b) / 2 )); }
# Host ms of the first kiosk log line containing $1 after host ms $2, waiting up to $3 seconds.
kiosk_log_after() {
  local text="$1" after="$2" off deadline; off=$(offset_ms); deadline=$(( $(date +%s) + $3 ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
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
clear_field() { adbs shell input keyevent KEYCODE_MOVE_END; for _ in $(seq 24); do adbs shell input keyevent KEYCODE_DEL; done; }
server_front() {
  adbs shell am start -n "$SERVER/com.allstar.faithid.serverapp.ui.MainActivity" >/dev/null 2>&1; sleep 3
  if ui_has "로그인"; then
    ui_tap loginIdInput; clear_field; ui_type admin
    ui_tap loginPasswordInput; clear_field; ui_type "$ADMIN_PASSWORD"; ui_tap loginButton; sleep 3
  fi
}
kiosk_front() { adbs shell am start -n "$KIOSK/com.allstar.faithid.kiosk.ui.MainActivity" >/dev/null 2>&1; }
kiosk_pid() { adbs shell pidof "$KIOSK" | tr -d '\r'; }

trap kiosk_front EXIT

echo "=== 0. 키오스크를 앞에 두고 연결 확인"
T0=$(now_ms)
kiosk_front; sleep 8
PID_BEFORE=$(kiosk_pid)
[ -n "$PID_BEFORE" ] && ok "키오스크 실행 중 (pid $PID_BEFORE)" || bad "키오스크가 실행되지 않음"

echo "=== 1. 키오스크 화면 종료 → 서버 앱을 앞으로"
adbs shell am broadcast -n "$KIOSK/com.allstar.faithid.kiosk.debug.DebugObservationReceiver" -a com.allstar.faithid.kiosk.DEBUG_FINISH >/dev/null
sleep 2
server_front
sleep 3
[ "$(kiosk_pid)" = "$PID_BEFORE" ] && ok "화면이 없어도 키오스크 프로세스 유지" || bad "키오스크 프로세스가 사라지거나 바뀜 ($(kiosk_pid))"

echo "=== 2. 백그라운드 프로세스 정리 (am kill)"
adbs shell am kill "$KIOSK"; sleep 3
[ "$(kiosk_pid)" = "$PID_BEFORE" ] && ok "정리 대상이 아님 (포그라운드 서비스)" || bad "정리되어 종료됨 (pid: '$(kiosk_pid)')"

echo "=== 3. 키오스크가 뒤에 있는 동안 서버 변경 받기"
T1=$(now_ms)
ui_tap "명단"; sleep 1; ui_tap "DB 다시 읽기"; sleep 1; ui_tap "다시 읽기"; sleep 1; ui_tap "현황"
if T2=$(kiosk_log_after "서버 DB 전체 수신" "$T1" 40); then
  ok "뒤에 있는 동안 전체 DB 수신·적용 ($(( T2 - T1 )) ms, 버튼 조작 포함)"
else
  bad "40초 안에 서버 변경을 받지 못함"
fi

echo "=== 4. 키오스크로 돌아오기"
SERVER_COUNT=$(ui_texts | grep -m1 -o "등록 [0-9]*명" | grep -o "[0-9]*")
T3=$(now_ms)
kiosk_front
sleep 6
STARTED=$(kiosk_log_after "앱 시작" "$T3" 1 >/dev/null && echo 예 || echo 아니오)
[ "$(kiosk_pid)" = "$PID_BEFORE" ] && ok "같은 프로세스로 복귀 (모델·DB 다시 열지 않음)" || bad "프로세스가 새로 시작됨"
if ui_texts | grep -q "환영해요\|얼굴이 등록되어\|반가워요"; then ok "바로 인식 화면 (대기 또는 인식 결과)"; else bad "인식 대기 화면이 아님: $(ui_texts | head -3 | tr '\n' ' ')"; fi
echo "  서버 ${SERVER_COUNT:-?}명 · 키오스크 화면 다시 생성: $STARTED"

echo
echo "Background sync: PASS=$PASS FAIL=$FAIL"
[ "$FAIL" = 0 ]
