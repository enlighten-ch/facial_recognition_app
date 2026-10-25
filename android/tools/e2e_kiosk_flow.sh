#!/usr/bin/env bash
# End-to-end kiosk scenario on one device that runs both apps (debug kiosk build).
#
# Preconditions:
#   - Faith DB Server is running on the device (port 18080) and the kiosk is approved
#   - Faith ID (debug build) is running in operation mode, server URL http://127.0.0.1:18080
#   - printer type "Android 기본 인쇄" (the print dialog is cancelled with BACK, which also
#     checks that a cancelled print still records the check-in)
# Usage: tools/e2e_kiosk_flow.sh <adb-serial> <server-admin-password>
set -uo pipefail
SERIAL="$1"
ADMIN_PASSWORD="${2:?server admin password required}"
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/ui_automation.sh" "$SERIAL"

KIOSK=com.example.facetestbed
SERVER=com.example.facedbserver
RECEIVER="$KIOSK/com.allstar.faithid.kiosk.debug.DebugObservationReceiver"
RUN=$(date +%H%M%S)
PERSON="e2e-face-$RUN"
NAME="E2E-$RUN"
TODAY=$(adbs shell date +%F | tr -d '\r')
SHEET="/sdcard/faceDB/attendance_$TODAY.csv"
PASS=0
FAIL=0

step() { echo; echo "=== $*"; }
ok() { echo "  PASS: $*"; PASS=$((PASS + 1)); }
bad() { echo "  FAIL: $*"; FAIL=$((FAIL + 1)); ui_shot "/tmp/e2e_fail_${RUN}_$FAIL.png" 2>/dev/null; }
check() { if eval "$2"; then ok "$1"; else bad "$1"; fi; }
broadcast() { adbs shell am broadcast -n "$RECEIVER" -a "com.allstar.faithid.kiosk.$1" "${@:2}" >/dev/null; }
face() { # identity [direction] [count]
  local args=(--es identity "$1" --ei count "${3:-1}")
  [ -n "${2:-}" ] && args+=(--es direction "$2")
  broadcast DEBUG_FACE "${args[@]}"
}
wait_for() { # seconds description command
  local deadline=$(( $(date +%s) + $1 ))
  while [ "$(date +%s)" -lt "$deadline" ]; do eval "$3" && return 0; sleep 1; done
  return 1
}
sheet_count() { # name -> check-in count in today's sheet
  adbs shell cat "$SHEET" 2>/dev/null | python3 -c '
import csv, sys
rows = list(csv.reader(sys.stdin.read().lstrip("﻿").splitlines()))
print(sum(int(r[6] or 0) for r in rows[2:] if len(r) > 6 and r[0] == sys.argv[1]))' "$1"
}
clear_field() { adbs shell input keyevent KEYCODE_MOVE_END; for _ in $(seq 40); do adbs shell input keyevent KEYCODE_DEL; done; }
kiosk_front() { adbs shell am start -n "$KIOSK/com.allstar.faithid.kiosk.ui.MainActivity" >/dev/null; sleep 2; }
cancel_print_dialog() { # the Android print dialog belongs to the print spooler
  sleep 3
  if adbs shell dumpsys window | grep -E "mCurrentFocus|mFocusedApp" | grep -qi printspooler; then
    adbs shell input keyevent KEYCODE_BACK; sleep 1
    echo "  (print dialog cancelled)"
  fi
}

kiosk_front
broadcast DEBUG_CAMERA --ez paused true
broadcast DEBUG_NO_FACE
sleep 1

step "1. 처음 보는 얼굴 → 등록"
face "$PERSON"; sleep 1
check "미등록 안내와 등록 버튼" 'ui_has "등록 시작하기"'
ui_tap "등록 시작하기"; sleep 1
for d in LEFT RIGHT UP DOWN; do face "$PERSON" "$d" 3; sleep 0.5; done
sleep 1
check "12장 수집 후 이름 입력 화면" 'ui_has "저장하기"'
ui_tap nameInput; ui_type "$NAME"; ui_tap departmentInput; ui_type "E2E"; ui_tap gradeInput; ui_type "G1"
adbs shell input keyevent KEYCODE_BACK # hide keyboard
ui_tap submitInfoButton; sleep 2
check "서버 명단에 반영 (face_db.csv)" 'wait_for 20 csv "adbs shell grep -c \"$NAME\" /sdcard/faceDB/face_db.csv | grep -q 1"'

step "2. 등록된 얼굴 → 인식 → 출력 버튼 → 출석"
face "$PERSON"; sleep 1
check "인식 문구에 이름 표시" "ui_texts | grep -q \"$NAME\""
ui_tap operationPrintButton
cancel_print_dialog
check "인쇄를 취소해도 출석 1회" 'wait_for 20 sheet "[ \"\$(sheet_count $NAME)\" = 1 ]"'

step "3. 이미 등록된 얼굴로 다시 등록 시도 → 본인 확인"
broadcast DEBUG_NO_FACE; sleep 4
face "other-$RUN"; sleep 1
ui_tap "등록 시작하기"; sleep 1
for d in LEFT RIGHT UP DOWN; do face "$PERSON" "$d" 3; sleep 0.5; done
sleep 1
check "'본인이 맞아요' 질문 표시" 'ui_has "본인이 맞아요"'
ui_tap "본인이 맞아요"; sleep 1
ui_tap operationPrintButton
cancel_print_dialog
check "본인 확인 후 출석 2회, 새 사람은 만들지 않음" 'wait_for 20 sheet "[ \"\$(sheet_count $NAME)\" = 2 ]" && [ "$(adbs shell grep -c "$NAME" /sdcard/faceDB/face_db.csv | tr -d "\r")" = 1 ]'

step "4. 이름표만 출력 → 명단 외 출석"
broadcast DEBUG_NO_FACE; sleep 4
face "guest-$RUN"; sleep 1
ui_tap "이름표만 출력하기"; sleep 1
ui_tap nameInput; ui_type "GUEST-$RUN"
adbs shell input keyevent KEYCODE_BACK
ui_tap submitInfoButton
cancel_print_dialog
check "명단 외 출석 기록" 'wait_for 20 sheet "[ \"\$(sheet_count GUEST-$RUN)\" = 1 ]"'

step "5. 서버에서 삭제 → 키오스크에서도 사라짐"
adbs shell am start -n "$SERVER/com.allstar.faithid.serverapp.ui.MainActivity" >/dev/null; sleep 3
if ui_has "로그인"; then ui_tap loginIdInput; clear_field; ui_type admin; ui_tap loginPasswordInput; clear_field; ui_type "$ADMIN_PASSWORD"; ui_tap loginButton; sleep 2; fi
ui_tap "명단"; sleep 1
ui_tap peopleSearchInput; clear_field; ui_type "$NAME"; adbs shell input keyevent KEYCODE_BACK; sleep 1
ui_tap personDeleteButton; sleep 1; ui_tap "삭제"; sleep 2
check "서버 명단에서 삭제" 'wait_for 10 csv "! adbs shell grep -q \"$NAME\" /sdcard/faceDB/face_db.csv"'
ui_tap "현황"
kiosk_front
broadcast DEBUG_CAMERA --ez paused true
broadcast DEBUG_NO_FACE; sleep 6
face "$PERSON"; sleep 1
check "삭제된 사람은 미등록으로 보임" 'wait_for 15 ui "face $PERSON; sleep 1; ui_has \"등록 시작하기\""'
check "삭제 후에도 오늘 출석 기록은 유지" '[ "$(sheet_count $NAME)" = 2 ]'

broadcast DEBUG_NO_FACE
broadcast DEBUG_CAMERA --ez paused false
echo
echo "E2E kiosk flow: PASS=$PASS FAIL=$FAIL (run $RUN)"
[ "$FAIL" = 0 ]
