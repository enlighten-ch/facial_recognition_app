#!/usr/bin/env bash
# face_db.csv and the 명단 columns, on one device that runs both apps (emulator, test data only):
#   1. edit the CSV (rename 부서 → 반, change a value, add 메모, remove a row) while the server runs:
#      the server leaves the file alone until it starts again
#   2. restart the server: the edits are applied after a backup; the kiosk's form says 반
#   3. long press in 명단: one value, then a column name → the CSV and the kiosk follow at once
#   4. restore the backup taken before the CSV was applied: everything is back
#   5. an emptied CSV deletes no one; "CSV 편집 열기" without a spreadsheet app explains where the file is
# Usage: [DATA_DIR=/sdcard/faceDB] tools/e2e_roster_columns.sh <adb-serial> <server-admin-password>
set -uo pipefail
SERIAL="$1"; ADMIN_PASSWORD="$2"; DATA_DIR="${DATA_DIR:-/sdcard/faceDB}"
cd "$(dirname "$0")/.."
source tools/ui_automation.sh "$SERIAL"
SERVER=com.example.facedbserver
KIOSK=com.example.facetestbed
R="$KIOSK/com.allstar.faithid.kiosk.debug.DebugObservationReceiver"
WORK=$(mktemp -d)
P=0; F=0
check() { if eval "$2"; then echo "  PASS $1"; P=$((P+1)); else echo "  FAIL $1"; F=$((F+1)); fi; }
clear_field() { adbs shell input keyevent KEYCODE_MOVE_END; local i; for i in $(seq 1 24); do adbs shell input keyevent KEYCODE_DEL; done; }
b() { adbs shell am broadcast -n "$R" -a "com.allstar.faithid.kiosk.$1" "${@:2}" >/dev/null; }

# The server app on its dashboard: logged in, developer screen closed.
server_front() {
  adbs shell am start -n "$SERVER/com.allstar.faithid.serverapp.ui.MainActivity" >/dev/null; sleep 2
  if ui_has "로그인"; then
    ui_tap loginIdInput; clear_field; ui_type admin
    ui_tap loginPasswordInput; clear_field; ui_type "$ADMIN_PASSWORD"
    ui_tap loginButton; sleep 2
  fi
  dev_close
}
# Starts the server if it is stopped and waits until the developer screen says it runs.
ensure_server() {
  dev_section devNavStatus
  local i
  for i in 1 2 3 4 5 6; do
    ui_dump | grep -q ':id/stopButton' && break
    ui_tap startButton; sleep 5
  done
  dev_close
}
dev_section() { ui_dump | grep -q ':id/devBackButton' || { ui_tap devScreenButton; sleep 1; }; ui_tap "$1"; sleep 1; }
dev_close() { local i; for i in 1 2 3; do ui_dump | grep -q ':id/devBackButton' || return 0; ui_tap devBackButton; sleep 1; done; }
# Stop (confirming the dialog) and start again: the server reads face_db.csv when it starts.
restart_server() {
  dev_section devNavStatus
  ui_tap stopButton; sleep 1; ui_tap "중지"; sleep 4
  ui_tap startButton; sleep 8
  dev_close
  ensure_server
}
roster() { dev_close; ui_tap tabPeople; sleep 2; ui_texts; }
find_person() { dev_close; ui_tap tabPeople; sleep 1; ui_tap peopleSearchInput; clear_field; ui_type "$1"; adbs shell input keyevent KEYCODE_BACK; sleep 1; }
pull_csv() { adbs pull "$DATA_DIR/face_db.csv" "$WORK/now.csv" >/dev/null; cat "$WORK/now.csv"; }
kiosk_log() { adbs shell "run-as $KIOSK tail -n 30 files/logs/kiosk.log"; }
# The kiosk's name form (label only, nothing saved), kept as a screenshot: a UI dump of the form
# waits for the screen to settle and often times out on the emulator.
kiosk_form() {
  adbs shell am start -n "$KIOSK/com.allstar.faithid.kiosk.ui.MainActivity" >/dev/null; sleep 3
  b DEBUG_CAMERA --ez paused true; b DEBUG_FACE --es identity "form-$RANDOM" --ei count 5; sleep 2
  ui_tap "오늘은 이름표만"; sleep 2
  ui_shot "$WORK/kiosk-form-$1.png"
  adbs shell input keyevent KEYCODE_BACK; sleep 1; b DEBUG_CAMERA --ez paused false
}

server_front
ensure_server
adbs pull "$DATA_DIR/face_db.csv" "$WORK/original.csv" >/dev/null || { echo "face_db.csv not found"; exit 1; }
# Two test people whose names adb can type (ASCII): one is edited, one removed.
read -r EDIT_KEY EDIT_NAME GONE_KEY GONE_NAME < <(python3 - "$WORK/original.csv" "$WORK/edited.csv" <<'PY'
import csv, re, sys
rows = list(csv.reader(open(sys.argv[1], encoding="utf-8-sig")))
h = rows[0]
key, family, given = h.index("person_key"), h.index("family_name"), h.index("given_name")
ascii_rows = [r for r in rows[1:] if not r[family] and re.fullmatch(r"[A-Za-z][A-Za-z0-9-]{3,}", r[given])]
edit, gone = ascii_rows[0], ascii_rows[1]
h[h.index("department")] = "반"
h.insert(given + 1, "메모")
for row in rows[1:]:
    row.insert(given + 1, "E2E-MEMO" if row is edit else "")
grade = h.index("grade")
edit[grade] = "E2E-GRADE"
rows.remove(gone)
key, given = h.index("person_key"), h.index("given_name")
csv.writer(open(sys.argv[2], "w", encoding="utf-8", newline="")).writerows(rows)
print(edit[key], edit[given], gone[key], gone[given])
PY
)
echo "edited row: $EDIT_NAME ($EDIT_KEY), removed row: $GONE_NAME ($GONE_KEY)"

echo "1. 서버가 도는 동안 CSV 편집 → 덮어쓰지 않음"
adbs push "$WORK/edited.csv" "$DATA_DIR/face_db.csv" >/dev/null
dev_section devNavData; ui_tap exportCsvButton; sleep 2
T=$(ui_texts); dev_close
check "편집 대기 표시" 'grep -q "편집 대기 중" <<<"$T"'
check "CSV는 편집한 그대로" 'pull_csv | head -1 | grep -q "반"'
H=$(roster)
check "아직 반영 전 (부서 열 그대로)" 'grep -q "^부서" <<<"$H"'

echo "2. 서버 다시 시작 → 반영"
restart_server
H=$(roster)
check "열 이름이 반으로 (내용은 그대로)" 'grep -q "^반" <<<"$H" && ! grep -q "^부서" <<<"$H"'
check "새 메모 열" 'grep -q "^메모" <<<"$H"'
# The search box holds the typed name, so the footer's count tells whether the person is there.
check "지운 줄의 사람은 명단에서 삭제" 'find_person "$GONE_NAME"; ui_has "0명 표시 중"'
find_person "$EDIT_NAME"
check "고친 값 반영 (학년, 메모)" 'T=$(ui_texts); grep -qx "E2E-GRADE" <<<"$T" && grep -qx "E2E-MEMO" <<<"$T"'
dev_section devNavData; T=$(ui_texts); dev_close
check "반영 전 백업이 생김" 'grep -q "_csv.db" <<<"$T"'
check "다시 쓴 CSV 머리글도 반, 성·이름 열" 'pull_csv | head -1 | grep -q "family_name,given_name,\(.*,\)\?반,"'
sleep 3
check "키오스크가 열 이름을 받음 (입력 칸 이름: 반)" 'kiosk_log | grep -q "열 이름: department=반"'
kiosk_form ban

echo "3. 명단에서 길게 눌러 고치기"
server_front
find_person "$EDIT_NAME"
ui_longpress "E2E-MEMO"; sleep 1
ui_tap "메모 값"; clear_field; ui_type "E2E-LONG"; ui_hide_keyboard; ui_tap "저장"; sleep 3
check "칸 하나 고치면 CSV에 바로 반영" 'pull_csv | grep -q "E2E-LONG"'
ui_longpress "반 ▲" || ui_longpress "반"; sleep 1
ui_tap "열 이름"; clear_field; ui_type "BAN"; ui_hide_keyboard; ui_tap "저장"; sleep 3
check "열 이름 바꾸면 CSV 머리글에 바로 반영" 'pull_csv | head -1 | grep -q ",BAN,"'
check "키오스크도 바로 BAN을 받음" 'sleep 3; kiosk_log | grep -q "열 이름: department=BAN"'
kiosk_form BAN

echo "4. 백업으로 되돌리기"
server_front
dev_section devNavData
# Restore the copy taken just before the CSV was applied (its row ends in _csv.db).
T=$(ui_dump)
BOUNDS=$(python3 -c '
import re, sys
nodes = re.findall(r"<node [^>]*>", sys.stdin.read())
rows = [dict(re.findall(r"([\w-]+)=\"([^\"]*)\"", n)) for n in nodes]
csv_row = next(i for i, a in enumerate(rows) if a.get("text", "").endswith("_csv.db"))
btn = next(a for a in rows[csv_row:] if a.get("text") == "되돌리기")
print(btn["bounds"])' <<<"$T")
set -- $(echo "$BOUNDS" | tr '[],' '   ')
adbs shell input tap $(( ($1 + $3) / 2 )) $(( ($2 + $4) / 2 )); sleep 1
ui_tap "되돌리기"; sleep 6
dev_close
H=$(roster)
check "열 이름과 지운 사람이 돌아옴" 'grep -q "^부서" <<<"$H" && find_person "$GONE_NAME" && ui_has "1명 표시 중"'
check "CSV도 되돌린 상태" 'C=$(pull_csv); head -1 <<<"$C" | grep -q ",department," && ! grep -q "E2E-GRADE\|E2E-LONG" <<<"$C"'

echo "5. 빈 CSV, CSV 편집 열기"
COUNT=$(pull_csv | wc -l)
head -1 "$WORK/now.csv" > "$WORK/empty.csv"
adbs push "$WORK/empty.csv" "$DATA_DIR/face_db.csv" >/dev/null
server_front; restart_server
check "줄이 없는 CSV로는 아무도 지우지 않음" '[ "$(pull_csv | wc -l)" = "$COUNT" ]'
dev_close; ui_tap tabPeople; sleep 1; ui_tap peopleOpenCsvButton; sleep 2
check "CSV 편집 열기: 앱이 없으면 파일 위치 안내" 'ui_has "CSV를 열 앱이 없습니다"'
ui_tap "확인"
ui_tap peopleSearchInput; clear_field; adbs shell input keyevent KEYCODE_BACK

echo "ROSTER COLUMNS: PASS=$P FAIL=$F (work files in $WORK)"
[ "$F" = 0 ]
