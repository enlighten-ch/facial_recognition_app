#!/usr/bin/env bash
# 명단 열: the roster follows the columns of face_db.csv, and the developer screen changes the
# headers and hides columns.
#   1. face_db.csv with "department" renamed to "반" and a new "메모" column → 'CSV 편집 반영'
#      → the 명단 shows 반 and 메모, no empty 부서, and the filter chips use 반
#   2. developer screen: header of 반 → "BAN-1", grade hidden → the 명단 follows, config.yaml keeps it
#   3. the original CSV and settings are put back
# Test data only (emulator): it rewrites every person's columns and restores them at the end.
# Usage: [DATA_DIR=/sdcard/faceDB] tools/e2e_roster_columns.sh <adb-serial> <server-admin-password>
set -uo pipefail
SERIAL="$1"; ADMIN_PASSWORD="$2"; DATA_DIR="${DATA_DIR:-/sdcard/faceDB}"
cd "$(dirname "$0")/.."
source tools/ui_automation.sh "$SERIAL"
SERVER=com.example.facedbserver
WORK=$(mktemp -d)
P=0; F=0
check() { if eval "$2"; then echo "  PASS $1"; P=$((P+1)); else echo "  FAIL $1"; F=$((F+1)); fi; }
clear_field() { adbs shell input keyevent KEYCODE_MOVE_END; local i; for i in $(seq 1 24); do adbs shell input keyevent KEYCODE_DEL; done; }

server_front() {
  adbs shell am start -n "$SERVER/com.allstar.faithid.serverapp.ui.MainActivity" >/dev/null; sleep 2
  if ui_has "로그인"; then
    ui_tap loginIdInput; clear_field; ui_type admin
    ui_tap loginPasswordInput; clear_field; ui_type "$ADMIN_PASSWORD"
    ui_tap loginButton; sleep 2
  fi
}
dev_section() { ui_tap devScreenButton; sleep 1; ui_tap "$1"; sleep 1; }
dev_close() { ui_tap devBackButton; sleep 1; }
apply_csv() {
  adbs push "$1" "$DATA_DIR/face_db.csv" >/dev/null
  dev_section devNavData; ui_tap importCsvButton; sleep 4
  APPLIED=$(ui_texts | grep "바뀐 사람" || true); echo "    $APPLIED"
  ui_tap "확인"; sleep 1; dev_close
}
roster() { ui_tap tabPeople; sleep 2; ui_texts; }
config() { adbs shell run-as "$SERVER" cat files/face_db_server/config.yaml; }

server_front
adbs pull "$DATA_DIR/face_db.csv" "$WORK/original.csv" >/dev/null || { echo "face_db.csv not found"; exit 1; }
FIRST_DEPT=$(python3 - "$WORK/original.csv" "$WORK/renamed.csv" <<'PY'
import csv, sys
rows = list(csv.reader(open(sys.argv[1], encoding="utf-8-sig")))
header = rows[0]
d = header.index("department")
header[d] = "반"
header.append("메모")
for i, row in enumerate(rows[1:]):
    row.append("새가족" if i == 0 else "")
csv.writer(open(sys.argv[2], "w", encoding="utf-8", newline="")).writerows(rows)
print(next((r[d] for r in rows[1:] if r[d]), ""))
PY
)

echo "1. CSV의 열을 바꾸면 명단 열이 따라옴"
apply_csv "$WORK/renamed.csv"
H=$(roster)
check "부서 열을 반으로 바꾸면 명단에 반 열" 'grep -q "^반" <<<"$H"'
check "값이 없어진 부서 열은 빠짐" '! grep -q "^부서" <<<"$H"'
check "새로 더한 메모 열도 나타남" 'grep -q "^메모" <<<"$H"'
check "분류 버튼은 반 값으로 ($FIRST_DEPT)" 'grep -qx "●  $FIRST_DEPT" <<<"$H"'
ui_shot "$WORK/roster-renamed.png"

echo "2. 개발자 화면에서 머리글·보이기"
dev_section devNavSettings
T=$(ui_texts)
check "열 목록: 반, 메모, 값 없는 department" 'grep -qx "반" <<<"$T" && grep -qx "메모" <<<"$T" && grep -q "^department  · 값 없음" <<<"$T"'
ui_tap "반 열의 머리글"; clear_field; ui_type "BAN-1"; ui_hide_keyboard
ui_tap "grade 열 보이기"; sleep 0.5
ui_tap devSaveButton; sleep 1; dev_close
H=$(roster)
check "머리글이 BAN-1로" 'grep -q "^BAN-1" <<<"$H" && ! grep -q "^반" <<<"$H"'
check "학년 열 숨김" '! grep -q "^학년" <<<"$H"'
check "config.yaml에 저장" 'config | grep -q "roster_column_labels: \"반=BAN-1\"" && config | grep -q "roster_hidden_columns: \".*grade"'
ui_shot "$WORK/roster-labels.png"

echo "3. 되돌리기"
dev_section devNavSettings
ui_tap "반 열의 머리글"; clear_field; ui_hide_keyboard
ui_tap "grade 열 보이기"; sleep 0.5
ui_tap devSaveButton; sleep 1; dev_close
apply_csv "$WORK/original.csv"
H=$(roster)
check "원래 CSV로 부서·학년 열이 돌아옴" 'grep -q "^부서" <<<"$H" && grep -q "^학년" <<<"$H" && ! grep -q "^반" <<<"$H" && ! grep -q "^메모" <<<"$H"'
check "설정도 원래대로" 'config | grep -q "roster_column_labels: \"\"" && config | grep -q "roster_hidden_columns: \"family_name\""'

echo "ROSTER COLUMNS: PASS=$P FAIL=$F (screenshots in $WORK)"
[ "$F" = 0 ]
