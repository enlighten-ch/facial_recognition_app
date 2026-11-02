#!/usr/bin/env bash
# Walks through every kiosk screen with simulated faces (debug build) and checks the texts; keeps screenshots.
# Usage: [BLUR=1] [NOPRINT=1] tools/ui_walkthrough_kiosk.sh <serial> <out-dir> <known-identity|register>
#   BLUR=1     real tablets: the front camera may show people, keep only heavily blurred screenshots
#   NOPRINT=1  a real label printer is attached: do not press the print button
#   register   registers a test person first and uses it as the known face
set -uo pipefail
SERIAL="$1"; OUT="$2"; KNOWN="$3"
cd "$(dirname "$0")/.."
source tools/ui_automation.sh "$SERIAL"
mkdir -p "$OUT"
R="com.example.facetestbed/com.allstar.faithid.kiosk.debug.DebugObservationReceiver"
b() { adbs shell am broadcast -n "$R" -a "com.allstar.faithid.kiosk.$1" "${@:2}" >/dev/null; }
face() { local a=(--es identity "$1" --ei count "${3:-1}"); [ -n "${2:-}" ] && a+=(--es direction "$2"); [ -n "${4:-}" ] && a+=(--ef size "$4"); b DEBUG_FACE "${a[@]}"; }
noface() { for _ in 1 2 3 4 5 6 7; do b DEBUG_NO_FACE; done; }
seen() { face "$1" "" 5; }
# Front and four directions per circle (normal, big = closer, small = further); 2 frames to hold each pose.
motions() { for size in 1.0 1.18 0.73; do for d in FRONT LEFT RIGHT UP DOWN; do face "$1" $d 2 $size; sleep 0.6; done; done; }
# Text goes in through the debug receiver (no keyboard: Samsung's keyboard can crash under adb input).
# BACK only while the keyboard shows: otherwise it would close the kiosk screen.
hidekb() { adbs shell dumpsys input_method | grep -q "mInputShown=true" && adbs shell input keyevent KEYCODE_BACK; sleep 0.5; }
typein() { ui_tap "$1"; sleep 1.5; popup; b DEBUG_TYPE --es text "$2"; sleep 0.5; adbs shell input keyevent KEYCODE_ENTER; sleep 1; popup; }
# Dismisses a system "app keeps stopping" dialog of another app if one is showing.
popup() { local i; for i in 1 2 3; do if ui_has "앱 닫기"; then ui_tap "앱 닫기"; elif ui_has "Close app"; then ui_tap "Close app"; else break; fi; sleep 1; done; true; }
shot() {
  ui_shot "$OUT/$1.png"
  if [ "${BLUR:-0}" = 1 ]; then python3 -c "
from PIL import Image, ImageFilter; import sys
p=sys.argv[1]; Image.open(p).convert('RGB').filter(ImageFilter.GaussianBlur(28)).save(p)" "$OUT/$1.png"; fi
}
P=0; F=0
check() { if eval "$2"; then echo "  PASS $1"; P=$((P+1)); else echo "  FAIL $1"; F=$((F+1)); fi; }
RUN=$(date +%H%M%S); NEW="ui-new-$RUN"

adbs shell input keyevent KEYCODE_WAKEUP; adbs shell am start -n com.example.facetestbed/com.allstar.faithid.kiosk.ui.MainActivity >/dev/null; sleep 3
b DEBUG_CAMERA --ez paused true; noface; sleep 2
if [ "$KNOWN" = register ]; then
  KNOWN="ui-known-$RUN"
  seen "$KNOWN"; sleep 2; popup; ui_tap "좋아, 등록할래!"; sleep 2; motions "$KNOWN"; sleep 5; popup
  typein opNameInput "UIKNOWN-$RUN"; popup; ui_tap opSubmitButton; sleep 6; noface; sleep 3
fi
shot 01-idle
check "기본 안내 (빛이)" 'ui_has "안녕! 나는 빛이야. 동그라미 안에 얼굴을 보여 줄래?"'
check "설정 버튼은 짧게 누르면 아무 반응 없음" 'ui_tap opSettingsButton; sleep 1; ! ui_has "안내 문구"'

seen "$KNOWN"; sleep 2; shot 02-recognized
check "인식: 이름표 받기 버튼" 'ui_has "이름표 받기"'
check "인식: 나 아닌데? 버튼" 'ui_has "나 아닌데?"'
noface; sleep 4

seen "$NEW"; sleep 2; shot 03-unknown
check "처음 보는 얼굴 안내" 'ui_has "처음 보는 얼굴이네! 우리 친구 할까?"'
ui_tap "좋아, 등록할래!"; sleep 2; shot 04-reg-front-normal
check "등록 첫 안내: 정면 + 보통 동그라미" 'ui_has "앞을 보면서 동그라미에 얼굴을 쏙 맞춰 줄래?"'
check "별 개수 0/15" 'ui_has "★ 별 0 / 15"'
b DEBUG_SEVERAL --ei faces 2; sleep 1.5; shot 05-reg-several
check "얼굴이 여럿이면 한 명씩 안내" 'ui_has "한 번에 한 친구만 등록할게!"'
face "$NEW" FRONT 1 1.0; sleep 1.5; shot 06-reg-hold
check "맞추면 그대로 있으라는 안내" 'ui_has "좋아, 그대로! 하나, 둘…"'
face "$NEW" FRONT 1 1.0; sleep 1.5
check "머문 뒤 별 1개" 'ui_has "★ 별 1 / 15"'
face "$NEW" LEFT 2 0.85; sleep 1.5; shot 07-reg-closer
check "크기가 작으면 가까이 오라는 안내" 'ui_has "조금 더 가까이 와 볼까?"'
check "원 사이 크기는 받지 않음" 'ui_has "★ 별 1 / 15"'
face "$NEW" UP 2 0.73; sleep 1.5
check "권하는 칸이 아니어도 빈칸이면 받음 (위·멀리)" 'ui_has "★ 별 2 / 15"'
for d in LEFT RIGHT UP DOWN; do face "$NEW" $d 2 1.0; sleep 0.6; done
sleep 1; shot 08-reg-big
check "보통 판을 마치면 가까이 판" 'ui_has "이번엔 가까이 와서 앞을 보면서 큰 동그라미를 꽉 채워 줄래?"'
for d in FRONT LEFT RIGHT UP DOWN; do face "$NEW" $d 2 1.18; sleep 0.6; done
for d in FRONT LEFT RIGHT DOWN; do face "$NEW" $d 2 0.73; sleep 0.6; done
shot 09-celebration
sleep 3; shot 10-form
check "이름 입력 화면" 'ui_has "선생님, 이름을 적어 주세요"'
typein opNameInput "UINEW-$RUN"; shot 11-form-filled
ui_tap opSubmitButton; sleep 2; shot 12-registered
check "등록 완료 안내" 'ui_texts | grep -q "등록 완료!"'
sleep 3
seen "$NEW"; sleep 2; shot 13-new-recognized
check "방금 등록한 사람 인식" 'ui_texts | grep -q "반가워, "'
if [ "${NOPRINT:-0}" = 1 ]; then
  ui_tap "나 아닌데?"; sleep 1; echo "  (print skipped: real label printer)"
else
  ui_tap "이름표 받기"; sleep 1.2; shot 14-output
  check "이름표 화면" 'ui_texts | grep -q "이름표를 가져가요"'
  sleep 3; adbs shell dumpsys window | grep -E "mCurrentFocus" | grep -qi printspooler && { adbs shell input keyevent KEYCODE_BACK; sleep 1; }
fi
sleep 5; noface; sleep 3

seen "other-$RUN"; sleep 2; ui_tap "좋아, 등록할래!"; sleep 1
motions "$KNOWN"
sleep 4; shot 15-duplicate
check "이미 아는 얼굴 질문" 'ui_texts | grep -q "맞아요?"'
ui_tap "아니에요, 처음이에요"; sleep 2; shot 16-dup-to-form
check "처음이에요 → 이름 입력" 'ui_has "선생님, 이름을 적어 주세요"'
ui_tap opHomeButton; sleep 2; noface; sleep 3

seen "guest-$RUN"; sleep 2; ui_tap "오늘은 이름표만"; sleep 2; shot 17-labelonly
check "이름표만 화면" 'ui_has "이름표에 뭐라고 쓸까?"'
ui_tap opHomeButton; sleep 2

noface; b DEBUG_CAMERA --ez paused false
echo "KIOSK UI: PASS=$P FAIL=$F"
