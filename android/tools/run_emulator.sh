#!/usr/bin/env bash
# Starts the test emulator (tablet-sized screen) and installs both debug apps.
# Usage: tools/run_emulator.sh
set -euo pipefail
SDK_ROOT="${ANDROID_SDK_ROOT:-$HOME/Android/Sdk}"
export PATH="$PATH:$SDK_ROOT/platform-tools:$SDK_ROOT/emulator"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AVD_NAME="${AVD_NAME:-FaceTestbed_API35}"

if ! adb devices | grep -q "emulator-"; then
  nohup emulator "@$AVD_NAME" -no-window -gpu swiftshader_indirect -no-snapshot-load -no-boot-anim \
    -camera-front emulated -camera-back emulated -no-audio > "$ROOT/build/emulator.log" 2>&1 < /dev/null &
fi
adb wait-for-device
until [ "$(adb shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" = "1" ]; do sleep 2; done
# Galaxy Tab S7+ resolution so the tablet layout (sw600dp) is used.
adb shell wm size 1752x2800
adb shell wm density 320

(cd "$ROOT" && ./gradlew :kiosk-app:assembleDebug :server-app:assembleDebug)
adb install -r -g "$ROOT/server-app/build/outputs/apk/debug/faith_db.apk"
adb install -r -g "$ROOT/kiosk-app/build/outputs/apk/debug/faith_id.apk"
adb shell appops set com.example.facedbserver MANAGE_EXTERNAL_STORAGE allow
echo "Installed Faith DB Server and Faith ID on $(adb get-serialno)."
