#!/usr/bin/env bash
set -euo pipefail

SDK_ROOT="${ANDROID_SDK_ROOT:-$HOME/Android/Sdk}"
export ANDROID_SDK_ROOT="$SDK_ROOT"
export ANDROID_HOME="$SDK_ROOT"
export PATH="$PATH:$SDK_ROOT/platform-tools:$SDK_ROOT/emulator:$SDK_ROOT/cmdline-tools/latest/bin"

APP_APK="$(cd "$(dirname "$0")/.." && pwd)/app/build/outputs/apk/debug/app-debug.apk"
AVD_NAME="FaceTestbed_API35"
PKG="com.example.facetestbed"

if ! adb devices | grep -q "emulator-"; then
  nohup emulator @${AVD_NAME} -no-window -gpu swiftshader_indirect -no-snapshot-load -no-boot-anim -camera-back webcam0 -camera-front none > "$HOME/tools/face_testbed_emulator_headless.log" 2>&1 < /dev/null &
fi

adb wait-for-device
for _ in $(seq 1 120); do
  if [ "$(adb shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" = "1" ]; then
    break
  fi
  sleep 2
done

if [ ! -f "$APP_APK" ]; then
  (cd "$(dirname "$0")/.." && ./gradlew :app:assembleDebug)
fi

adb install -r "$APP_APK"
adb shell monkey -p "$PKG" -c android.intent.category.LAUNCHER 1 >/dev/null
adb devices -l
echo "Installed and launched $PKG on emulator."
