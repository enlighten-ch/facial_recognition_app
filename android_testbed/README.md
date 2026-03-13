# Android APK Testbed (Emulator only)

## What is set up
- Native Android app project at `android_testbed/`
- SDK + emulator + API 35 system image installed in `$HOME/Android/Sdk`
- AVD created: `FaceTestbed_API35`
- Debug APK builds with `./gradlew :app:assembleDebug`

## One-command run
```bash
cd /home/enlighten/ws/facial_recognition/android_testbed
./scripts/run_testbed.sh
```

## Manual run
```bash
export ANDROID_SDK_ROOT=$HOME/Android/Sdk
export PATH=$PATH:$ANDROID_SDK_ROOT/platform-tools:$ANDROID_SDK_ROOT/emulator:$ANDROID_SDK_ROOT/cmdline-tools/latest/bin

cd /home/enlighten/ws/facial_recognition/android_testbed
./gradlew :app:assembleDebug
emulator @FaceTestbed_API35 -no-window -gpu swiftshader_indirect -camera-back webcam0 -camera-front none
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell monkey -p com.example.facetestbed -c android.intent.category.LAUNCHER 1
```

## Android Studio install status
Android Studio Linux tarball is downloading in background:
- File: `/home/enlighten/tools/android-studio-panda2-linux.tar.gz`
- Log: `/home/enlighten/tools/android_studio_download.log`

Check progress:
```bash
ls -lh /home/enlighten/tools/android-studio-panda2-linux.tar.gz
tail -f /home/enlighten/tools/android_studio_download.log
```

When complete:
```bash
cd /home/enlighten/tools
tar -xzf android-studio-panda2-linux.tar.gz
/home/enlighten/tools/android-studio/bin/studio.sh
```
