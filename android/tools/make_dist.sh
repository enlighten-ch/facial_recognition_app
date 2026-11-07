#!/usr/bin/env bash
# Builds both apps and copies the install files to dist/ at the project root:
#   dist/faith_id.apk   Faith ID (kiosk)
#   dist/faith_db.apk   Faith DB Server
#   dist/BUILD.txt      version, commit and build time
# Debug builds, like the ones on the tablets: an update over them keeps each tablet's data.
# Usage: tools/make_dist.sh
set -euo pipefail
cd "$(dirname "$0")/.."
DIST=../dist
./gradlew -q :kiosk-app:assembleDebug :server-app:assembleDebug
mkdir -p "$DIST"
cp kiosk-app/build/outputs/apk/debug/faith_id.apk "$DIST/faith_id.apk"
cp server-app/build/outputs/apk/debug/faith_db.apk "$DIST/faith_db.apk"
{
  echo "built:   $(date '+%Y-%m-%d %H:%M')"
  echo "commit:  $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- . || echo ' (with uncommitted changes)')"
  echo "version: $(grep -m1 versionName kiosk-app/build.gradle.kts | sed 's/.*"\(.*\)".*/\1/')"
} > "$DIST/BUILD.txt"
ls -lh "$DIST"/*.apk
cat "$DIST/BUILD.txt"
