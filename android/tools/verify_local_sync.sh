#!/usr/bin/env bash
set -euo pipefail

ADB_BIN="${ADB_BIN:-adb}"
CLIENT_PKG="com.example.facetestbed"
SERVER_PKG="com.example.facedbserver"

log() {
  printf '[verify] %s\n' "$*"
}

require_adb() {
  command -v "$ADB_BIN" >/dev/null 2>&1 || {
    echo "adb not found: $ADB_BIN" >&2
    exit 1
  }
}

dump_client_log() {
  log "client log tail"
  "$ADB_BIN" shell run-as "$CLIENT_PKG" sh -lc 'tail -n 120 files/logs/db_sync_client.log' || true
}

dump_server_ui_state() {
  log "server ui snapshot"
  "$ADB_BIN" shell am start -n "$SERVER_PKG"/.MainActivity >/dev/null || true
  "$ADB_BIN" shell uiautomator dump /sdcard/dbserver_verify.xml >/dev/null 2>&1 || true
  "$ADB_BIN" pull /sdcard/dbserver_verify.xml /tmp/dbserver_verify.xml >/dev/null 2>&1 || true
  rg -n "서버 실행중|서버 중지|승인 대기|승인 완료|연결 상태|엔드포인트|client-" /tmp/dbserver_verify.xml -S || true
}

probe_health() {
  local url="$1"
  log "health probe: $url"
  "$ADB_BIN" shell "curl -sS --max-time 3 -w '\nHTTP:%{http_code} TIME:%{time_total}\n' '$url/api/health'" || true
}

detect_device_ipv4() {
  "$ADB_BIN" shell "ip -f inet addr show wlan0 2>/dev/null | awk '/inet /{print \$2}' | cut -d/ -f1 | head -n1" \
    | tr -d '\r'
}

main() {
  require_adb
  log "device list"
  "$ADB_BIN" devices -l

  probe_health "http://127.0.0.1:18080"
  local_ip="$(detect_device_ipv4)"
  if [[ -n "$local_ip" ]]; then
    probe_health "http://${local_ip}:18080"
  fi

  dump_server_ui_state
  dump_client_log
}

main "$@"
