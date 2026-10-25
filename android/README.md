# Faith ID Android

| 모듈 | 종류 | 역할 |
|---|---|---|
| `core` | Kotlin/JVM | 인물·메타데이터 모델, 바이너리/CSV 코덱, 인식 판정 규칙, 등록 세션, 얼굴 정렬 수학, 출석부 생성, 저장소(`PeopleStore`, 스키마 v3와 이전 DB 이전) |
| `server-core` | Kotlin/JVM | 서버 로직: 리비전 기반 인물 저장소, 키오스크 승인(`ClientRegistry`), 출석 원장, HTTP/WebSocket API v2(Ktor), 관리자 비밀번호·설정 |
| `client-core` | Kotlin/JVM | 키오스크 동기화: 로컬 얼굴 DB + 미전송함(outbox), `SyncEngine` |
| `android-support` | Android 라이브러리 | 플랫폼 SQLite 어댑터, 시스템 바 여백 처리, 파일 로그 |
| `kiosk-app` | Android 앱 | Faith ID 키오스크 (`com.example.facetestbed`) |
| `server-app` | Android 앱 | Faith DB Server (`com.example.facedbserver`) |
| `integration-tests` | JVM 테스트 | 실제 HTTP/WebSocket으로 여러 키오스크와 서버를 동시에 돌리는 시나리오 |
| `tools/` | 스크립트 | 에뮬레이터 실행, 기기 E2E 시나리오 |

앱 ID(`com.example.*`)는 첫 배포 그대로 두어 설치된 기기가 그대로 업데이트됩니다. 소스 패키지는 `com.allstar.faithid.*`입니다.

## 동작 개요

- 키오스크는 얼굴 DB 사본을 앱 전용 저장소에 두고 인식합니다. 서버가 끊겨도 인식·등록·출석이 계속되고, 변경은 미전송함에 쌓였다가 연결되면 전송됩니다.
- 서버는 변경마다 리비전을 올리고, 키오스크는 "마지막 리비전 이후 변경분"을 받습니다. 삭제는 묘비(tombstone)로 남아 늦게 접속한 키오스크에도 전달됩니다.
- 키오스크가 보내는 변경은 "등록"과 "샘플 추가"뿐이며 합집합으로 병합되므로, 여러 키오스크의 동시·중복·순서 바뀐 전송이 같은 결과로 수렴합니다.
- 출석은 이름표 출력 버튼을 누를 때 기록됩니다(인쇄 성공 여부와 무관). 이벤트 ID로 중복 없이 한 번만 저장됩니다.
- 키오스크는 스스로 만든 비밀키로 인증하고 서버는 해시만 보관합니다. 승인·해제는 서버 앱 화면에서만 할 수 있습니다.

## 빌드

```bash
./gradlew :kiosk-app:assembleDebug    # kiosk-app/build/outputs/apk/debug/faith_id.apk (arm64 + x86_64)
./gradlew :server-app:assembleDebug   # server-app/build/outputs/apk/debug/faith_db.apk
```

### 모델 파일

`kiosk-app/src/main/assets/models/buffalo_l/w600k_r50.onnx` (InsightFace Buffalo-L 인식 모델, 174 MB)를 직접 넣어야 합니다. 용량 때문에 git에서 제외되어 있습니다.

## 테스트

| 명령 | 내용 |
|---|---|
| `./gradlew :core:test :server-core:test :client-core:test` | JVM 단위 테스트 |
| `./gradlew :kiosk-app:testDebugUnitTest` | 키오스크 흐름(`KioskFlow`), 이름표 배치, 설정 저장 |
| `./gradlew :integration-tests:test` | 다중 키오스크 통합 시나리오 (약 2~3분) |
| `./gradlew :android-support:connectedDebugAndroidTest` | 기기 SQLite에서 저장소·이전 DB 이전 |
| `tools/e2e_server_scenario.sh <serial> [url]` | 기기에서 실행 중인 서버 앱에 6개 모의 키오스크 접속, 승인은 실제 UI로 |
| `tools/e2e_kiosk_flow.sh <serial> [admin-pw]` | 기기의 키오스크·서버 앱으로 등록→인식→출석→본인 확인→이름표만→삭제 전파 (디버그 빌드의 얼굴 주입 사용) |

디버그 빌드에만 있는 `DebugObservationReceiver`로 카메라 없이 합성 얼굴을 넣을 수 있습니다. 릴리스 빌드에는 포함되지 않습니다.
