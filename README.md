# Allstar Faith ID

출석 키오스크 시스템입니다. 태블릿 키오스크(**Faith ID**)가 얼굴을 인식해 출석을 기록하고 이름표를 출력하며,
별도 태블릿의 **Faith DB Server**가 얼굴 DB·키오스크 승인·출석부를 관리합니다.

```
facial_recognition/
├── android/                     Android 앱과 공용 모듈 (Gradle 프로젝트) → android/README.md
│   ├── core/ server-core/ client-core/ android-support/   공용 모듈
│   ├── kiosk-app/               Faith ID (키오스크)
│   ├── server-app/              Faith DB Server
│   ├── integration-tests/       여러 키오스크 + 서버 통합 시나리오
│   └── tools/                   에뮬레이터, 기기 E2E, 화면 점검, 설치 파일 만들기
├── dist/                        설치 파일 (faith_id.apk, faith_db.apk; git 제외, README만 올라감)
├── docs/
│   ├── user-guide/              사용 안내서 (교인·운영자·관리자용)
│   ├── design/
│   │   ├── registration/        얼굴 등록 로직·화면 설계와 검증
│   │   └── ui-proposals/
│   │       ├── registration/    등록 화면 시안 v1~v11 (v11 채택)
│   │       └── kiosk-screen/    키오스크 운영 화면 시안 (proto.js 공용)
│   ├── reports/                 개발 보고서 (날짜별, archive/에 2026-03 보고서)
│   └── test-reports/            테스트 보고서 (날짜별)
├── assets/
│   ├── branding/                앱 아이콘 원본
│   └── print-templates/         이름표 템플릿 예시
├── desktop-prototype/           초기 Python 데스크톱 프로토타입 (참고용, 더 이상 개발하지 않음)
└── local-data/                  (git 제외) 실제 인물 데이터·태블릿 백업·예전 프로토타입 데이터. 커밋하지 마세요.
```

## 설치 파일

```bash
cd android
tools/make_dist.sh        # dist/faith_id.apk, dist/faith_db.apk, dist/BUILD.txt
```

태블릿에 덮어 설치하면 설정과 데이터는 그대로입니다. 자세한 내용은 `dist/README.md`.

## 빠른 시작 (개발)

```bash
cd android
./gradlew test :integration-tests:test      # 단위 + 다중 키오스크 통합 테스트 (JVM)
./gradlew :kiosk-app:assembleDebug :server-app:assembleDebug
```

얼굴 인식 모델(`w600k_r50.onnx`, 174 MB)은 저장소에 없습니다. `android/README.md`의 "모델 파일"을 참고하세요.
