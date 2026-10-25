# Allstar Faith ID

출석 키오스크 시스템입니다. 태블릿 키오스크(**Faith ID**)가 얼굴을 인식해 출석을 기록하고 이름표를 출력하며,
별도 태블릿의 **Faith DB Server**가 얼굴 DB·키오스크 승인·출석부를 관리합니다.

```
facial_recognition/
├── android/                 Android 앱과 공용 모듈 (Gradle 프로젝트) → android/README.md
├── desktop-prototype/       초기 Python 데스크톱 프로토타입 (참고용, 더 이상 개발하지 않음)
├── docs/
│   ├── reports/             개발 현황 보고서 (archive/에 2026-03 보고서)
│   └── test-reports/        테스트 보고서
├── assets/
│   ├── branding/            앱 아이콘 원본
│   └── print-templates/     이름표 템플릿 예시
└── local-data/              (git 제외) 실제 인물 데이터·백업. 커밋하지 마세요.
```

## 빠른 시작

```bash
cd android
./gradlew test :integration-tests:test      # 단위 + 다중 키오스크 통합 테스트 (JVM)
./gradlew :kiosk-app:assembleDebug :server-app:assembleDebug
```

얼굴 인식 모델(`w600k_r50.onnx`, 174 MB)은 저장소에 없습니다. `android/README.md`의 "모델 파일"을 참고하세요.
