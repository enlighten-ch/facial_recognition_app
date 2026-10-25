# 안드로이드 얼굴 인식 앱 개발 보고서

## 1. 문서 개요
- 프로젝트: `android_testbed` 클라이언트 앱 (`com.example.facetestbed`)
- 앱 이름: `Allstar Faith ID`
- 산출물 APK: `android_testbed/app/build/outputs/apk/debug/faith_id.apk`
- 기준일: 2026-03-14

## 2. 요구사항 정리
1. 네이티브 Android APK 기반 얼굴 인식 테스트베드 구성
2. 기존 Python 앱 기능/흐름을 Android로 이식
3. Buffalo 모델 기반 얼굴 임베딩 적용
4. 등록/인식 화면 타원형 마스크 적용
5. 얼굴 DB 저장 경로를 설정 가능하게 구성(폴더 트리 선택)
6. 설정 UI를 일반 사용자용 한글로 제공
7. 앱 이름/아이콘 반영, 태블릿(갤럭시 탭 S7+) 화면 대응
8. 영상 좌우 반전(전면 카메라 UX)
9. 프린팅 기능 유지 + Nemonic 프린터 모델 옵션 추가
10. 서버 동기화 클라이언트 기능 추가(인증/승인 기반)
11. 한 번 승인된 클라이언트는 재승인 없이 데이터 송수신
12. 얼굴 등록 시 이름과 함께 `부서` 입력/저장
13. 출력 시 이름 오른쪽에 부서 표시
14. 출력 설정에서 이름/부서 폰트 크기 개별 조절, 부서 오프셋 조절

## 3. 구현 기능 요약

### 3.1 얼굴 인식/등록 엔진
- ML Kit 얼굴 검출 + ONNX Runtime 기반 Buffalo 임베딩 로딩/추론
- Buffalo 모델 로드 실패 시 fallback 경로 유지
- 등록 단계에서 좌/우/상/하 방향 샘플 수집 로직 구현
- 유사도 상/하한 기반 샘플 품질 필터링
- 적응형(Adaptive) 임베딩 업데이트 기능 구현

### 3.2 데이터 구조(Face DB)
- 저장 파일: `face_db.json`
- 현재 스키마:
  - `name` (문자열)
  - `department` (문자열)
  - `centroid` (float 배열)
  - `embeddings` (float 배열의 배열)
- 기존 데이터 호환:
  - `department`가 없는 과거 JSON도 로딩 가능
- DB read/write 동기화 보호(`@Synchronized`) 적용
- JSON export/import 및 서버 동기화용 교체 함수 제공

예시:
```json
{
  "people": [
    {
      "name": "홍길동",
      "department": "청년부",
      "centroid": [0.01, -0.03, 0.22],
      "embeddings": [[0.02, -0.04, 0.23], [0.01, -0.03, 0.21]]
    }
  ]
}
```

### 3.3 UI/UX
- 등록 완료 화면에서 `이름 + 부서` 입력 후 저장
- 인식 후보/확정 표시 시 부서 병기
- 출력 화면에서 이름/부서를 분리 표시(부서 오른쪽 배치)
- 전면 카메라 미러링 반영
- 타원 마스크 오버레이 등록/인식 화면 적용
- 태블릿용 `layout-sw600dp` 별도 리소스 반영

### 3.4 설정 기능
- 저장 위치: 얼굴 DB 폴더 URI 선택(SAF `OpenDocumentTree`)
- 인식/등록/타원/적응형 업데이트 파라미터 조절
- 동기화 설정: 사용 여부, 서버 주소, 클라이언트 ID
- 프린터 설정: 시스템/Nemonic 모델, MAC, 타임아웃, 매수, 템플릿
- 출력 설정 확장:
  - 이름 글자 크기
  - 부서 글자 크기
  - 부서 X 오프셋
  - 부서 Y 오프셋

### 3.5 프린팅
- 기본 프린터 출력 경로 유지
- Nemonic SDK 연동 출력 경로 추가 유지
- 출력 템플릿 토큰:
  - `{name}`
  - `{department}`
- Nemonic 비트맵 렌더링 시 이름/부서 개별 크기 및 오프셋 반영

### 3.6 서버 동기화 클라이언트
- REST + WebSocket 기반 DB 동기화
- 초기 Pull, 변경 시 Push, 서버 브로드캐스트 수신 후 Pull
- 승인 기반 인증:
  - `/api/auth/request`
  - `/api/db`, `/api/db/push`
  - `/ws/db-updates`
- 승인 토큰 재사용(저장된 토큰으로 즉시 통신)
- `401/403` 인증 오류 시에만 토큰 무효화 후 재승인 수행
- 동기화 상태 UI 배너 표시

## 4. 요구사항 대비 구현 매핑
- 요구사항 1~11: 구현 완료
- 요구사항 12~14: 구현 완료 (2026-03-14 반영)

## 5. 주요 파일
- `android_testbed/app/src/main/java/com/example/facetestbed/MainActivity.kt`
- `android_testbed/app/src/main/java/com/example/facetestbed/FaceDatabase.kt`
- `android_testbed/app/src/main/java/com/example/facetestbed/FaceRecognitionEngine.kt`
- `android_testbed/app/src/main/java/com/example/facetestbed/FaceDbSyncClient.kt`
- `android_testbed/app/src/main/java/com/example/facetestbed/AppSettings.kt`
- `android_testbed/app/src/main/java/com/example/facetestbed/SettingsDialog.kt`
- `android_testbed/app/src/main/res/layout/activity_main.xml`
- `android_testbed/app/src/main/res/layout-sw600dp/activity_main.xml`

## 6. 비고
- 현재 서버 연동은 로컬 네트워크/평문 HTTP 기준(개발/사내망 용도)으로 구성됨
- 운영 배포 시 TLS, 인증 강화(토큰 만료/폐기 정책), 감사로그 확장이 필요
