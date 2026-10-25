# 안드로이드 Face DB 서버 앱 개발 보고서

## 1. 문서 개요
- 프로젝트: `android_testbed` 서버 앱 모듈 (`com.example.facedbserver`)
- 앱 이름: `Faith DB Server`
- 산출물 APK: `android_testbed/dbserver/build/outputs/apk/debug/dbserver-debug.apk`
- 기준일: 2026-03-14

## 2. 요구사항 정리
1. Android에서 별도 앱 형태의 DB 서버 구동
2. 다중 클라이언트 동시 접속 고려
3. 클라이언트 승인 기반 인증 체계
4. 한 번 승인된 클라이언트 재승인 없이 지속 통신
5. DB 변경 시 연결된 클라이언트에게 변경사항 브로드캐스트
6. 서버 UI에서 DB를 테이블 형태로 조회
7. 서버 UI에서 연결된 클라이언트 ID/연결상태 표시
8. 서버 앱 첫 화면 로그인(ID/PW)
9. 서버 앱 설정 화면 제공 + 설정 파일 저장/재로드(`config.yaml`)

## 3. 구현 기능 요약

### 3.1 서버 엔진(HTTP/WS)
- Ktor CIO 기반 서버 실행/중지
- 기본 포트: 18080(설정으로 변경 가능)
- 주요 엔드포인트:
  - `GET /api/health`
  - `POST /api/auth/request`
  - `GET /api/auth/status`
  - `POST /api/auth/approve`
  - `POST /api/auth/approve-all`
  - `GET /api/auth/pending`
  - `GET /api/auth/approved`
  - `GET /api/db`
  - `POST /api/db/push`
  - `WS /ws/db-updates`

### 3.2 인증/승인 구조
- 신규 클라이언트는 승인 요청 후 `pending`
- 관리자가 개별/전체 승인 시 토큰 발급
- 승인 토큰 영속 저장(`auth_state.json`)
- 서버 재시작 후에도 승인 상태 복구
- 승인된 클라이언트는 재승인 없이 토큰으로 지속 통신

### 3.3 동기화/브로드캐스트
- 클라이언트가 `POST /api/db/push`로 DB 업로드
- 서버가 revision 증가 후 저장
- `WS /ws/db-updates`로 모든 세션에 변경 이벤트 전송
- 클라이언트는 이벤트 수신 시 DB 재동기화 수행

### 3.4 서버 UI
- 로그인 화면(ID/PW) 선행
- 로그인 성공 후 관리 대시보드 노출
- 대시보드 기능:
  - 서버 시작/중지
  - 엔드포인트 표시(IP:Port)
  - 승인 대기/승인 완료 목록
  - 개별 승인/전체 승인
  - 클라이언트 연결 상태 표시(ID, 상태, 세션 수)
  - DB 테이블 표시(이름/부서/임베딩 수)
  - 서버 로그 패널

### 3.5 설정 관리(config 파일)
- 설정 저장소: `ServerConfigStore`
- 파일 경로: 앱 내부 `files/face_db_server/config.yaml`
- 저장 항목:
  - `admin_id`
  - `admin_password`
  - `default_port`
- 동작:
  - 첫 실행 시 기본 config 자동 생성
  - 설정창 저장 시 파일 갱신
  - 앱 시작 시 자동 로드

예시:
```yaml
admin_id: "admin"
admin_password: "1234"
default_port: 18080
```

## 4. 요구사항 대비 구현 매핑
- 요구사항 1~9: 구현 완료

## 5. 주요 파일
- `android_testbed/dbserver/src/main/java/com/example/facedbserver/FaceDbServerEngine.kt`
- `android_testbed/dbserver/src/main/java/com/example/facedbserver/MainActivity.kt`
- `android_testbed/dbserver/src/main/java/com/example/facedbserver/ServerConfigStore.kt`
- `android_testbed/dbserver/src/main/res/layout/activity_main.xml`

## 6. 비고
- 현재 로그인/설정은 로컬 파일 기반 단순 인증 구조
- 운영 환경 권장 보완:
  - 비밀번호 해시 저장
  - TLS(HTTPS/WSS) 적용
  - 토큰 만료/폐기 정책
  - 관리행위 감사로그
