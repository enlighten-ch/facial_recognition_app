import json
import os
import sys

APP_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(APP_DIR, "face_app_data")
PEOPLE_DIR = os.path.join(DATA_DIR, "people")
DB_PATH = os.path.join(DATA_DIR, "face_db.json")
SETTINGS_FILE_PATH = os.path.join(DATA_DIR, "runtime_settings.json")
DB_JSON_INDENT = 2
DB_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
DB_IMAGE_FILENAME_PATTERN = "{idx:03d}.jpg"
SANITIZE_UNKNOWN_NAME = "unknown"
# Per-person maximum number of stored face embeddings.
# If exceeded, least-similar embeddings to centroid are removed first.
MAX_EMBEDDINGS_PER_PERSON = 100

AUTH_SIM_THRESHOLD = 0.40
AUTH_MARGIN_THRESHOLD = 0.10
TOP_K = 3
CAMERA_INDEX = 0

# facial_recognition.py reference: GPU preferred with CPU fallback
MODEL_NAME = "buffalo_l"
DET_SIZE = (320, 320)
USE_GPU = True
FACE_FORCE_CPU_ENV_VAR = "FACE_FORCE_CPU"
NVIDIA_SMI_TIMEOUT_SEC = 3

# UI text
APP_WINDOW_TITLE = "로컬 얼굴 등록 / 인증 앱"
ERROR_DIALOG_TITLE = "오류"
BACK_BUTTON_TEXT = "뒤로가기"
QUIT_BUTTON_TEXT = "종료"
SETTINGS_BUTTON_TEXT = "⚙"
VIDEO_PANEL_READY_TEXT = "카메라 준비 중..."

MONITOR_PAGE_TITLE = "얼굴 인식"
MONITOR_PAGE_SUBTITLE = "카메라를 보고 있으면 자동으로 등록/인식이 진행됩니다."
WELCOME_TEXT = "환영해요 :) 얼굴 등록 및 인식을 위해 앞을 똑바로 바라봐주세요"
UNKNOWN_TEXT = "등록이 되어있지 않아요! 등록할까요?"
MONITOR_UNREGISTERED_DETAIL_TEXT_FMT = "등록시작 버튼을 누르면 좌/우/위/아래 방향별로 {count}장씩 수집합니다."
MONITOR_REGISTERED_STATUS_FMT = "등록된 사용자입니다: {name}"
MONITOR_REGISTERED_DETAIL_FMT = "1위 점수 {top_score:.3f}, 2위와 차이 {margin:.3f}"
MONITOR_AMBIGUOUS_STATUS_TEXT = "유사한 얼굴이 있습니다. 이름을 선택해 주세요."
MONITOR_AMBIGUOUS_DETAIL_TEXT = "1~3위 후보 중 이름 선택 후 이름 출력하기 버튼이 활성화됩니다."
MONITOR_CANDIDATE_BUTTON_FMT = "{rank}위 선택: {name} (점수 {score:.3f})"
MONITOR_TOP1_BUTTON_FMT = "1위 {name}"
MONITOR_SELECTED_NAME_DETAIL_FMT = "선택됨: {name} / 이름 출력하기 버튼을 눌러주세요."
REGISTER_START_BUTTON_TEXT = "등록시작"
OUTPUT_NAME_BUTTON_TEXT = "이름 출력하기"

REGISTER_PAGE_TITLE = "얼굴 등록"
REGISTER_INTRO_TEXT = "랜드마크 기반으로 좌/우/위/아래 샘플을 자동 수집합니다."
REGISTER_SUBTITLE_FMT = "{intro} (방향당 {count}장, 총 {total}장)"
REGISTER_STATUS_COLLECTING_TEXT = "얼굴 샘플을 수집 중입니다."
REGISTER_FACE_NOT_FOUND_TEXT = "얼굴을 찾지 못했습니다. 화면 중앙으로 와주세요."
REGISTER_FACE_ALIGN_TEXT = "얼굴을 중앙에 맞추고 눈/코/입이 잘 보이게 해주세요."
REGISTER_FACE_OUTSIDE_ELLIPSE_TEXT = "얼굴을 타원 가이드 안으로 맞춰주세요."
REGISTER_DUPLICATE_TEXT = "중복 샘플입니다. 조금 더 다른 각도로 움직여주세요."
REGISTER_OUTLIER_TEXT = "유사도가 낮습니다. 같은 사람이 카메라 앞에 있어야 합니다."
REGISTER_DONE_TEXT = "얼굴등록이 완료되었어요! 이름을 입력해주세요"
REGISTER_PROGRESS_PREFIX = "진행"
REGISTER_DETECTED_DIRECTION_FMT = "현재 감지 방향: {detected_label}"
REGISTER_DETECTED_DIRECTION_FRONT_LABEL = "정면"
REGISTER_SAMPLE_SAVED_FMT = "샘플 저장 완료. {msg}"
REGISTER_SIM_IN_BAND_TEXT_FMT = "유사도 적합 구간 (max={max_sim:.3f})"
REGISTER_DIRECTION_PROGRESS_ITEM_FMT = "{label}:{count}/{target}"

REGISTER_NAME_PAGE_TITLE = "이름 저장"
REGISTER_NAME_PAGE_SUBTITLE = "얼굴등록이 완료되었어요! 이름을 입력해주세요"
NAME_INPUT_PLACEHOLDER_TEXT = "이름 입력"
NAME_SAVE_BUTTON_TEXT = "이름 저장"

OUTPUT_PAGE_TITLE = "이름 출력"
OUTPUT_PAGE_SUBTITLE = "선택된 이름을 출력합니다."
OUTPUT_DONE_TEXT = "이름이 출력되었습니다."
PRINT_SENT_TEXT_FMT = "프린터로 출력 전송됨: {name}"

ERROR_SELECT_NAME_FIRST_TEXT = "먼저 이름을 선택해 주세요."
ERROR_NO_CAMERA_FRAME_TEXT = "카메라 프레임이 아직 없습니다."
ERROR_ENTER_NAME_TEXT = "이름을 입력해 주세요."
ERROR_NOT_ENOUGH_SAMPLES_TEXT = "얼굴 샘플이 아직 충분하지 않습니다."
ERROR_CAMERA_OPEN_TEXT = "카메라를 열 수 없습니다."
ERROR_CAMERA_READ_TEXT = "카메라 프레임을 읽지 못했습니다."
ERROR_PRINT_FAILED_TEXT_FMT = "프린터 출력 실패: {reason}"
ERROR_SETTINGS_SAVE_FAILED_TEXT_FMT = "설정 저장 실패: {reason}"
SETTINGS_WINDOW_TITLE = "설정"
SETTINGS_CATEGORY_LABEL_TEXT = "카테고리"
SETTINGS_SAVED_TEXT = "설정이 저장되었습니다. 프로그램 재시작 후 전체 반영됩니다."

# UI loop tuning
ANALYZE_EVERY_N_FRAMES = 6

# Adaptive embedding update during recognition
ADAPTIVE_UPDATE_ENABLED = True
# If recognized score is in [AUTH_SIM_THRESHOLD, ADAPTIVE_UPDATE_MAX_SIM], add sample to existing person.
ADAPTIVE_UPDATE_MAX_SIM = 0.62
ADAPTIVE_UPDATE_MIN_INTERVAL_FRAMES = 30
ADAPTIVE_UPDATE_MAX_SAMPLES_PER_SESSION = 20
ADAPTIVE_UPDATE_NOTICE_TEXT = "동일 인물의 얼굴 변화가 감지되었습니다. 데이터를 업데이트 합니다."
ADAPTIVE_UPDATE_LOG_PREFIX = "[ADAPTIVE_UPDATE]"

# Printer
PRINT_ENABLED = True
PRINTER_NAME = ""
PRINTER_LP_COMMAND = "lp"
PRINTER_TIMEOUT_SEC = 10
PRINTER_COPIES = 1
PRINTER_JOB_TITLE_PREFIX = "FaceRecognition"
PRINTER_CONTENT_TEMPLATE = "이름: {name}\n출력시각: {timestamp}\n"

# Registration data collection
REGISTER_DIRECTIONS = ("left", "right", "up", "down")
REGISTER_SAMPLES_PER_DIRECTION = 3
REGISTER_CAPTURE_EVERY_N_FRAMES = 4
REGISTER_DIRECTION_ORDER = ("left", "right", "up", "down")
REGISTER_ELLIPSE_AXIS_X_RATIO = 0.22
REGISTER_ELLIPSE_AXIS_Y_RATIO = 0.30
REGISTER_ELLIPSE_CENTER_Y_OFFSET_RATIO = 0.00
REGISTER_ELLIPSE_OUTSIDE_DIM_ALPHA = 0.60

DIRECTION_PROMPTS = {
    "left": "다음 동작: 얼굴을 화면 기준 왼쪽으로 돌려주세요.",
    "right": "다음 동작: 얼굴을 화면 기준 오른쪽으로 돌려주세요.",
    "up": "다음 동작: 턱을 살짝 들고 위를 봐주세요.",
    "down": "다음 동작: 턱을 살짝 내리고 아래를 봐주세요.",
}

DIRECTION_LABELS = {
    "left": "좌",
    "right": "우",
    "up": "위",
    "down": "아래",
}

# Only accept samples in [LOWER, HIGHER] to avoid outliers / duplicates
REGISTER_SIMILARITY_LOWER_BOUNDARY = 0.35
REGISTER_SIMILARITY_HIGHER_BOUNDARY = 0.85

# Direction classification from 5-point landmark
# yaw = (nose_x - eye_center_x) / eye_distance
DIRECTION_YAW_THRESHOLD = 0.10
# pitch = (nose_y - eye_center_y) / (mouth_center_y - eye_center_y)
DIRECTION_PITCH_UP_THRESHOLD = 0.42
DIRECTION_PITCH_DOWN_THRESHOLD = 0.62


def ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PEOPLE_DIR, exist_ok=True)


def configure_qt_plugin_env() -> None:
    if not sys.platform.startswith("linux"):
        return

    cv2_qt_marker = os.path.join("site-packages", "cv2", "qt", "plugins")
    for key in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_PLUGIN_PATH"):
        value = os.environ.get(key)
        if value and cv2_qt_marker in value:
            os.environ.pop(key, None)


RUNTIME_SETTING_CATEGORIES = {
    "Recognition": [
        "AUTH_SIM_THRESHOLD",
        "AUTH_MARGIN_THRESHOLD",
        "TOP_K",
        "ANALYZE_EVERY_N_FRAMES",
        "CAMERA_INDEX",
    ],
    "Registration": [
        "REGISTER_SAMPLES_PER_DIRECTION",
        "REGISTER_CAPTURE_EVERY_N_FRAMES",
        "REGISTER_SIMILARITY_LOWER_BOUNDARY",
        "REGISTER_SIMILARITY_HIGHER_BOUNDARY",
        "REGISTER_ELLIPSE_AXIS_X_RATIO",
        "REGISTER_ELLIPSE_AXIS_Y_RATIO",
        "REGISTER_ELLIPSE_CENTER_Y_OFFSET_RATIO",
        "REGISTER_ELLIPSE_OUTSIDE_DIM_ALPHA",
        "MAX_EMBEDDINGS_PER_PERSON",
    ],
    "AdaptiveUpdate": [
        "ADAPTIVE_UPDATE_ENABLED",
        "ADAPTIVE_UPDATE_MAX_SIM",
        "ADAPTIVE_UPDATE_MIN_INTERVAL_FRAMES",
        "ADAPTIVE_UPDATE_MAX_SAMPLES_PER_SESSION",
    ],
    "Printer": [
        "PRINT_ENABLED",
        "PRINTER_NAME",
    ],
}

RUNTIME_CATEGORY_DISPLAY_NAMES = {
    "Recognition": "인식",
    "Registration": "등록",
    "AdaptiveUpdate": "자동 보정",
    "Printer": "프린터",
}

RUNTIME_SETTING_DISPLAY_NAMES = {
    "AUTH_SIM_THRESHOLD": "인식 최소 유사도 (높을수록 엄격)",
    "AUTH_MARGIN_THRESHOLD": "1,2위 점수 차 최소값",
    "TOP_K": "후보 표시 개수",
    "ANALYZE_EVERY_N_FRAMES": "몇 프레임마다 분석할지",
    "CAMERA_INDEX": "카메라 번호",
    "REGISTER_SAMPLES_PER_DIRECTION": "방향별 수집 장수",
    "REGISTER_CAPTURE_EVERY_N_FRAMES": "몇 프레임마다 샘플 수집할지",
    "REGISTER_SIMILARITY_LOWER_BOUNDARY": "등록 유사도 하한",
    "REGISTER_SIMILARITY_HIGHER_BOUNDARY": "등록 유사도 상한 (중복 기준)",
    "REGISTER_ELLIPSE_AXIS_X_RATIO": "등록 가이드 타원 너비 비율",
    "REGISTER_ELLIPSE_AXIS_Y_RATIO": "등록 가이드 타원 높이 비율",
    "REGISTER_ELLIPSE_CENTER_Y_OFFSET_RATIO": "타원 중심 세로 오프셋 비율",
    "REGISTER_ELLIPSE_OUTSIDE_DIM_ALPHA": "타원 외부 어둡게 처리 강도",
    "MAX_EMBEDDINGS_PER_PERSON": "사람별 최대 임베딩 수",
    "ADAPTIVE_UPDATE_ENABLED": "자동 보정 사용",
    "ADAPTIVE_UPDATE_MAX_SIM": "자동 보정 유사도 상한",
    "ADAPTIVE_UPDATE_MIN_INTERVAL_FRAMES": "자동 보정 최소 프레임 간격",
    "ADAPTIVE_UPDATE_MAX_SAMPLES_PER_SESSION": "세션당 자동 보정 최대 수",
    "PRINT_ENABLED": "프린터 사용",
    "PRINTER_NAME": "프린터 이름 (비우면 기본 프린터)",
}

_RUNTIME_SETTING_KEYS = []
for _keys in RUNTIME_SETTING_CATEGORIES.values():
    for _k in _keys:
        if _k not in _RUNTIME_SETTING_KEYS:
            _RUNTIME_SETTING_KEYS.append(_k)

_RUNTIME_DEFAULTS = {k: globals()[k] for k in _RUNTIME_SETTING_KEYS}


def cast_runtime_value(key: str, raw):
    if key not in _RUNTIME_DEFAULTS:
        raise KeyError(f"Unsupported runtime setting key: {key}")

    default = _RUNTIME_DEFAULTS[key]
    if isinstance(default, bool):
        if isinstance(raw, bool):
            return raw
        s = str(raw).strip().lower()
        if s in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "f", "no", "n", "off"}:
            return False
        raise ValueError(f"Invalid bool value for {key}: {raw}")

    if isinstance(default, int):
        return int(raw)

    if isinstance(default, float):
        return float(raw)

    return str(raw)


def get_runtime_settings_dict() -> dict:
    return {k: globals()[k] for k in _RUNTIME_SETTING_KEYS}


def load_runtime_settings() -> None:
    if not os.path.exists(SETTINGS_FILE_PATH):
        return

    try:
        with open(SETTINGS_FILE_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return

    if not isinstance(raw, dict):
        return

    for key, value in raw.items():
        if key not in _RUNTIME_DEFAULTS:
            continue
        try:
            globals()[key] = cast_runtime_value(key, value)
        except Exception:
            continue


def save_runtime_settings(settings: dict) -> None:
    merged = get_runtime_settings_dict()
    for key, raw in settings.items():
        if key not in _RUNTIME_DEFAULTS:
            continue
        merged[key] = cast_runtime_value(key, raw)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SETTINGS_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    for key, value in merged.items():
        globals()[key] = value


load_runtime_settings()
