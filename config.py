import os
import sys

APP_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(APP_DIR, "face_app_data")
PEOPLE_DIR = os.path.join(DATA_DIR, "people")
DB_PATH = os.path.join(DATA_DIR, "face_db.json")
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

ERROR_SELECT_NAME_FIRST_TEXT = "먼저 이름을 선택해 주세요."
ERROR_NO_CAMERA_FRAME_TEXT = "카메라 프레임이 아직 없습니다."
ERROR_ENTER_NAME_TEXT = "이름을 입력해 주세요."
ERROR_NOT_ENOUGH_SAMPLES_TEXT = "얼굴 샘플이 아직 충분하지 않습니다."
ERROR_CAMERA_OPEN_TEXT = "카메라를 열 수 없습니다."
ERROR_CAMERA_READ_TEXT = "카메라 프레임을 읽지 못했습니다."

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

# Registration data collection
REGISTER_DIRECTIONS = ("left", "right", "up", "down")
REGISTER_SAMPLES_PER_DIRECTION = 3
REGISTER_CAPTURE_EVERY_N_FRAMES = 4
REGISTER_DIRECTION_ORDER = ("left", "right", "up", "down")

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
