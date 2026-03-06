import os
import sys

APP_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(APP_DIR, "face_app_data")
PEOPLE_DIR = os.path.join(DATA_DIR, "people")
DB_PATH = os.path.join(DATA_DIR, "face_db.json")

REQUIRED_SIMILAR_COUNT = 5
REGISTER_SIM_THRESHOLD = 0.45
AUTH_SIM_THRESHOLD = 0.40
AUTH_MARGIN_THRESHOLD = 0.10
TOP_K = 3
CAMERA_INDEX = 0

# facial_recognition.py reference: GPU preferred with CPU fallback
MODEL_NAME = "buffalo_l"
DET_SIZE = (320, 320)
USE_GPU = True


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
