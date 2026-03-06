import sys
from typing import List, Optional, Tuple

import numpy as np

from config import (
    AUTH_MARGIN_THRESHOLD,
    AUTH_SIM_THRESHOLD,
    CAMERA_INDEX,
    REGISTER_SIM_THRESHOLD,
    TOP_K,
    configure_qt_plugin_env,
    ensure_dirs,
)

configure_qt_plugin_env()

from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
import cv2

# OpenCV import can overwrite QT_* env vars to cv2 plugin paths.
configure_qt_plugin_env()

from face_db import FaceDatabase
from face_recognition_engine import FaceEngine, cosine_sim

WELCOME_TEXT = "환영해요 :) 얼굴 등록 및 인식을 위해 앞을 똑바로 바라봐주세요"
UNKNOWN_TEXT = "등록이 되어있지 않아요! 등록할까요?"
REGISTER_TARGET_COUNT = 10
REGISTER_CAPTURE_EVERY_N_FRAMES = 4
ANALYZE_EVERY_N_FRAMES = 6


class UiPage:
    MONITOR = 0
    REGISTER_CAPTURE = 1
    REGISTER_NAME = 2
    OUTPUT = 3


def np_to_qpixmap(frame_bgr: np.ndarray, target_size: Optional[QSize] = None) -> QPixmap:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    pix = QPixmap.fromImage(image)
    if target_size is not None:
        pix = pix.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return pix


class CameraWorker(QThread):
    frame_ready = Signal(object)
    error_signal = Signal(str)

    def __init__(self, camera_index: int = CAMERA_INDEX):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        self.cap = None

    def run(self) -> None:
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            self.error_signal.emit("카메라를 열 수 없습니다.")
            return

        self.running = True
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                self.error_signal.emit("카메라 프레임을 읽지 못했습니다.")
                break
            self.frame_ready.emit(frame.copy())
            self.msleep(30)

        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def stop(self) -> None:
        self.running = False
        self.wait()


class VideoPanel(QLabel):
    def __init__(self):
        super().__init__("카메라 준비 중...")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 420)
        self.setStyleSheet("background:#111;color:#ddd;border-radius:20px;")

    def update_frame(self, frame_bgr: np.ndarray) -> None:
        self.setPixmap(np_to_qpixmap(frame_bgr, self.size()))


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("로컬 얼굴 등록 / 인증 앱")
        self.resize(1180, 760)
        ensure_dirs()

        self.engine = FaceEngine()
        self.db = FaceDatabase()

        self.camera = CameraWorker()
        self.camera.frame_ready.connect(self.on_frame)
        self.camera.error_signal.connect(self.show_error)

        self.current_frame: Optional[np.ndarray] = None
        self.current_bbox: Optional[Tuple[int, int, int, int]] = None

        self.selected_name: Optional[str] = None
        self.latest_rankings: List[Tuple[str, float]] = []
        self.last_monitor_state_key: Optional[str] = None

        self.register_embeddings: List[np.ndarray] = []
        self.register_crops: List[np.ndarray] = []
        self.register_frame_tick = 0
        self.analyze_frame_tick = 0

        self.stack = QStackedWidget()
        self.video_panel_monitor = VideoPanel()
        self.video_panel_register = VideoPanel()

        self.monitor_page = self.build_monitor_page()
        self.register_capture_page = self.build_register_capture_page()
        self.register_name_page = self.build_register_name_page()
        self.output_page = self.build_output_page()

        for page in [
            self.monitor_page,
            self.register_capture_page,
            self.register_name_page,
            self.output_page,
        ]:
            self.stack.addWidget(page)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.addWidget(self.stack)

        self.go_to(UiPage.MONITOR)
        self.apply_monitor_idle_state()
        self.camera.start()

    def shell(self, title: str, subtitle: str, with_back: bool = False):
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(6, 6, 6, 6)

        header = QHBoxLayout()
        text_box = QVBoxLayout()
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size:28px;font-weight:700;")
        subtitle_label = QLabel(subtitle)
        subtitle_label.setWordWrap(True)
        subtitle_label.setStyleSheet("font-size:14px;color:#666;")
        text_box.addWidget(title_label)
        text_box.addWidget(subtitle_label)
        header.addLayout(text_box)
        header.addStretch(1)

        if with_back:
            back_btn = QPushButton("뒤로가기")
            back_btn.setMinimumHeight(42)
            back_btn.clicked.connect(self.go_initial_state)
            back_btn.setStyleSheet("QPushButton{padding:8px 16px;border-radius:14px;background:#fff;border:1px solid #ddd;} QPushButton:hover{background:#f4f4f4;}")
            header.addWidget(back_btn)

        layout.addLayout(header)
        layout.addSpacing(10)
        return wrapper, layout

    def card_style(self):
        return "background:#fff;border:1px solid #e7e7e7;border-radius:22px;"

    def primary_btn_style(self):
        return "QPushButton{background:#111;color:#fff;border-radius:16px;font-size:16px;font-weight:600;} QPushButton:hover{background:#222;} QPushButton:disabled{background:#999;color:#eee;}"

    def build_monitor_page(self):
        page, layout = self.shell("얼굴 인식", "카메라를 보고 있으면 자동으로 등록/인식이 진행됩니다.", with_back=False)

        body = QHBoxLayout()
        body.setSpacing(14)

        left = QFrame()
        left.setStyleSheet(self.card_style())
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(16, 16, 16, 16)
        left_layout.addWidget(self.video_panel_monitor, 1)

        right = QFrame()
        right.setStyleSheet(self.card_style())
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(20, 20, 20, 20)

        self.monitor_status_label = QLabel(WELCOME_TEXT)
        self.monitor_status_label.setWordWrap(True)
        self.monitor_status_label.setStyleSheet("font-size:18px;font-weight:600;")

        self.monitor_detail_label = QLabel("")
        self.monitor_detail_label.setWordWrap(True)
        self.monitor_detail_label.setStyleSheet("font-size:14px;color:#666;")

        self.register_start_btn = QPushButton("등록시작")
        self.register_start_btn.setMinimumHeight(50)
        self.register_start_btn.setStyleSheet(self.primary_btn_style())
        self.register_start_btn.clicked.connect(self.start_auto_registration)

        self.candidate_buttons_layout = QVBoxLayout()

        self.output_name_btn = QPushButton("이름 출력하기")
        self.output_name_btn.setMinimumHeight(50)
        self.output_name_btn.setStyleSheet(self.primary_btn_style())
        self.output_name_btn.clicked.connect(self.output_selected_name)

        right_layout.addWidget(self.monitor_status_label)
        right_layout.addWidget(self.monitor_detail_label)
        right_layout.addSpacing(10)
        right_layout.addWidget(self.register_start_btn)
        right_layout.addLayout(self.candidate_buttons_layout)
        right_layout.addStretch(1)
        right_layout.addWidget(self.output_name_btn)

        body.addWidget(left, 3)
        body.addWidget(right, 2)
        layout.addLayout(body, 1)
        return page

    def build_register_capture_page(self):
        page, layout = self.shell("얼굴 등록", "등록시작 후 얼굴 샘플 10개를 자동으로 수집합니다.", with_back=True)

        body = QHBoxLayout()
        body.setSpacing(14)

        left = QFrame()
        left.setStyleSheet(self.card_style())
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(16, 16, 16, 16)
        left_layout.addWidget(self.video_panel_register, 1)

        right = QFrame()
        right.setStyleSheet(self.card_style())
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(20, 20, 20, 20)

        self.register_status_label = QLabel("얼굴 샘플을 수집 중입니다.")
        self.register_status_label.setWordWrap(True)
        self.register_status_label.setStyleSheet("font-size:16px;font-weight:600;")

        self.register_progress = QProgressBar()
        self.register_progress.setRange(0, REGISTER_TARGET_COUNT)
        self.register_progress.setValue(0)

        self.register_count_label = QLabel(f"0 / {REGISTER_TARGET_COUNT}")
        self.register_count_label.setStyleSheet("font-size:20px;font-weight:700;")

        right_layout.addWidget(self.register_status_label)
        right_layout.addWidget(self.register_progress)
        right_layout.addWidget(self.register_count_label)
        right_layout.addStretch(1)

        body.addWidget(left, 3)
        body.addWidget(right, 2)
        layout.addLayout(body, 1)
        return page

    def build_register_name_page(self):
        page, layout = self.shell("이름 저장", "얼굴등록이 완료되었어요! 이름을 입력해주세요", with_back=True)

        card = QFrame()
        card.setStyleSheet(self.card_style())
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 20, 20, 20)

        self.register_complete_label = QLabel("얼굴등록이 완료되었어요! 이름을 입력해주세요")
        self.register_complete_label.setWordWrap(True)
        self.register_complete_label.setStyleSheet("font-size:24px;font-weight:700;")

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("이름 입력")
        self.name_input.setMinimumHeight(48)
        self.name_input.setStyleSheet("QLineEdit{font-size:16px;padding:10px 14px;border:1px solid #ddd;border-radius:14px;}")

        save_btn = QPushButton("이름 저장")
        save_btn.setMinimumHeight(50)
        save_btn.setStyleSheet(self.primary_btn_style())
        save_btn.clicked.connect(self.save_registration)

        card_layout.addWidget(self.register_complete_label)
        card_layout.addSpacing(10)
        card_layout.addWidget(self.name_input)
        card_layout.addWidget(save_btn)
        card_layout.addStretch(1)

        layout.addWidget(card, 1)
        return page

    def build_output_page(self):
        page, layout = self.shell("이름 출력", "선택된 이름을 출력합니다.", with_back=True)

        card = QFrame()
        card.setStyleSheet(self.card_style())
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 30, 20, 30)

        self.output_label = QLabel("")
        self.output_label.setAlignment(Qt.AlignCenter)
        self.output_label.setStyleSheet("font-size:36px;font-weight:700;")

        self.output_sub_label = QLabel("")
        self.output_sub_label.setAlignment(Qt.AlignCenter)
        self.output_sub_label.setStyleSheet("font-size:16px;color:#666;")

        card_layout.addStretch(1)
        card_layout.addWidget(self.output_label)
        card_layout.addWidget(self.output_sub_label)
        card_layout.addStretch(1)

        layout.addWidget(card, 1)
        return page

    def go_to(self, index: int):
        self.stack.setCurrentIndex(index)

    def go_initial_state(self):
        self.register_embeddings = []
        self.register_crops = []
        self.selected_name = None
        self.latest_rankings = []
        self.last_monitor_state_key = None
        self.name_input.clear()
        self.apply_monitor_idle_state()
        self.go_to(UiPage.MONITOR)

    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                self.clear_layout(child_layout)

    def set_register_count(self):
        count = len(self.register_embeddings)
        self.register_progress.setValue(min(count, REGISTER_TARGET_COUNT))
        self.register_count_label.setText(f"{count} / {REGISTER_TARGET_COUNT}")

    def set_output_name(self, name: str):
        self.output_label.setText(f"{name}")
        self.output_sub_label.setText("이름이 출력되었습니다.")
        self.go_to(UiPage.OUTPUT)

    def apply_monitor_idle_state(self):
        self.monitor_status_label.setText(WELCOME_TEXT)
        self.monitor_detail_label.setText("")
        self.register_start_btn.setVisible(False)
        self.register_start_btn.setEnabled(False)
        self.output_name_btn.setEnabled(False)
        self.clear_layout(self.candidate_buttons_layout)

    def apply_unregistered_state(self):
        self.monitor_status_label.setText(UNKNOWN_TEXT)
        self.monitor_detail_label.setText("등록시작 버튼을 누르면 얼굴 샘플 10개를 자동으로 수집합니다.")
        self.register_start_btn.setVisible(True)
        self.register_start_btn.setEnabled(True)
        self.output_name_btn.setEnabled(False)
        self.clear_layout(self.candidate_buttons_layout)

    def apply_registered_clear_state(self, top_name: str, top_score: float, margin: float):
        self.monitor_status_label.setText(f"등록된 사용자입니다: {top_name}")
        self.monitor_detail_label.setText(f"1위 점수 {top_score:.3f}, 2위와 차이 {margin:.3f}")
        self.register_start_btn.setVisible(False)
        self.register_start_btn.setEnabled(False)
        self.clear_layout(self.candidate_buttons_layout)

        candidate = QPushButton(f"1위 {top_name}")
        candidate.setEnabled(False)
        candidate.setMinimumHeight(48)
        candidate.setStyleSheet("QPushButton{background:#f4f4f4;color:#333;border-radius:14px;border:1px solid #ddd;font-size:15px;}")
        self.candidate_buttons_layout.addWidget(candidate)

        self.selected_name = top_name
        self.output_name_btn.setEnabled(True)

    def apply_registered_ambiguous_state(self, rankings: List[Tuple[str, float]]):
        self.monitor_status_label.setText("유사한 얼굴이 있습니다. 이름을 선택해 주세요.")
        self.monitor_detail_label.setText("1~3위 후보 중 이름 선택 후 이름 출력하기 버튼이 활성화됩니다.")
        self.register_start_btn.setVisible(False)
        self.register_start_btn.setEnabled(False)
        self.clear_layout(self.candidate_buttons_layout)

        self.selected_name = None
        self.output_name_btn.setEnabled(False)

        for idx, (name, score) in enumerate(rankings, start=1):
            btn = QPushButton(f"{idx}위 선택: {name} (점수 {score:.3f})")
            btn.setMinimumHeight(48)
            btn.setStyleSheet(self.primary_btn_style())
            btn.clicked.connect(lambda checked=False, n=name: self.choose_candidate_name(n))
            self.candidate_buttons_layout.addWidget(btn)

    def choose_candidate_name(self, name: str):
        self.selected_name = name
        self.monitor_detail_label.setText(f"선택됨: {name} / 이름 출력하기 버튼을 눌러주세요.")
        self.output_name_btn.setEnabled(True)

    def output_selected_name(self):
        if not self.selected_name:
            self.show_error("먼저 이름을 선택해 주세요.")
            return
        self.set_output_name(self.selected_name)

    def start_auto_registration(self):
        if self.current_frame is None:
            self.show_error("카메라 프레임이 아직 없습니다.")
            return

        self.register_embeddings = []
        self.register_crops = []
        self.register_frame_tick = 0
        self.set_register_count()
        self.register_status_label.setText("얼굴 샘플을 자동으로 수집 중입니다. 정면을 바라봐주세요.")
        self.go_to(UiPage.REGISTER_CAPTURE)

    def handle_register_capture(self, frame: np.ndarray):
        self.register_frame_tick += 1
        if self.register_frame_tick % REGISTER_CAPTURE_EVERY_N_FRAMES != 0:
            return

        emb, crop, _ = self.engine.embedding_and_crop(frame)
        if emb is None:
            self.register_status_label.setText("얼굴을 찾지 못했습니다. 정면을 바라봐주세요.")
            return

        if not self.register_embeddings:
            self.register_embeddings.append(emb)
            self.register_crops.append(crop)
            self.set_register_count()
            self.register_status_label.setText("샘플 수집 시작. 계속 정면을 바라봐주세요.")
            return

        base = self.register_embeddings[0]
        sim = cosine_sim(base, emb)
        if sim < REGISTER_SIM_THRESHOLD:
            self.register_status_label.setText(f"동일 인물 유사도 부족({sim:.3f}). 같은 인물이 계속 보이게 유지해 주세요.")
            return

        self.register_embeddings.append(emb)
        self.register_crops.append(crop)
        self.set_register_count()
        self.register_status_label.setText(f"샘플 수집 중... ({len(self.register_embeddings)}/{REGISTER_TARGET_COUNT})")

        if len(self.register_embeddings) >= REGISTER_TARGET_COUNT:
            self.register_status_label.setText("얼굴등록이 완료되었어요! 이름을 입력해주세요")
            self.go_to(UiPage.REGISTER_NAME)

    def save_registration(self):
        name = self.name_input.text().strip()
        if not name:
            self.show_error("이름을 입력해 주세요.")
            return

        if len(self.register_embeddings) < REGISTER_TARGET_COUNT:
            self.show_error("얼굴 샘플이 아직 충분하지 않습니다.")
            return

        self.db.upsert_person(name, self.register_embeddings, self.register_crops)
        self.db.load()
        self.go_initial_state()

    def apply_monitor_state_from_frame(self, frame: np.ndarray):
        emb, _, _ = self.engine.embedding_and_crop(frame)
        if emb is None:
            state_key = "idle"
            if state_key != self.last_monitor_state_key:
                self.apply_monitor_idle_state()
                self.last_monitor_state_key = state_key
            return

        rankings = self.db.rank(emb, TOP_K)
        if not rankings:
            state_key = "unknown:none"
            if state_key != self.last_monitor_state_key:
                self.apply_unregistered_state()
                self.last_monitor_state_key = state_key
            return

        top_name, top_score = rankings[0]
        if top_score < AUTH_SIM_THRESHOLD:
            state_key = f"unknown:score:{round(top_score, 3)}"
            if state_key != self.last_monitor_state_key:
                self.apply_unregistered_state()
                self.last_monitor_state_key = state_key
            return

        second_score = rankings[1][1] if len(rankings) > 1 else -1.0
        margin = top_score - second_score if second_score >= 0 else top_score

        if margin >= AUTH_MARGIN_THRESHOLD:
            state_key = f"known:clear:{top_name}:{round(top_score, 3)}:{round(margin, 3)}"
            if state_key != self.last_monitor_state_key:
                self.apply_registered_clear_state(top_name, top_score, margin)
                self.last_monitor_state_key = state_key
            return

        top3 = rankings[:3]
        state_key = "known:ambiguous:" + "|".join([f"{n}:{round(s, 3)}" for n, s in top3])
        if state_key != self.last_monitor_state_key:
            self.apply_registered_ambiguous_state(top3)
            self.last_monitor_state_key = state_key

    def on_frame(self, frame: np.ndarray):
        self.current_frame = frame.copy()
        display, bbox = self.engine.detect_and_draw_bbox(frame)
        self.current_bbox = bbox

        page = self.stack.currentIndex()
        if page == UiPage.MONITOR:
            self.video_panel_monitor.update_frame(display)
            self.analyze_frame_tick += 1
            if self.analyze_frame_tick % ANALYZE_EVERY_N_FRAMES == 0:
                self.apply_monitor_state_from_frame(frame)
        elif page == UiPage.REGISTER_CAPTURE:
            self.video_panel_register.update_frame(display)
            self.handle_register_capture(frame)

    def show_error(self, message: str):
        QMessageBox.critical(self, "오류", message)

    def closeEvent(self, event):
        try:
            self.camera.stop()
        except Exception:
            pass
        super().closeEvent(event)


def run_app() -> int:
    app = QApplication(sys.argv)
    app.setStyleSheet(
        """
    QWidget { background: #f6f7fb; color: #111; font-family: Arial, Helvetica, sans-serif; }
    QProgressBar { background:#ededed; border:none; border-radius:10px; text-align:center; min-height:18px; }
    QProgressBar::chunk { background:#111; border-radius:10px; }
    """
    )
    win = MainWindow()
    win.show()
    return app.exec()
