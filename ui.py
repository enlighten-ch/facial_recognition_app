import sys
from typing import List, Optional, Tuple

import numpy as np

from config import (
    ADAPTIVE_UPDATE_ENABLED,
    ADAPTIVE_UPDATE_MAX_SAMPLES_PER_SESSION,
    ADAPTIVE_UPDATE_MAX_SIM,
    ADAPTIVE_UPDATE_MIN_INTERVAL_FRAMES,
    ADAPTIVE_UPDATE_NOTICE_TEXT,
    ANALYZE_EVERY_N_FRAMES,
    APP_WINDOW_TITLE,
    BACK_BUTTON_TEXT,
    AUTH_MARGIN_THRESHOLD,
    AUTH_SIM_THRESHOLD,
    CAMERA_INDEX,
    DIRECTION_LABELS,
    DIRECTION_PROMPTS,
    ERROR_CAMERA_OPEN_TEXT,
    ERROR_CAMERA_READ_TEXT,
    ERROR_DIALOG_TITLE,
    ERROR_ENTER_NAME_TEXT,
    ERROR_NOT_ENOUGH_SAMPLES_TEXT,
    ERROR_NO_CAMERA_FRAME_TEXT,
    ERROR_SELECT_NAME_FIRST_TEXT,
    MONITOR_AMBIGUOUS_DETAIL_TEXT,
    MONITOR_AMBIGUOUS_STATUS_TEXT,
    MONITOR_CANDIDATE_BUTTON_FMT,
    MONITOR_PAGE_SUBTITLE,
    MONITOR_PAGE_TITLE,
    MONITOR_REGISTERED_DETAIL_FMT,
    MONITOR_REGISTERED_STATUS_FMT,
    MONITOR_SELECTED_NAME_DETAIL_FMT,
    MONITOR_TOP1_BUTTON_FMT,
    MONITOR_UNREGISTERED_DETAIL_TEXT_FMT,
    NAME_INPUT_PLACEHOLDER_TEXT,
    NAME_SAVE_BUTTON_TEXT,
    OUTPUT_DONE_TEXT,
    OUTPUT_NAME_BUTTON_TEXT,
    OUTPUT_PAGE_SUBTITLE,
    OUTPUT_PAGE_TITLE,
    PRINT_ENABLED,
    QUIT_BUTTON_TEXT,
    REGISTER_CAPTURE_EVERY_N_FRAMES,
    REGISTER_DIRECTIONS,
    REGISTER_DIRECTION_ORDER,
    REGISTER_DETECTED_DIRECTION_FMT,
    REGISTER_DETECTED_DIRECTION_FRONT_LABEL,
    REGISTER_DONE_TEXT,
    REGISTER_DIRECTION_PROGRESS_ITEM_FMT,
    REGISTER_FACE_ALIGN_TEXT,
    REGISTER_FACE_NOT_FOUND_TEXT,
    REGISTER_DUPLICATE_TEXT,
    REGISTER_INTRO_TEXT,
    REGISTER_NAME_PAGE_SUBTITLE,
    REGISTER_NAME_PAGE_TITLE,
    REGISTER_PAGE_TITLE,
    REGISTER_PROGRESS_PREFIX,
    REGISTER_SAMPLE_SAVED_FMT,
    REGISTER_OUTLIER_TEXT,
    REGISTER_SUBTITLE_FMT,
    REGISTER_SIM_IN_BAND_TEXT_FMT,
    REGISTER_SAMPLES_PER_DIRECTION,
    REGISTER_START_BUTTON_TEXT,
    REGISTER_STATUS_COLLECTING_TEXT,
    REGISTER_SIMILARITY_HIGHER_BOUNDARY,
    REGISTER_SIMILARITY_LOWER_BOUNDARY,
    TOP_K,
    UNKNOWN_TEXT,
    VIDEO_PANEL_READY_TEXT,
    WELCOME_TEXT,
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
from printer_service import PrinterService


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
            self.error_signal.emit(ERROR_CAMERA_OPEN_TEXT)
            return

        self.running = True
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                self.error_signal.emit(ERROR_CAMERA_READ_TEXT)
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
        super().__init__(VIDEO_PANEL_READY_TEXT)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 420)
        self.setStyleSheet("background:#111;color:#ddd;border-radius:20px;")

    def update_frame(self, frame_bgr: np.ndarray) -> None:
        self.setPixmap(np_to_qpixmap(frame_bgr, self.size()))


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_WINDOW_TITLE)
        self.resize(1180, 760)
        ensure_dirs()

        self.engine = FaceEngine()
        self.db = FaceDatabase()
        self.printer = PrinterService()

        self.camera = CameraWorker()
        self.camera.frame_ready.connect(self.on_frame)
        self.camera.error_signal.connect(self.show_error)

        self.current_frame: Optional[np.ndarray] = None
        self.current_bbox: Optional[Tuple[int, int, int, int]] = None

        self.selected_name: Optional[str] = None
        self.latest_rankings: List[Tuple[str, float]] = []
        self.last_monitor_state_key: Optional[str] = None

        self.register_direction_embeddings = {d: [] for d in REGISTER_DIRECTIONS}
        self.register_direction_crops = {d: [] for d in REGISTER_DIRECTIONS}
        self.register_total_target = REGISTER_SAMPLES_PER_DIRECTION * len(REGISTER_DIRECTIONS)
        self.register_frame_tick = 0
        self.analyze_frame_tick = 0
        self.last_adaptive_update_tick = -10**9
        self.adaptive_update_count = 0

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
            back_btn = QPushButton(BACK_BUTTON_TEXT)
            back_btn.setMinimumHeight(42)
            back_btn.clicked.connect(self.go_initial_state)
            back_btn.setStyleSheet("QPushButton{padding:8px 16px;border-radius:14px;background:#fff;border:1px solid #ddd;} QPushButton:hover{background:#f4f4f4;}")
            header.addWidget(back_btn)

        quit_btn = QPushButton(QUIT_BUTTON_TEXT)
        quit_btn.setMinimumHeight(42)
        quit_btn.clicked.connect(self.close_app)
        quit_btn.setStyleSheet("QPushButton{padding:8px 16px;border-radius:14px;background:#fff;border:1px solid #ddd;} QPushButton:hover{background:#f4f4f4;}")
        header.addWidget(quit_btn)

        layout.addLayout(header)
        layout.addSpacing(10)
        return wrapper, layout

    def card_style(self):
        return "background:#fff;border:1px solid #e7e7e7;border-radius:22px;"

    def primary_btn_style(self):
        return "QPushButton{background:#111;color:#fff;border-radius:16px;font-size:16px;font-weight:600;} QPushButton:hover{background:#222;} QPushButton:disabled{background:#999;color:#eee;}"

    def build_monitor_page(self):
        page, layout = self.shell(MONITOR_PAGE_TITLE, MONITOR_PAGE_SUBTITLE, with_back=False)

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

        self.register_start_btn = QPushButton(REGISTER_START_BUTTON_TEXT)
        self.register_start_btn.setMinimumHeight(50)
        self.register_start_btn.setStyleSheet(self.primary_btn_style())
        self.register_start_btn.clicked.connect(self.start_auto_registration)

        self.candidate_buttons_layout = QVBoxLayout()

        self.output_name_btn = QPushButton(OUTPUT_NAME_BUTTON_TEXT)
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
        subtitle = REGISTER_SUBTITLE_FMT.format(
            intro=REGISTER_INTRO_TEXT,
            count=REGISTER_SAMPLES_PER_DIRECTION,
            total=self.register_total_target,
        )
        page, layout = self.shell(REGISTER_PAGE_TITLE, subtitle, with_back=True)

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

        self.register_status_label = QLabel(REGISTER_STATUS_COLLECTING_TEXT)
        self.register_status_label.setWordWrap(True)
        self.register_status_label.setStyleSheet("font-size:16px;font-weight:600;")

        self.register_progress = QProgressBar()
        self.register_progress.setRange(0, self.register_total_target)
        self.register_progress.setValue(0)

        self.register_count_label = QLabel(f"0 / {self.register_total_target}")
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
        page, layout = self.shell(REGISTER_NAME_PAGE_TITLE, REGISTER_NAME_PAGE_SUBTITLE, with_back=True)

        card = QFrame()
        card.setStyleSheet(self.card_style())
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 20, 20, 20)

        self.register_complete_label = QLabel(REGISTER_NAME_PAGE_SUBTITLE)
        self.register_complete_label.setWordWrap(True)
        self.register_complete_label.setStyleSheet("font-size:24px;font-weight:700;")

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText(NAME_INPUT_PLACEHOLDER_TEXT)
        self.name_input.setMinimumHeight(48)
        self.name_input.setStyleSheet("QLineEdit{font-size:16px;padding:10px 14px;border:1px solid #ddd;border-radius:14px;}")

        save_btn = QPushButton(NAME_SAVE_BUTTON_TEXT)
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
        page, layout = self.shell(OUTPUT_PAGE_TITLE, OUTPUT_PAGE_SUBTITLE, with_back=True)

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
        self.reset_registration_state()
        self.selected_name = None
        self.latest_rankings = []
        self.last_monitor_state_key = None
        self.last_adaptive_update_tick = -10**9
        self.adaptive_update_count = 0
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
        count = self.total_registered_samples()
        self.register_progress.setValue(min(count, self.register_total_target))
        self.register_count_label.setText(f"{count} / {self.register_total_target}")

    def reset_registration_state(self):
        self.register_direction_embeddings = {d: [] for d in REGISTER_DIRECTIONS}
        self.register_direction_crops = {d: [] for d in REGISTER_DIRECTIONS}
        self.register_frame_tick = 0
        self.set_register_count()

    def total_registered_samples(self) -> int:
        return sum(len(v) for v in self.register_direction_embeddings.values())

    def next_target_direction(self) -> Optional[str]:
        for direction in REGISTER_DIRECTION_ORDER:
            if len(self.register_direction_embeddings[direction]) < REGISTER_SAMPLES_PER_DIRECTION:
                return direction
        return None

    def summarize_direction_progress(self) -> str:
        parts = []
        for direction in REGISTER_DIRECTION_ORDER:
            count = len(self.register_direction_embeddings[direction])
            label = DIRECTION_LABELS.get(direction, direction)
            parts.append(
                REGISTER_DIRECTION_PROGRESS_ITEM_FMT.format(
                    label=label,
                    count=count,
                    target=REGISTER_SAMPLES_PER_DIRECTION,
                )
            )
        return " | ".join(parts)

    def all_registered_embeddings(self) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for direction in REGISTER_DIRECTION_ORDER:
            out.extend(self.register_direction_embeddings[direction])
        return out

    def all_registered_crops(self) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for direction in REGISTER_DIRECTION_ORDER:
            out.extend(self.register_direction_crops[direction])
        return out

    def set_output_name(self, name: str):
        self.output_label.setText(f"{name}")
        self.output_sub_label.setText(OUTPUT_DONE_TEXT)
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
        self.monitor_detail_label.setText(MONITOR_UNREGISTERED_DETAIL_TEXT_FMT.format(count=REGISTER_SAMPLES_PER_DIRECTION))
        self.register_start_btn.setVisible(True)
        self.register_start_btn.setEnabled(True)
        self.output_name_btn.setEnabled(False)
        self.clear_layout(self.candidate_buttons_layout)

    def apply_registered_clear_state(self, top_name: str, top_score: float, margin: float):
        self.monitor_status_label.setText(MONITOR_REGISTERED_STATUS_FMT.format(name=top_name))
        self.monitor_detail_label.setText(MONITOR_REGISTERED_DETAIL_FMT.format(top_score=top_score, margin=margin))
        self.register_start_btn.setVisible(False)
        self.register_start_btn.setEnabled(False)
        self.clear_layout(self.candidate_buttons_layout)

        candidate = QPushButton(MONITOR_TOP1_BUTTON_FMT.format(name=top_name))
        candidate.setEnabled(False)
        candidate.setMinimumHeight(48)
        candidate.setStyleSheet("QPushButton{background:#f4f4f4;color:#333;border-radius:14px;border:1px solid #ddd;font-size:15px;}")
        self.candidate_buttons_layout.addWidget(candidate)

        self.selected_name = top_name
        self.output_name_btn.setEnabled(True)

    def apply_registered_ambiguous_state(self, rankings: List[Tuple[str, float]]):
        self.monitor_status_label.setText(MONITOR_AMBIGUOUS_STATUS_TEXT)
        self.monitor_detail_label.setText(MONITOR_AMBIGUOUS_DETAIL_TEXT)
        self.register_start_btn.setVisible(False)
        self.register_start_btn.setEnabled(False)
        self.clear_layout(self.candidate_buttons_layout)

        self.selected_name = None
        self.output_name_btn.setEnabled(False)

        for idx, (name, score) in enumerate(rankings, start=1):
            btn = QPushButton(MONITOR_CANDIDATE_BUTTON_FMT.format(rank=idx, name=name, score=score))
            btn.setMinimumHeight(48)
            btn.setStyleSheet(self.primary_btn_style())
            btn.clicked.connect(lambda checked=False, n=name: self.choose_candidate_name(n))
            self.candidate_buttons_layout.addWidget(btn)

    def choose_candidate_name(self, name: str):
        self.selected_name = name
        self.monitor_detail_label.setText(MONITOR_SELECTED_NAME_DETAIL_FMT.format(name=name))
        self.output_name_btn.setEnabled(True)

    def output_selected_name(self):
        if not self.selected_name:
            self.show_error(ERROR_SELECT_NAME_FIRST_TEXT)
            return
        self.set_output_name(self.selected_name)
        if PRINT_ENABLED:
            ok, message = self.printer.print_name(self.selected_name)
            if ok:
                if message:
                    self.output_sub_label.setText(message)
            else:
                self.show_error(message)
                self.output_sub_label.setText(message)

    def start_auto_registration(self):
        if self.current_frame is None:
            self.show_error(ERROR_NO_CAMERA_FRAME_TEXT)
            return

        self.reset_registration_state()
        first_dir = self.next_target_direction()
        first_prompt = DIRECTION_PROMPTS.get(first_dir, REGISTER_FACE_ALIGN_TEXT) if first_dir else ""
        self.register_status_label.setText(f"{first_prompt}\n{REGISTER_PROGRESS_PREFIX}: {self.summarize_direction_progress()}")
        self.go_to(UiPage.REGISTER_CAPTURE)

    def evaluate_similarity_band(self, emb: np.ndarray) -> Tuple[bool, str]:
        existing = self.all_registered_embeddings()
        if not existing:
            return True, ""

        sims = [cosine_sim(e, emb) for e in existing]
        max_sim = max(sims)
        if max_sim > REGISTER_SIMILARITY_HIGHER_BOUNDARY:
            return False, f"{REGISTER_DUPLICATE_TEXT} (max={max_sim:.3f})"
        if max_sim < REGISTER_SIMILARITY_LOWER_BOUNDARY:
            return False, f"{REGISTER_OUTLIER_TEXT} (max={max_sim:.3f})"
        return True, REGISTER_SIM_IN_BAND_TEXT_FMT.format(max_sim=max_sim)

    def handle_register_capture(self, frame: np.ndarray):
        self.register_frame_tick += 1
        if self.register_frame_tick % REGISTER_CAPTURE_EVERY_N_FRAMES != 0:
            return

        target_direction = self.next_target_direction()
        if target_direction is None:
            self.register_status_label.setText(REGISTER_DONE_TEXT)
            self.go_to(UiPage.REGISTER_NAME)
            return

        emb, crop, _, detected_direction, _ = self.engine.analyze_face(frame)
        if emb is None:
            self.register_status_label.setText(
                f"{REGISTER_FACE_NOT_FOUND_TEXT}\n{DIRECTION_PROMPTS[target_direction]}\n{REGISTER_PROGRESS_PREFIX}: {self.summarize_direction_progress()}"
            )
            return

        if crop is None or crop.size == 0:
            self.register_status_label.setText(
                f"{REGISTER_FACE_ALIGN_TEXT}\n{DIRECTION_PROMPTS[target_direction]}\n{REGISTER_PROGRESS_PREFIX}: {self.summarize_direction_progress()}"
            )
            return

        if detected_direction != target_direction:
            detected_label = DIRECTION_LABELS.get(detected_direction, REGISTER_DETECTED_DIRECTION_FRONT_LABEL)
            self.register_status_label.setText(
                f"{REGISTER_DETECTED_DIRECTION_FMT.format(detected_label=detected_label)}\n{DIRECTION_PROMPTS[target_direction]}\n{REGISTER_PROGRESS_PREFIX}: {self.summarize_direction_progress()}"
            )
            return

        accepted, msg = self.evaluate_similarity_band(emb)
        if not accepted:
            self.register_status_label.setText(
                f"{msg}\n{DIRECTION_PROMPTS[target_direction]}\n{REGISTER_PROGRESS_PREFIX}: {self.summarize_direction_progress()}"
            )
            return

        self.register_direction_embeddings[target_direction].append(emb)
        self.register_direction_crops[target_direction].append(crop)
        self.set_register_count()
        next_direction = self.next_target_direction()
        if next_direction is None:
            self.register_status_label.setText(REGISTER_DONE_TEXT)
            self.go_to(UiPage.REGISTER_NAME)
            return

        self.register_status_label.setText(
            f"{REGISTER_SAMPLE_SAVED_FMT.format(msg=msg)}\n{DIRECTION_PROMPTS[next_direction]}\n{REGISTER_PROGRESS_PREFIX}: {self.summarize_direction_progress()}"
        )

    def save_registration(self):
        name = self.name_input.text().strip()
        if not name:
            self.show_error(ERROR_ENTER_NAME_TEXT)
            return

        embeddings = self.all_registered_embeddings()
        crops = self.all_registered_crops()
        if len(embeddings) < self.register_total_target:
            self.show_error(ERROR_NOT_ENOUGH_SAMPLES_TEXT)
            return

        self.db.upsert_person(name, embeddings, crops)
        self.go_initial_state()

    def try_adaptive_update(self, person_name: str, top_score: float, emb: np.ndarray, crop: np.ndarray):
        if not ADAPTIVE_UPDATE_ENABLED:
            return
        if crop is None or crop.size == 0:
            return
        if top_score < AUTH_SIM_THRESHOLD or top_score > ADAPTIVE_UPDATE_MAX_SIM:
            return
        if self.adaptive_update_count >= ADAPTIVE_UPDATE_MAX_SAMPLES_PER_SESSION:
            return
        if (self.analyze_frame_tick - self.last_adaptive_update_tick) < ADAPTIVE_UPDATE_MIN_INTERVAL_FRAMES:
            return

        if self.db.append_sample_to_person(person_name, emb, crop):
            self.last_adaptive_update_tick = self.analyze_frame_tick
            self.adaptive_update_count += 1
            self.monitor_detail_label.setText(ADAPTIVE_UPDATE_NOTICE_TEXT)

    def apply_monitor_state_from_frame(self, frame: np.ndarray):
        emb, crop, _ = self.engine.embedding_and_crop(frame)
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
            self.try_adaptive_update(top_name, top_score, emb, crop)
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
        QMessageBox.critical(self, ERROR_DIALOG_TITLE, message)

    def close_app(self):
        self.close()

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
