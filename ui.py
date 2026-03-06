import sys
from typing import List, Optional, Tuple

import numpy as np

from config import (
    AUTH_MARGIN_THRESHOLD,
    AUTH_SIM_THRESHOLD,
    CAMERA_INDEX,
    REQUIRED_SIMILAR_COUNT,
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
    QListWidget,
    QListWidgetItem,
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
        self.last_frame = None

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
            self.last_frame = frame
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
        pix = np_to_qpixmap(frame_bgr, self.size())
        self.setPixmap(pix)


class CandidateButton(QPushButton):
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setMinimumHeight(56)
        self.setStyleSheet("QPushButton{font-size:16px;padding:10px 14px;border-radius:16px;border:1px solid #d6d6d6;background:#fff;} QPushButton:hover{background:#f6f6f6;}")


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
        self.current_frame = None
        self.current_bbox = None

        self.history: List[int] = []
        self.register_embeddings: List[np.ndarray] = []
        self.register_crops: List[np.ndarray] = []
        self.auth_rankings: List[Tuple[str, float]] = []
        self.selected_name: Optional[str] = None

        self.stack = QStackedWidget()
        self.video_panel_register = VideoPanel()
        self.video_panel_auth = VideoPanel()

        self.home_page = self.build_home_page()
        self.register_capture_page = self.build_register_capture_page()
        self.register_name_page = self.build_register_name_page()
        self.register_done_page = self.build_register_done_page()
        self.auth_capture_page = self.build_auth_capture_page()
        self.auth_result_page = self.build_auth_result_page()
        self.greeting_page = self.build_greeting_page()

        for page in [
            self.home_page,
            self.register_capture_page,
            self.register_name_page,
            self.register_done_page,
            self.auth_capture_page,
            self.auth_result_page,
            self.greeting_page,
        ]:
            self.stack.addWidget(page)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.addWidget(self.stack)

        self.go_to(0, push_history=False)
        self.camera.start()
        self.refresh_people_list()

    def shell(self, title: str, subtitle: str, back_handler=None):
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
        if back_handler is not None:
            back_btn = QPushButton("뒤로가기")
            back_btn.setMinimumHeight(42)
            back_btn.clicked.connect(back_handler)
            back_btn.setStyleSheet("QPushButton{padding:8px 16px;border-radius:14px;background:#fff;border:1px solid #ddd;} QPushButton:hover{background:#f4f4f4;}")
            header.addWidget(back_btn)
        layout.addLayout(header)
        layout.addSpacing(10)
        return wrapper, layout

    def card_style(self):
        return "background:#fff;border:1px solid #e7e7e7;border-radius:22px;"

    def build_home_page(self):
        page, layout = self.shell("로컬 얼굴 등록 / 인증 앱", "웹이 아니라 데스크톱 로컬에서 실행되는 앱입니다.")

        row = QHBoxLayout()
        row.setSpacing(14)

        left = QFrame()
        left.setStyleSheet(self.card_style())
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(20, 20, 20, 20)
        t1 = QLabel("얼굴 등록")
        t1.setStyleSheet("font-size:22px;font-weight:600;")
        d1 = QLabel(f"같은 인물의 얼굴이 유사도 기준 이상으로 {REQUIRED_SIMILAR_COUNT}개 확보되면 이름 저장 단계로 이동합니다.")
        d1.setWordWrap(True)
        d1.setStyleSheet("color:#666;font-size:14px;")
        b1 = QPushButton("등록 시작")
        b1.setMinimumHeight(52)
        b1.clicked.connect(self.start_register)
        b1.setStyleSheet("QPushButton{font-size:16px;font-weight:600;background:#111;color:white;border-radius:16px;} QPushButton:hover{background:#222;}")
        left_layout.addWidget(t1)
        left_layout.addWidget(d1)
        left_layout.addStretch(1)
        left_layout.addWidget(b1)

        right = QFrame()
        right.setStyleSheet(self.card_style())
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(20, 20, 20, 20)
        t2 = QLabel("얼굴 인증")
        t2.setStyleSheet("font-size:22px;font-weight:600;")
        d2 = QLabel("1위와 2위의 차이가 충분히 크면 1위만 표시하고, 작으면 1~3위를 선택할 수 있습니다.")
        d2.setWordWrap(True)
        d2.setStyleSheet("color:#666;font-size:14px;")
        b2 = QPushButton("인증 시작")
        b2.setMinimumHeight(52)
        b2.clicked.connect(self.start_auth)
        b2.setStyleSheet("QPushButton{font-size:16px;font-weight:600;background:#111;color:white;border-radius:16px;} QPushButton:hover{background:#222;}")
        right_layout.addWidget(t2)
        right_layout.addWidget(d2)
        right_layout.addStretch(1)
        right_layout.addWidget(b2)

        row.addWidget(left, 1)
        row.addWidget(right, 1)
        layout.addLayout(row)

        people_card = QFrame()
        people_card.setStyleSheet(self.card_style())
        people_layout = QVBoxLayout(people_card)
        people_layout.setContentsMargins(20, 20, 20, 20)
        title = QLabel("등록된 사용자")
        title.setStyleSheet("font-size:20px;font-weight:600;")
        self.people_list = QListWidget()
        people_layout.addWidget(title)
        people_layout.addWidget(self.people_list)
        layout.addWidget(people_card, 1)
        return page

    def build_register_capture_page(self):
        page, layout = self.shell("얼굴 등록 모드", "같은 사람의 얼굴을 여러 번 촬영해 유사 샘플을 확보합니다.", self.go_back)
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
        self.register_status = QLabel("정면에서 자연스럽게 여러 번 촬영해 주세요.")
        self.register_status.setWordWrap(True)
        self.register_status.setStyleSheet("font-size:14px;color:#555;")
        self.register_progress = QProgressBar()
        self.register_progress.setRange(0, REQUIRED_SIMILAR_COUNT)
        self.register_progress.setValue(0)
        self.register_count_label = QLabel(f"0 / {REQUIRED_SIMILAR_COUNT}")
        self.register_count_label.setStyleSheet("font-size:18px;font-weight:600;")
        capture_btn = QPushButton("얼굴 촬영")
        capture_btn.setMinimumHeight(50)
        capture_btn.clicked.connect(self.capture_for_register)
        capture_btn.setStyleSheet("QPushButton{background:#111;color:#fff;border-radius:16px;font-size:16px;font-weight:600;} QPushButton:hover{background:#222;}")
        reset_btn = QPushButton("다시 시작")
        reset_btn.setMinimumHeight(50)
        reset_btn.clicked.connect(self.reset_register)
        reset_btn.setStyleSheet("QPushButton{background:#fff;border:1px solid #ddd;border-radius:16px;font-size:16px;} QPushButton:hover{background:#f6f6f6;}")
        self.register_preview_list = QListWidget()
        right_layout.addWidget(QLabel("등록 진행도"))
        right_layout.addWidget(self.register_progress)
        right_layout.addWidget(self.register_count_label)
        right_layout.addWidget(self.register_status)
        right_layout.addWidget(capture_btn)
        right_layout.addWidget(reset_btn)
        right_layout.addWidget(QLabel("최근 얼굴 샘플"))
        right_layout.addWidget(self.register_preview_list, 1)

        body.addWidget(left, 3)
        body.addWidget(right, 2)
        layout.addLayout(body, 1)
        return page

    def build_register_name_page(self):
        page, layout = self.shell("이름 저장", "이름을 입력하면 해당 이름 폴더를 생성하고 얼굴 샘플을 저장합니다.", self.go_back)
        card = QFrame()
        card.setStyleSheet(self.card_style())
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 20, 20, 20)
        msg = QLabel("얼굴 등록이 완료되었습니다.")
        msg.setStyleSheet("font-size:24px;font-weight:700;")
        msg2 = QLabel("이름을 입력한 뒤 저장을 누르세요.")
        msg2.setStyleSheet("color:#666;font-size:14px;")
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("이름 입력")
        self.name_input.setMinimumHeight(48)
        self.name_input.setStyleSheet("QLineEdit{font-size:16px;padding:10px 14px;border:1px solid #ddd;border-radius:14px;}")
        save_btn = QPushButton("저장")
        save_btn.setMinimumHeight(50)
        save_btn.clicked.connect(self.save_registration)
        save_btn.setStyleSheet("QPushButton{background:#111;color:#fff;border-radius:16px;font-size:16px;font-weight:600;} QPushButton:hover{background:#222;}")
        card_layout.addWidget(msg)
        card_layout.addWidget(msg2)
        card_layout.addSpacing(10)
        card_layout.addWidget(self.name_input)
        card_layout.addWidget(save_btn)
        card_layout.addStretch(1)
        layout.addWidget(card)
        return page

    def build_register_done_page(self):
        page, layout = self.shell("등록 완료", "얼굴과 이름이 저장되었습니다.", self.go_back)
        card = QFrame()
        card.setStyleSheet(self.card_style())
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 30, 20, 30)
        self.register_done_label = QLabel("등록 완료")
        self.register_done_label.setAlignment(Qt.AlignCenter)
        self.register_done_label.setStyleSheet("font-size:30px;font-weight:700;")
        home_btn = QPushButton("홈으로")
        home_btn.setMinimumHeight(50)
        home_btn.clicked.connect(lambda: self.go_to(0))
        auth_btn = QPushButton("인증으로 이동")
        auth_btn.setMinimumHeight(50)
        auth_btn.clicked.connect(self.start_auth)
        for btn in [home_btn, auth_btn]:
            btn.setStyleSheet("QPushButton{background:#111;color:#fff;border-radius:16px;font-size:16px;font-weight:600;} QPushButton:hover{background:#222;}")
        card_layout.addStretch(1)
        card_layout.addWidget(self.register_done_label)
        card_layout.addSpacing(14)
        card_layout.addWidget(home_btn)
        card_layout.addWidget(auth_btn)
        card_layout.addStretch(1)
        layout.addWidget(card, 1)
        return page

    def build_auth_capture_page(self):
        page, layout = self.shell("얼굴 인증 모드", "등록된 얼굴과 비교해 1위 또는 1~3위 후보를 제안합니다.", self.go_back)
        body = QHBoxLayout()
        body.setSpacing(14)

        left = QFrame()
        left.setStyleSheet(self.card_style())
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(16, 16, 16, 16)
        left_layout.addWidget(self.video_panel_auth, 1)

        right = QFrame()
        right.setStyleSheet(self.card_style())
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(20, 20, 20, 20)
        info = QLabel("1위와 다른 순위의 차이가 클 경우 1위만 표현하고, 차이가 적을 경우 1~3위 이름을 출력합니다.")
        info.setWordWrap(True)
        info.setStyleSheet("font-size:14px;color:#666;")
        capture_btn = QPushButton("얼굴 인증 시작")
        capture_btn.setMinimumHeight(50)
        capture_btn.clicked.connect(self.capture_for_auth)
        capture_btn.setStyleSheet("QPushButton{background:#111;color:#fff;border-radius:16px;font-size:16px;font-weight:600;} QPushButton:hover{background:#222;}")
        self.auth_info_label = QLabel(f"등록 사용자 수: {len(self.db.people)}")
        self.auth_info_label.setStyleSheet("font-size:18px;font-weight:600;")
        right_layout.addWidget(self.auth_info_label)
        right_layout.addWidget(info)
        right_layout.addWidget(capture_btn)
        right_layout.addStretch(1)

        body.addWidget(left, 3)
        body.addWidget(right, 2)
        layout.addLayout(body, 1)
        return page

    def build_auth_result_page(self):
        page, layout = self.shell("인증 결과", "점수 차이에 따라 1명만 표시하거나 1~3명을 선택하게 합니다.", self.go_back)
        card = QFrame()
        card.setStyleSheet(self.card_style())
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 20, 20, 20)
        self.auth_result_label = QLabel("결과 없음")
        self.auth_result_label.setWordWrap(True)
        self.auth_result_label.setStyleSheet("font-size:20px;font-weight:600;")
        self.auth_candidates_box = QVBoxLayout()
        card_layout.addWidget(self.auth_result_label)
        card_layout.addSpacing(10)
        card_layout.addLayout(self.auth_candidates_box)
        card_layout.addStretch(1)
        layout.addWidget(card, 1)
        return page

    def build_greeting_page(self):
        page, layout = self.shell("인증 완료", "선택된 사용자를 확정했습니다.", self.go_back)
        card = QFrame()
        card.setStyleSheet(self.card_style())
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 30, 20, 30)
        self.greeting_label = QLabel("반갑습니다.")
        self.greeting_label.setAlignment(Qt.AlignCenter)
        self.greeting_label.setStyleSheet("font-size:34px;font-weight:700;")
        self.greeting_name_label = QLabel("")
        self.greeting_name_label.setAlignment(Qt.AlignCenter)
        self.greeting_name_label.setStyleSheet("font-size:20px;color:#555;")
        home_btn = QPushButton("홈으로")
        home_btn.clicked.connect(lambda: self.go_to(0))
        home_btn.setMinimumHeight(50)
        home_btn.setStyleSheet("QPushButton{background:#111;color:#fff;border-radius:16px;font-size:16px;font-weight:600;} QPushButton:hover{background:#222;}")
        card_layout.addStretch(1)
        card_layout.addWidget(self.greeting_label)
        card_layout.addWidget(self.greeting_name_label)
        card_layout.addSpacing(16)
        card_layout.addWidget(home_btn)
        card_layout.addStretch(1)
        layout.addWidget(card, 1)
        return page

    def go_to(self, index: int, push_history: bool = True):
        current = self.stack.currentIndex()
        if push_history and current != index:
            self.history.append(current)
        self.stack.setCurrentIndex(index)
        self.refresh_people_list()
        self.auth_info_label.setText(f"등록 사용자 수: {len(self.db.people)}")

    def go_back(self):
        if self.history:
            index = self.history.pop()
            self.stack.setCurrentIndex(index)
        else:
            self.stack.setCurrentIndex(0)

    def refresh_people_list(self):
        self.db.load()
        self.people_list.clear()
        if not self.db.people:
            self.people_list.addItem(QListWidgetItem("아직 등록된 사용자가 없습니다."))
            return

        for person in sorted(self.db.people, key=lambda x: x.name):
            self.people_list.addItem(QListWidgetItem(f"{person.name}  |  샘플 {person.num_samples}개  |  폴더 {person.folder}"))

    def set_register_status(self, text: str):
        self.register_status.setText(text)
        self.register_count_label.setText(f"{len(self.register_embeddings)} / {REQUIRED_SIMILAR_COUNT}")
        self.register_progress.setValue(min(len(self.register_embeddings), REQUIRED_SIMILAR_COUNT))

    def update_register_preview(self):
        self.register_preview_list.clear()
        for idx in range(len(self.register_crops)):
            self.register_preview_list.addItem(QListWidgetItem(f"샘플 {idx + 1}"))

    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                self.clear_layout(child_layout)

    def on_frame(self, frame: np.ndarray):
        self.current_frame = frame.copy()
        display, bbox = self.engine.detect_and_draw_bbox(frame)
        self.current_bbox = bbox

        if self.stack.currentIndex() == 1:
            self.video_panel_register.update_frame(display)
        elif self.stack.currentIndex() == 4:
            self.video_panel_auth.update_frame(display)

    def show_error(self, message: str):
        QMessageBox.critical(self, "오류", message)

    def reset_register(self):
        self.register_embeddings = []
        self.register_crops = []
        self.name_input.clear()
        self.register_preview_list.clear()
        self.set_register_status("정면에서 자연스럽게 여러 번 촬영해 주세요.")

    def start_register(self):
        self.reset_register()
        self.go_to(1)

    def capture_for_register(self):
        if self.current_frame is None:
            self.show_error("카메라 프레임이 아직 없습니다.")
            return

        emb, crop, _ = self.engine.embedding_and_crop(self.current_frame)
        if emb is None:
            self.set_register_status("얼굴을 찾지 못했습니다. 카메라를 정면으로 봐 주세요.")
            return

        if not self.register_embeddings:
            self.register_embeddings.append(emb)
            self.register_crops.append(crop)
            self.update_register_preview()
            self.set_register_status("첫 얼굴 샘플이 확보되었습니다. 같은 인물로 추가 촬영해 주세요.")
            return

        base = self.register_embeddings[0]
        sim = cosine_sim(base, emb)
        if sim >= REGISTER_SIM_THRESHOLD:
            self.register_embeddings.append(emb)
            self.register_crops.append(crop)
            self.update_register_preview()
            if len(self.register_embeddings) >= REQUIRED_SIMILAR_COUNT:
                self.set_register_status("얼굴 등록이 완료되었습니다.")
                self.go_to(2)
            else:
                self.set_register_status(
                    f"유사한 얼굴 확보: {len(self.register_embeddings)} / {REQUIRED_SIMILAR_COUNT} (유사도 {sim:.3f})"
                )
        else:
            self.set_register_status(f"같은 사람으로 보기 어려운 샘플입니다. 다시 촬영해 주세요. (유사도 {sim:.3f})")

    def save_registration(self):
        name = self.name_input.text().strip()
        if not name:
            self.show_error("이름을 입력해 주세요.")
            return
        if len(self.register_embeddings) < REQUIRED_SIMILAR_COUNT:
            self.show_error("아직 충분한 얼굴 샘플이 확보되지 않았습니다.")
            return

        self.db.upsert_person(name, self.register_embeddings, self.register_crops)
        self.register_done_label.setText(f"{name} 님의 얼굴 등록이 완료되었습니다.")
        self.refresh_people_list()
        self.go_to(3)

    def start_auth(self):
        self.db.load()
        if not self.db.people:
            self.show_error("먼저 얼굴 등록을 해 주세요.")
            return
        self.auth_rankings = []
        self.selected_name = None
        self.go_to(4)

    def capture_for_auth(self):
        if self.current_frame is None:
            self.show_error("카메라 프레임이 아직 없습니다.")
            return

        emb, _, _ = self.engine.embedding_and_crop(self.current_frame)
        if emb is None:
            self.show_error("얼굴을 찾지 못했습니다. 다시 시도해 주세요.")
            return

        rankings = self.db.rank(emb, TOP_K)
        if not rankings:
            self.show_error("등록된 사용자가 없습니다.")
            return

        top_name, top_score = rankings[0]
        second_score = rankings[1][1] if len(rankings) > 1 else -1.0
        margin = top_score - second_score if second_score >= 0 else top_score
        self.auth_rankings = rankings
        self.clear_layout(self.auth_candidates_box)

        if top_score >= AUTH_SIM_THRESHOLD and margin >= AUTH_MARGIN_THRESHOLD:
            self.auth_result_label.setText(f"1위가 명확합니다: {top_name} (점수 {top_score:.3f}, 차이 {margin:.3f})")
            btn = CandidateButton(f"{top_name} 선택")
            btn.clicked.connect(lambda: self.confirm_name(top_name))
            self.auth_candidates_box.addWidget(btn)
        else:
            self.auth_result_label.setText("차이가 적어 상위 1~3위를 선택할 수 있습니다.")
            for idx, (name, score) in enumerate(rankings, start=1):
                btn = CandidateButton(f"{idx}위  {name}  |  점수 {score:.3f}")
                btn.clicked.connect(lambda checked=False, n=name: self.confirm_name(n))
                self.auth_candidates_box.addWidget(btn)
        self.go_to(5)

    def confirm_name(self, name: str):
        self.selected_name = name
        self.greeting_name_label.setText(f"선택된 사용자: {name}")
        self.go_to(6)

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
    QListWidget { background:#fff; border:1px solid #e7e7e7; border-radius:14px; padding:8px; }
    QProgressBar { background:#ededed; border:none; border-radius:10px; text-align:center; min-height:18px; }
    QProgressBar::chunk { background:#111; border-radius:10px; }
    """
    )
    win = MainWindow()
    win.show()
    return app.exec()
