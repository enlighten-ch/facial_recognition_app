import importlib.util
import os
import subprocess
import sys
from typing import Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

from config import DET_SIZE, MODEL_NAME, USE_GPU


def ensure_onnxruntime_compat() -> None:
    if hasattr(ort, "InferenceSession"):
        return

    try:
        from onnxruntime.capi.onnxruntime_inference_collection import InferenceSession
    except Exception as exc:
        module_path = getattr(ort, "__file__", "<unknown>")
        module_ns_path = list(getattr(ort, "__path__", []))
        spec = importlib.util.find_spec("onnxruntime")
        spec_origin = getattr(spec, "origin", None)
        raise RuntimeError(
            "Invalid onnxruntime installation: 'InferenceSession' is missing. "
            f"Imported module __file__: {module_path}, __path__: {module_ns_path}, "
            f"spec.origin: {spec_origin}. "
            f"Python executable: {sys.executable}. "
            "Reinstall with the same interpreter: "
            "'python -m pip uninstall -y onnxruntime onnxruntime-gpu && "
            "python -m pip install onnxruntime-gpu' "
            "(or install 'onnxruntime' for CPU-only)."
        ) from exc

    ort.InferenceSession = InferenceSession


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x) + eps)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


class FaceEngine:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        det_size: Tuple[int, int] = DET_SIZE,
        use_gpu: bool = USE_GPU,
    ):
        ensure_onnxruntime_compat()
        self.model_name = model_name
        self.det_size = det_size
        self.use_gpu = use_gpu
        self.providers = ort.get_available_providers()
        self.app = self._build_app()

    def _can_try_cuda(self) -> Tuple[bool, str]:
        if not self.use_gpu:
            return False, "GPU disabled by config"

        if os.environ.get("FACE_FORCE_CPU", "").lower() in {"1", "true", "yes"}:
            return False, "FACE_FORCE_CPU is enabled"

        if "CUDAExecutionProvider" not in self.providers:
            return False, "CUDAExecutionProvider is not available in onnxruntime"

        try:
            probe = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except FileNotFoundError:
            return False, "nvidia-smi not found"
        except Exception as exc:
            return False, f"nvidia-smi probe failed: {exc}"

        if probe.returncode != 0:
            err = (probe.stderr or probe.stdout or "").strip().splitlines()
            reason = err[0] if err else f"exit code {probe.returncode}"
            return False, f"nvidia-smi failed: {reason}"

        return True, "CUDA runtime looks available"

    def _build_app(self) -> FaceAnalysis:
        can_try_cuda, reason = self._can_try_cuda()
        if can_try_cuda:
            try:
                app = FaceAnalysis(
                    name=self.model_name,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
                app.prepare(ctx_id=0, det_size=self.det_size)
                print("[INFO] Using CUDAExecutionProvider")
                return app
            except Exception as exc:
                print(f"[WARN] GPU provider initialization failed: {exc}")
                print("[WARN] Falling back to CPUExecutionProvider")
        else:
            print(f"[INFO] CUDA skipped: {reason}")

        app = FaceAnalysis(name=self.model_name, providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=self.det_size)
        return app

    def detect_largest_face(self, frame_bgr: np.ndarray):
        faces = self.app.get(frame_bgr)
        if not faces:
            return None

        best = None
        best_area = -1
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            area = max(0, x2 - x1) * max(0, y2 - y1)
            if area > best_area:
                best = face
                best_area = area
        return best

    def embedding_and_crop(self, frame_bgr: np.ndarray):
        face = self.detect_largest_face(frame_bgr)
        if face is None:
            return None, None, None

        emb = np.asarray(face.normed_embedding, dtype=np.float32)
        x1, y1, x2, y2 = face.bbox.astype(int)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_bgr.shape[1], x2)
        y2 = min(frame_bgr.shape[0], y2)
        crop = frame_bgr[y1:y2, x1:x2].copy() if y2 > y1 and x2 > x1 else None
        return emb, crop, (x1, y1, x2, y2)

    def detect_and_draw_bbox(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
        display = frame_bgr.copy()
        face = self.detect_largest_face(display)
        if face is None:
            return display, None

        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return display, (x1, y1, x2, y2)
