import os
import subprocess
import tempfile
from typing import Tuple

from config import (
    DATA_DIR,
    ERROR_PRINT_FAILED_TEXT_FMT,
    PRINT_ENABLED,
    PRINT_SENT_TEXT_FMT,
    PRINTER_CONTENT_TEMPLATE,
    PRINTER_COPIES,
    PRINT_FONT_CANDIDATES,
    PRINT_TEXT_BASE_FONT_SIZE,
    PRINT_TEXT_SCALE_MULTIPLIER,
    PRINTER_JOB_TITLE_PREFIX,
    PRINTER_LP_COMMAND,
    PRINTER_NAME,
    PRINTER_TIMEOUT_SEC,
)


class PrinterService:
    def __init__(self):
        self.enabled = PRINT_ENABLED
        self.printer_name = PRINTER_NAME
        self.lp_cmd = PRINTER_LP_COMMAND
        self.timeout_sec = PRINTER_TIMEOUT_SEC
        self.copies = max(1, int(PRINTER_COPIES))
        self.job_title_prefix = PRINTER_JOB_TITLE_PREFIX

    def _pick_font_path(self) -> str:
        for path in PRINT_FONT_CANDIDATES:
            if os.path.exists(path):
                return path
        # Fallback: ask fontconfig for any Korean-capable font path.
        try:
            proc = subprocess.run(
                ["fc-list", ":lang=ko", "file"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            if proc.returncode == 0:
                for line in (proc.stdout or "").splitlines():
                    p = line.strip().split(":", 1)[0]
                    if p and os.path.exists(p):
                        return p
        except Exception:
            pass
        return ""

    def _render_text_as_png(self, content: str) -> str:
        # Prefer image-based printing so Korean glyphs are preserved and text size is predictable.
        try:
            from PIL import Image, ImageDraw, ImageFont
        except Exception as exc:
            raise RuntimeError(f"Pillow is required for image print mode: {exc}") from exc

        font_size = max(12, int(PRINT_TEXT_BASE_FONT_SIZE * PRINT_TEXT_SCALE_MULTIPLIER))
        font_path = self._pick_font_path()
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()

        # A4 @300dpi canvas
        width, height = 2480, 3508
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)

        left_margin = 180
        top_margin = 240
        line_gap = int(font_size * 0.35)

        y = top_margin
        for line in content.splitlines():
            if not line:
                y += font_size + line_gap
                continue
            draw.text((left_margin, y), line, fill="black", font=font)
            y += font_size + line_gap

        fd, out_path = tempfile.mkstemp(
            suffix=".png",
            prefix="print_job_",
            dir=DATA_DIR,
        )
        os.close(fd)
        image.save(out_path, format="PNG")
        return out_path

    def print_name(self, name: str) -> Tuple[bool, str]:
        if not self.enabled:
            return True, ""

        content = PRINTER_CONTENT_TEMPLATE.format(name=name)

        os.makedirs(DATA_DIR, exist_ok=True)
        tmp_path = ""
        try:
            # Render to image first for Korean-safe output and large text control.
            tmp_path = self._render_text_as_png(content)

            cmd = [self.lp_cmd]
            if self.printer_name:
                cmd.extend(["-d", self.printer_name])
            if self.copies > 1:
                cmd.extend(["-n", str(self.copies)])
            if self.job_title_prefix:
                cmd.extend(["-t", f"{self.job_title_prefix}-{name}"])
            cmd.append(tmp_path)

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                check=False,
            )
            if proc.returncode != 0:
                reason = (proc.stderr or proc.stdout or "lp command failed").strip()
                return False, ERROR_PRINT_FAILED_TEXT_FMT.format(reason=reason)

            return True, PRINT_SENT_TEXT_FMT.format(name=name)
        except FileNotFoundError:
            return False, ERROR_PRINT_FAILED_TEXT_FMT.format(reason=f"'{self.lp_cmd}' 명령을 찾을 수 없습니다")
        except Exception as exc:
            return False, ERROR_PRINT_FAILED_TEXT_FMT.format(reason=str(exc))
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
