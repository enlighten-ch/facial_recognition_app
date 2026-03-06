import os
import subprocess
import tempfile
from datetime import datetime
from typing import Tuple

from config import (
    DATA_DIR,
    ERROR_PRINT_FAILED_TEXT_FMT,
    PRINT_ENABLED,
    PRINT_SENT_TEXT_FMT,
    PRINTER_CONTENT_TEMPLATE,
    PRINTER_COPIES,
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

    def print_name(self, name: str) -> Tuple[bool, str]:
        if not self.enabled:
            return True, ""

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content = PRINTER_CONTENT_TEMPLATE.format(name=name, timestamp=timestamp)

        os.makedirs(DATA_DIR, exist_ok=True)
        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                delete=False,
                suffix=".txt",
                prefix="print_job_",
                dir=DATA_DIR,
            ) as fp:
                fp.write(content)
                tmp_path = fp.name

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
