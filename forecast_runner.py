from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import timedelta
from datetime import datetime
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

STATUS_FILE = OUTPUT_DIR / "run_status.json"
LOCK_FILE = OUTPUT_DIR / ".forecast.lock"
LAST_RUN_FILE = OUTPUT_DIR / "last_run.txt"
COOLDOWN_MINUTES = 15
# A run that is killed mid-way (out of memory, container restart) leaves the
# lock file behind and the RUN button stays disabled forever. Treat a lock
# older than this as stale.
LOCK_TIMEOUT_MINUTES = 60

# The processing engine command can later be replaced by Google Cloud Run / Cloud Batch.
# For V1 it runs forecast_engine.py on the same server.
ENGINE_FILE = ROOT / "forecast_engine.py"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _write_status(
    *,
    running: bool,
    step: int,
    progress: int,
    message: str,
    error: str | None = None,
):
    payload = {
        "running": running,
        "step": step,
        "progress": progress,
        "message": message,
        "updated_at": _now(),
        "error": error,
    }
    STATUS_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _notify(callback, step, progress, message):
    _write_status(
        running=True,
        step=step,
        progress=progress,
        message=message,
        error=None,
    )
    if callback:
        callback(step, progress, message)


def _lock_is_stale() -> bool:
    try:
        age = time.time() - LOCK_FILE.stat().st_mtime
    except OSError:
        return True
    return age > LOCK_TIMEOUT_MINUTES * 60


def lock_is_active() -> bool:
    return LOCK_FILE.exists() and not _lock_is_stale()


def run_forecast(callback: Callable | None = None):
    """
    Run one forecast job only.

    Important:
    - The web page never asks users for an email.
    - Credentials belong in server-side environment variables / secret manager.
    - LOCK_FILE prevents users from starting duplicate jobs simultaneously.
    """
    if LOCK_FILE.exists():
        if _lock_is_stale():
            print("[warn] Removing stale lock file.", flush=True)
            LOCK_FILE.unlink(missing_ok=True)
        else:
            raise RuntimeError("ระบบกำลังประมวลผลข้อมูลอยู่")

    if LAST_RUN_FILE.exists():
        try:
            last_run = datetime.fromisoformat(LAST_RUN_FILE.read_text(encoding="utf-8").strip())
            elapsed = datetime.now() - last_run
            cooldown = timedelta(minutes=COOLDOWN_MINUTES)
            if elapsed < cooldown:
                remaining = cooldown - elapsed
                total_seconds = max(0, int(remaining.total_seconds()))
                minutes, seconds = divmod(total_seconds, 60)
                raise RuntimeError(
                    f"สามารถดำเนินการประมวลผลครั้งถัดไปได้ใน {minutes} นาที {seconds} วินาที"
                )
        except ValueError:
            pass

    LOCK_FILE.write_text(str(os.getpid()), encoding="utf-8")

    try:
        if not ENGINE_FILE.exists():
            raise FileNotFoundError(
                "forecast_engine.py is missing. "
                "Add the converted ECMWF processing engine before deployment."
            )

        _notify(callback, 1, 5, "Preparing system...")

        env = os.environ.copy()
        env["DASHBOARD_OUTPUT_DIR"] = str(OUTPUT_DIR)

        # The engine runs as a plain python subprocess, so it cannot rely on
        # st.secrets resolving. Forward the credentials explicitly.
        for key in (
            "GOOGLE_CLIENT_ID",
            "GOOGLE_CLIENT_SECRET",
            "GOOGLE_REFRESH_TOKEN",
            "GOOGLE_DRIVE_FOLDER_ID",
        ):
            try:
                import streamlit as st

                value = st.secrets.get(key, "")
            except Exception:
                value = ""
            if value:
                env[key] = str(value).strip()

        # forecast_engine.py prints status markers:
        # STATUS|<step>|<progress>|<message>
        process = subprocess.Popen(
            [sys.executable, "-u", str(ENGINE_FILE)],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        assert process.stdout is not None

        for raw_line in process.stdout:
            line = raw_line.rstrip()
            print(line, flush=True)

            if line.startswith("STATUS|"):
                parts = line.split("|", 3)
                if len(parts) == 4:
                    try:
                        step = int(parts[1])
                        progress = int(parts[2])
                    except ValueError:
                        continue
                    message = parts[3]
                    _notify(callback, step, progress, message)

        code = process.wait()

        if code != 0:
            raise RuntimeError(f"Forecast engine exited with code {code}.")

        _write_status(
            running=False,
            step=7,
            progress=100,
            message="ประมวลผลข้อมูลเสร็จสมบูรณ์",
            error=None,
        )
        LAST_RUN_FILE.write_text(datetime.now().isoformat(), encoding="utf-8")

    except Exception as exc:
        _write_status(
            running=False,
            step=0,
            progress=0,
            message="Forecast failed.",
            error=str(exc),
        )
        raise

    finally:
        LOCK_FILE.unlink(missing_ok=True)


def get_latest_day_images() -> dict[str, Path]:
    """
    Find PART A daily PNG files generated by the original code.

    Expected filename pattern from the notebook:
    ecmwf-24hr-day1-YYYYMMDDtHHMM-YYYYMMDDtHHMM-Local-time.png
    ...
    ecmwf-24hr-day10-...
    """
    result: dict[str, Path] = {}

    patterns = [
        "ecmwf-24hr-day*-Local-time.png",
        "ecmwf-24hr-day*.png",
    ]

    files = []
    for pattern in patterns:
        files.extend(OUTPUT_DIR.glob(pattern))

    # Also support output files written in project root during transition.
    for pattern in patterns:
        files.extend(ROOT.glob(pattern))

    files = sorted(
        set(files),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )

    day_re = re.compile(r"day(\d+)", re.I)

    for path in files:
        m = day_re.search(path.name)
        if not m:
            continue
        day_no = int(m.group(1))
        if 1 <= day_no <= 10:
            key = f"Day {day_no}"
            if key not in result:
                result[key] = path

    return dict(
        sorted(
            result.items(),
            key=lambda kv: int(kv[0].split()[1]),
        )
    )
