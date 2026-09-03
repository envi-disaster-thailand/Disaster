from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

STATUS_FILE = OUTPUT_DIR / "run_status.json"
LOCK_FILE = OUTPUT_DIR / ".forecast.lock"
LAST_RUN_FILE = OUTPUT_DIR / "last_run.txt"
COOLDOWN_MINUTES = 15

ENGINE_FILE = ROOT / "forecast_engine.py"

# A status is considered suspicious when no heartbeat has been received for
# this long. It is a warning first; it does not automatically kill the job.
STALE_WARNING_MINUTES = 8
STALE_LOCK_MINUTES = 45


def _now_dt() -> datetime:
    return datetime.now()


def _now() -> str:
    return _now_dt().strftime("%Y-%m-%d %H:%M:%S")


def _read_status() -> dict:
    if not STATUS_FILE.exists():
        return {}
    try:
        return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _status_started_at(previous: dict | None = None) -> str:
    previous = previous or {}
    return previous.get("started_at") or _now()


def _write_payload(payload: dict):
    tmp = STATUS_FILE.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(STATUS_FILE)


def _write_status(
    *,
    running: bool,
    step: int,
    progress: int,
    message: str,
    error: str | None = None,
    health: str = "normal",
    anomaly: str | None = None,
    detail: str | None = None,
    engine_pid: int | None = None,
    preserve_started_at: bool = True,
):
    previous = _read_status()

    payload = {
        "running": running,
        "step": int(step),
        "progress": int(progress),
        "message": message,
        "updated_at": _now(),
        "heartbeat_at": _now(),
        "started_at": (
            _status_started_at(previous)
            if preserve_started_at
            else _now()
        ),
        "error": error,
        "health": health,
        "anomaly": anomaly,
        "detail": detail,
        "engine_pid": engine_pid if engine_pid is not None else previous.get("engine_pid"),
    }
    _write_payload(payload)


def _transition_anomaly(previous: dict, step: int, progress: int) -> str | None:
    """
    Check the logical order of status markers.

    Allowed:
    - same step with increasing/equal progress
    - next step
    - first marker at step 1

    Suspicious:
    - step goes backwards
    - skips more than one step
    - progress goes backwards
    """
    if not previous or not previous.get("running"):
        return None

    old_step = int(previous.get("step", 0) or 0)
    old_progress = int(previous.get("progress", 0) or 0)

    if old_step and step < old_step:
        return (
            f"ลำดับขั้นตอนย้อนกลับจากขั้น {old_step} ไปขั้น {step} "
            "ระบบจะดำเนินการต่อแต่ควรตรวจสอบ Log"
        )

    if old_step and step > old_step + 1:
        return (
            f"พบการข้ามลำดับจากขั้น {old_step} ไปขั้น {step} "
            "โดยไม่พบสถานะขั้นกลาง"
        )

    if progress < old_progress:
        return (
            f"ค่าความคืบหน้าลดลงจาก {old_progress}% เหลือ {progress}% "
            "ซึ่งไม่เป็นไปตามลำดับปกติ"
        )

    return None


def _notify(callback, step, progress, message):
    previous = _read_status()
    anomaly = _transition_anomaly(previous, step, progress)

    _write_status(
        running=True,
        step=step,
        progress=progress,
        message=message,
        error=None,
        health="warning" if anomaly else "normal",
        anomaly=anomaly,
        detail=message,
        engine_pid=previous.get("engine_pid"),
    )
    if callback:
        callback(step, progress, message)


def _heartbeat(detail: str):
    """Refresh job heartbeat from any engine output line."""
    status = _read_status()
    if not status.get("running"):
        return

    status["heartbeat_at"] = _now()
    status["updated_at"] = _now()
    if detail:
        # Keep a short, readable last activity line.
        status["detail"] = detail[-500:]
    _write_payload(status)


def _lock_age_minutes() -> float | None:
    if not LOCK_FILE.exists():
        return None
    try:
        age_sec = _now_dt().timestamp() - LOCK_FILE.stat().st_mtime
        return max(0.0, age_sec / 60.0)
    except Exception:
        return None


def status_health(status: dict | None = None) -> dict:
    """
    Return a user-facing health assessment without modifying the job.

    health:
      normal   = sequence and heartbeat look normal
      warning  = no recent heartbeat or sequence anomaly
      error    = stale lock / failed job
    """
    status = status or _read_status()
    result = {
        "health": status.get("health", "normal"),
        "anomaly": status.get("anomaly"),
        "stale": False,
        "lock_stale": False,
        "heartbeat_age_seconds": None,
    }

    heartbeat = status.get("heartbeat_at") or status.get("updated_at")
    if heartbeat:
        try:
            hb = datetime.fromisoformat(heartbeat)
            age = (_now_dt() - hb).total_seconds()
            result["heartbeat_age_seconds"] = max(0, int(age))

            if status.get("running") and age > STALE_WARNING_MINUTES * 60:
                result["stale"] = True
                result["health"] = "warning"
                if not result["anomaly"]:
                    result["anomaly"] = (
                        f"ไม่พบการอัปเดตจากกระบวนการนานกว่า "
                        f"{STALE_WARNING_MINUTES} นาที อาจกำลังใช้เวลาคำนวณนาน "
                        "หรือกระบวนการอาจหยุดตอบสนอง"
                    )
        except Exception:
            pass

    lock_age = _lock_age_minutes()
    if (
        LOCK_FILE.exists()
        and lock_age is not None
        and lock_age > STALE_LOCK_MINUTES
    ):
        result["lock_stale"] = True
        result["health"] = "error"
        result["anomaly"] = (
            f"พบ Lock ค้างนานกว่า {STALE_LOCK_MINUTES} นาที "
            "มีความเป็นไปได้ว่ากระบวนการเดิมหยุดทำงาน"
        )

    if status.get("error"):
        result["health"] = "error"

    return result


def clear_stale_lock_if_safe() -> bool:
    """
    Clear only an obviously stale lock.
    This is deliberately conservative.
    """
    health = status_health()
    if not health.get("lock_stale"):
        return False

    status = _read_status()
    _write_status(
        running=False,
        step=int(status.get("step", 0) or 0),
        progress=int(status.get("progress", 0) or 0),
        message="ยกเลิกสถานะค้างของกระบวนการเดิมแล้ว",
        error=None,
        health="warning",
        anomaly="ตรวจพบ Lock ค้างและได้ปลด Lock อัตโนมัติเพื่อให้ระบบกลับมาใช้งานได้",
        detail=status.get("detail"),
    )
    LOCK_FILE.unlink(missing_ok=True)
    return True


def run_forecast(callback: Callable | None = None):
    """
    Run one forecast job only with sequence validation and heartbeat tracking.
    """
    # Recover only an obviously stale lock.
    clear_stale_lock_if_safe()

    if LOCK_FILE.exists():
        raise RuntimeError("ระบบกำลังประมวลผลข้อมูลอยู่")

    if LAST_RUN_FILE.exists():
        try:
            last_run = datetime.fromisoformat(
                LAST_RUN_FILE.read_text(encoding="utf-8").strip()
            )
            elapsed = _now_dt() - last_run
            cooldown = timedelta(minutes=COOLDOWN_MINUTES)
            if elapsed < cooldown:
                remaining = cooldown - elapsed
                total_seconds = max(0, int(remaining.total_seconds()))
                minutes, seconds = divmod(total_seconds, 60)
                raise RuntimeError(
                    f"สามารถดำเนินการประมวลผลครั้งถัดไปได้ใน "
                    f"{minutes} นาที {seconds} วินาที"
                )
        except ValueError:
            pass

    LOCK_FILE.write_text(str(os.getpid()), encoding="utf-8")

    # New run: reset previous status cleanly.
    _write_status(
        running=True,
        step=1,
        progress=1,
        message="เริ่มต้นกระบวนการประมวลผล",
        error=None,
        health="normal",
        anomaly=None,
        detail="กำลังเริ่มต้นระบบ",
        preserve_started_at=False,
    )

    process = None

    try:
        if not ENGINE_FILE.exists():
            raise FileNotFoundError(
                "ไม่พบ forecast_engine.py สำหรับดำเนินการประมวลผล"
            )

        _notify(callback, 1, 5, "Preparing system...")

        env = os.environ.copy()
        env["DASHBOARD_OUTPUT_DIR"] = str(OUTPUT_DIR)

        process = subprocess.Popen(
            [sys.executable, "-u", str(ENGINE_FILE)],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        # Save actual child process PID.
        status = _read_status()
        status["engine_pid"] = process.pid
        _write_payload(status)

        assert process.stdout is not None

        for raw_line in process.stdout:
            line = raw_line.rstrip()
            print(line, flush=True)

            # Every output line is evidence that the process is alive.
            _heartbeat(line)

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
            health="normal",
            anomaly=None,
            detail="ผลการประมวลผลถูกจัดทำครบถ้วนแล้ว",
        )
        LAST_RUN_FILE.write_text(_now_dt().isoformat(), encoding="utf-8")

    except Exception as exc:
        previous = _read_status()
        _write_status(
            running=False,
            step=int(previous.get("step", 0) or 0),
            progress=int(previous.get("progress", 0) or 0),
            message="การประมวลผลหยุดก่อนเสร็จสมบูรณ์",
            error=str(exc),
            health="error",
            anomaly=(
                previous.get("anomaly")
                or "กระบวนการหยุดทำงานก่อนดำเนินการครบทุกขั้นตอน"
            ),
            detail=previous.get("detail"),
        )
        raise

    finally:
        LOCK_FILE.unlink(missing_ok=True)


def get_latest_day_images() -> dict[str, Path]:
    result: dict[str, Path] = {}

    patterns = [
        "ecmwf-24hr-day*-Local-time.png",
        "ecmwf-24hr-day*.png",
    ]

    files = []
    for pattern in patterns:
        files.extend(OUTPUT_DIR.glob(pattern))
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
