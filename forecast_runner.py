from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import threading
import time
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable
import streamlit as st

def _env_int(name: str, default: int) -> int:
    """Read a whole-minute setting from Streamlit Secrets, then the environment."""
    value = None
    try:
        value = st.secrets.get(name, None)
    except Exception:
        value = None
    if value in (None, ""):
        value = os.getenv(name)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

STATUS_FILE = OUTPUT_DIR / "run_status.json"
LOCK_FILE = OUTPUT_DIR / ".forecast.lock"
LAST_RUN_FILE = OUTPUT_DIR / "last_run.txt"
# Minutes of rest after a run FINISHES before the next one may start.
# A dashboard run takes about 8 minutes, so the shortest possible gap between
# two runs starting is roughly 38 minutes.
COOLDOWN_MINUTES = int(_env_int("COOLDOWN_MINUTES", 30))

ENGINE_FILE = ROOT / "forecast_engine.py"

def _engine_env() -> dict:
    """Pass required Google Drive secrets from the Streamlit app to the engine subprocess."""
    env = os.environ.copy()
    secret_keys = (
        "GOOGLE_SERVICE_ACCOUNT_JSON",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLIENT_ID",
        "GOOGLE_CLIENT_SECRET",
        "GOOGLE_REFRESH_TOKEN",
        "GOOGLE_DRIVE_FOLDER_ID",
    )
    for key in secret_keys:
        try:
            value = st.secrets.get(key, None)
        except Exception:
            value = None
        if value in (None, ""):
            continue
        if not isinstance(value, str):
            # A service-account key stored as a TOML table arrives as a mapping;
            # environment variables can only carry strings.
            try:
                value = json.dumps(dict(value))
            except Exception:
                value = str(value)
        env[key] = value
    return env


# A status is considered suspicious when no heartbeat has been received for
# this long. It is a warning first; it does not automatically kill the job.
STALE_WARNING_MINUTES = 5
STALE_LOCK_MINUTES = 45

def _tunable(name: str, default: float) -> float:
    """Read a threshold from Streamlit Secrets, then the environment."""
    value = None
    try:
        value = st.secrets.get(name, None)
    except Exception:
        value = None
    if value in (None, ""):
        value = os.getenv(name)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# The watchdog terminates an engine that has gone silent or overrun. Without
# it a hung engine holds the lock until the Streamlit container is rebooted.
#
# Budget: V9.15 logs inside the contact-sheet build and before each Drive
# upload, so the longest legitimate gap between two output lines is now a
# single large upload (about 1-2 minutes). Warn at 5 minutes, kill at 10.
# Both thresholds can be overridden in Streamlit Secrets without code changes.
ENGINE_SILENCE_KILL_MINUTES = _tunable("ENGINE_SILENCE_KILL_MINUTES", 10)
ENGINE_MAX_RUNTIME_MINUTES = _tunable("ENGINE_MAX_RUNTIME_MINUTES", 120)
WATCHDOG_POLL_SECONDS = 20

# Guards start_forecast() against two sessions starting a run at the same time.
_START_LOCK = threading.Lock()


def _pid_is_alive(pid) -> bool:
    """Best-effort child-process liveness check on Linux."""
    try:
        pid = int(pid)
        if pid <= 0:
            return False
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (TypeError, ValueError, OSError):
        return False


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


def _write_payload(payload: dict) -> None:
    """
    Atomically write status using a unique temporary file per writer.

    Streamlit may execute multiple sessions/reruns concurrently. A fixed
    run_status.tmp filename allows one writer to rename another writer's
    temporary file, causing FileNotFoundError. mkstemp gives every writer
    its own file; os.replace then atomically publishes the completed JSON.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{STATUS_FILE.name}.",
        suffix=".tmp",
        dir=str(OUTPUT_DIR),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_name, STATUS_FILE)
    finally:
        # os.replace removes tmp_name on success. This only cleans up after
        # an exception before publication.
        try:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        except OSError:
            pass


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
    """Assess heartbeat, sequence, lock age, child PID, and legacy stale state."""
    status = status or _read_status()
    result = {
        "health": status.get("health", "normal"),
        "anomaly": status.get("anomaly"),
        "stale": False,
        "lock_stale": False,
        "process_dead": False,
        "legacy_stale": False,
        "heartbeat_age_seconds": None,
        "pid": status.get("engine_pid"),
        "pid_alive": None,
    }

    pid = status.get("engine_pid")
    if pid is not None:
        result["pid_alive"] = _pid_is_alive(pid)

    heartbeat = status.get("heartbeat_at") or status.get("updated_at")
    age = None
    if heartbeat:
        try:
            hb = datetime.fromisoformat(heartbeat)
            age = max(0.0, (_now_dt() - hb).total_seconds())
            result["heartbeat_age_seconds"] = int(age)
        except Exception:
            pass

    # Strong evidence: a recorded child PID no longer exists.
    if status.get("running") and pid is not None and result["pid_alive"] is False:
        result["process_dead"] = True
        result["health"] = "error"
        result["anomaly"] = (
            "ไม่พบกระบวนการประมวลผลเดิมแล้ว แต่สถานะยังค้างว่า Running"
        )
        return result

    # Recovery for V8/older status files: running=True, no engine_pid, and
    # heartbeat has been silent longer than the hard stale threshold.
    if (status.get("running") and pid is None and age is not None
            and age > STALE_LOCK_MINUTES * 60):
        result["legacy_stale"] = True
        result["lock_stale"] = True
        result["health"] = "error"
        result["anomaly"] = (
            f"พบสถานะงานเดิมที่ไม่มี Process ID และไม่มีการอัปเดตนานกว่า "
            f"{STALE_LOCK_MINUTES} นาที ระบบจะปลด Lock ของงานเดิม"
        )
        return result

    if status.get("running") and age is not None and age > STALE_WARNING_MINUTES * 60:
        result["stale"] = True
        result["health"] = "warning"
        if not result["anomaly"]:
            result["anomaly"] = (
                f"ไม่พบการอัปเดตนานกว่า {STALE_WARNING_MINUTES} นาที "
                "แต่จะไม่ถูกยกเลิกงานหาก Process ยังทำงานอยู่"
            )

    lock_age = _lock_age_minutes()
    if (LOCK_FILE.exists() and lock_age is not None
            and lock_age > STALE_LOCK_MINUTES
            and (pid is None or result["pid_alive"] is False)):
        result["lock_stale"] = True
        result["health"] = "error"
        result["anomaly"] = (
            f"พบ Lock ค้างนานกว่า {STALE_LOCK_MINUTES} นาที "
            "และไม่พบ Process ที่ยังทำงานอยู่"
        )

    if status.get("error"):
        result["health"] = "error"
    return result

def clear_stale_lock_if_safe() -> bool:
    """Clear only a lock/state whose process is demonstrably gone or legacy-stale."""
    status = _read_status()
    health = status_health(status)
    lock_age = _lock_age_minutes()

    should_clear = (
        health.get("process_dead")
        or health.get("legacy_stale")
        or health.get("lock_stale")
        or (
            LOCK_FILE.exists()
            and not status.get("running")
            and lock_age is not None
            and lock_age > 1
        )
    )
    if not should_clear:
        return False

    LOCK_FILE.unlink(missing_ok=True)

    # Convert the stale old run into a completed-abnormal state so app.py
    # no longer disables the Run button.
    _write_status(
        running=False,
        step=int(status.get("step", 0) or 0),
        progress=int(status.get("progress", 0) or 0),
        message="งานเดิมหยุดทำงาน ระบบปลด Lock แล้ว สามารถเริ่มประมวลผลใหม่ได้",
        error=None,
        health="warning",
        anomaly=health.get("anomaly") or "ตรวจพบ Lock/สถานะของงานเดิมค้าง",
        detail="STALE_STATE_RECOVERED",
        engine_pid=None,
    )
    return True

def _heartbeat_age_seconds() -> float | None:
    status = _read_status()
    stamp = status.get("heartbeat_at") or status.get("updated_at")
    if not stamp:
        return None
    try:
        return max(0.0, (_now_dt() - datetime.fromisoformat(stamp)).total_seconds())
    except Exception:
        return None


def _engine_watchdog(process: subprocess.Popen, started_at: datetime, state: dict):
    """Terminate an engine that stopped reporting or overran the time budget."""
    while process.poll() is None:
        time.sleep(WATCHDOG_POLL_SECONDS)
        if process.poll() is not None:
            return

        reason = None
        age = _heartbeat_age_seconds()
        runtime = (_now_dt() - started_at).total_seconds()

        if age is not None and age > ENGINE_SILENCE_KILL_MINUTES * 60:
            reason = (
                f"ไม่พบสัญญาณจากระบบประมวลผลนานกว่า "
                f"{ENGINE_SILENCE_KILL_MINUTES} นาที ระบบจึงหยุดงานนี้"
            )
        elif runtime > ENGINE_MAX_RUNTIME_MINUTES * 60:
            reason = (
                f"ใช้เวลาประมวลผลเกิน {ENGINE_MAX_RUNTIME_MINUTES} นาที "
                "ระบบจึงหยุดงานนี้"
            )

        if reason:
            state["kill_reason"] = reason
            print(f"WATCHDOG|KILL|{reason}", flush=True)
            try:
                process.kill()
            except Exception:
                pass
            return


def _engine_reader(process: subprocess.Popen, state: dict):
    """Own the engine process for its whole life, independent of any session.

    This runs in a background thread. Previously the same loop ran inside the
    Streamlit script thread of whoever pressed Run, so closing that browser tab
    released the lock while the engine was still working.
    """
    try:
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
                    _notify(None, step, progress, parts[3])

        code = process.wait()

        if state.get("kill_reason"):
            raise RuntimeError(state["kill_reason"])
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

    finally:
        try:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
        except Exception:
            pass
        LOCK_FILE.unlink(missing_ok=True)


def _check_cooldown():
    if not LAST_RUN_FILE.exists():
        return
    try:
        last_run = datetime.fromisoformat(
            LAST_RUN_FILE.read_text(encoding="utf-8").strip()
        )
    except ValueError:
        return

    elapsed = _now_dt() - last_run
    cooldown = timedelta(minutes=COOLDOWN_MINUTES)
    if elapsed >= cooldown:
        return

    total_seconds = max(0, int((cooldown - elapsed).total_seconds()))
    minutes, seconds = divmod(total_seconds, 60)
    raise RuntimeError(
        f"สามารถดำเนินการประมวลผลครั้งถัดไปได้ใน "
        f"{minutes} นาที {seconds} วินาที"
    )


def start_forecast() -> int:
    """Start one forecast job and return immediately with the engine PID.

    The caller's Streamlit script run is not held open. A background reader
    thread owns the engine process, updates the status file and releases the
    lock, so the job survives page refreshes and disconnected viewers.
    """
    with _START_LOCK:
        # Recover only an obviously stale lock.
        clear_stale_lock_if_safe()

        if LOCK_FILE.exists():
            raise RuntimeError("ระบบกำลังประมวลผลข้อมูลอยู่")

        _check_cooldown()

        if not ENGINE_FILE.exists():
            raise FileNotFoundError(
                "ไม่พบ forecast_engine.py สำหรับดำเนินการประมวลผล"
            )

        # st.secrets must be read here, in the Streamlit script thread.
        env = _engine_env()
        env["DASHBOARD_OUTPUT_DIR"] = str(OUTPUT_DIR)
        # The public button builds only what the dashboard displays.
        # The 12h/6h/3h products come from the scheduled FORECAST_MODE=full job.
        env["FORECAST_MODE"] = "dashboard"

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
            engine_pid=None,
            preserve_started_at=False,
        )

        try:
            process = subprocess.Popen(
                [sys.executable, "-u", str(ENGINE_FILE)],
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
        except Exception as exc:
            LOCK_FILE.unlink(missing_ok=True)
            _write_status(
                running=False,
                step=1,
                progress=0,
                message="ไม่สามารถเริ่มกระบวนการประมวลผลได้",
                error=str(exc),
                health="error",
                anomaly="ไม่สามารถเปิดกระบวนการประมวลผลได้",
            )
            raise

        # Save actual child process PID.
        status = _read_status()
        status["engine_pid"] = process.pid
        _write_payload(status)

        _notify(None, 1, 5, "Preparing system...")

        state: dict = {"kill_reason": None}
        threading.Thread(
            target=_engine_reader,
            args=(process, state),
            name="forecast-engine-reader",
            daemon=True,
        ).start()
        threading.Thread(
            target=_engine_watchdog,
            args=(process, _now_dt(), state),
            name="forecast-engine-watchdog",
            daemon=True,
        ).start()

        return process.pid


def run_forecast(callback: Callable | None = None) -> int:
    """Backward-compatible alias. The job no longer blocks the caller."""
    return start_forecast()


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
