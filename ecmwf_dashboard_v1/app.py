from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path

import streamlit as st
print("APP_VERSION|V9.11_DRIVE_ENV_FIX", flush=True)

from forecast_runner import (
    STATUS_FILE,
    LOCK_FILE,
    LAST_RUN_FILE,
    COOLDOWN_MINUTES,
    STALE_WARNING_MINUTES,
    run_forecast,
    get_latest_day_images,
    status_health,
    clear_stale_lock_if_safe,
)
from drive_reader import (
    drive_is_configured,
    find_latest_forecast_set,
    download_drive_file,
    clear_drive_cache,
)

st.set_page_config(
    page_title="ข้อมูลพยากรณ์ปริมาณฝนจากแบบจำลอง ECMWF",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown("""
<style>
div[data-testid="stButton"] > button[kind="primary"]:not(:disabled) {
    background-color: #2E7D5B !important;
    border-color: #2E7D5B !important;
    color: #FFFFFF !important;
}
div[data-testid="stButton"] > button[kind="primary"]:not(:disabled):hover {
    background-color: #25684B !important;
    border-color: #25684B !important;
    color: #FFFFFF !important;
}
div[data-testid="stButton"] > button[kind="primary"]:not(:disabled):focus {
    background-color: #2E7D5B !important;
    border-color: #2E7D5B !important;
    color: #FFFFFF !important;
}
div[data-testid="stButton"] > button[kind="primary"]:disabled {
    background-color: #F0F2F6 !important;
    border-color: #D9DDE3 !important;
    color: #A3A8B0 !important;
    opacity: 1 !important;
    cursor: not-allowed !important;
}
</style>

<style>
/* Equal-size Run and processing-status tools */
div[data-testid="stButton"] > button {
    width: 100% !important;
}
div[data-testid="stAlert"] {
    width: 100% !important;
    box-sizing: border-box !important;
}
</style>

""", unsafe_allow_html=True)


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1180px;
    }
    h1 { margin-bottom: 0.15rem; }
    .subtitle {
        color: #59636e;
        margin-bottom: 1.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

STEPS = [
    "เตรียมระบบประมวลผล",
    "ดาวน์โหลดข้อมูลพยากรณ์จาก ECMWF",
    "เตรียมข้อมูลปริมาณฝน",
    "นำเข้าข้อมูลแผนการถ่ายภาพดาวเทียม",
    "คำนวณปริมาณฝนสะสม 24 ชั่วโมง",
    "จัดทำแผนที่พยากรณ์ Day 1–Day 10",
    "บันทึกผลการประมวลผล",
]



ICT = ZoneInfo("Asia/Bangkok")
UTC = ZoneInfo("UTC")

def format_ict(dt_text):
    """Convert stored UTC/ISO status timestamps to Thailand ICT for display only."""
    if not dt_text:
        return "-"
    try:
        dt = datetime.fromisoformat(str(dt_text))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(ICT).strftime("%Y-%m-%d %H:%M:%S ICT")
    except Exception:
        return str(dt_text)

def load_status() -> dict:
    if not STATUS_FILE.exists():
        return {
            "running": False,
            "step": 0,
            "message": "พร้อมดำเนินการ",
            "progress": 0,
            "updated_at": None,
            "heartbeat_at": None,
            "started_at": None,
            "error": None,
            "health": "normal",
            "anomaly": None,
            "detail": None,
        }
    try:
        return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {
            "running": False,
            "step": 0,
            "message": "ไม่สามารถอ่านสถานะล่าสุดได้",
            "progress": 0,
            "updated_at": None,
            "heartbeat_at": None,
            "started_at": None,
            "error": "ไฟล์สถานะไม่สมบูรณ์",
            "health": "error",
            "anomaly": "ไม่สามารถอ่านลำดับการทำงานจากไฟล์สถานะได้",
            "detail": None,
        }


def cooldown_remaining() -> int:
    if not LAST_RUN_FILE.exists():
        return 0
    try:
        last_run = datetime.fromisoformat(
            LAST_RUN_FILE.read_text(encoding="utf-8").strip()
        )
    except Exception:
        return 0

    end = last_run + timedelta(minutes=COOLDOWN_MINUTES)
    return max(0, int((end - datetime.now()).total_seconds()))


def _thai_message(message: str | None) -> str | None:
    translations = {
        "Preparing system...": "กำลังเตรียมระบบประมวลผล",
        "Downloading ECMWF forecast...": "กำลังดาวน์โหลดข้อมูลพยากรณ์จาก ECMWF",
        "Reading and preparing rainfall data...": "กำลังเตรียมข้อมูลปริมาณฝน",
        "Loading satellite acquisition plans...": "กำลังนำเข้าข้อมูลแผนการถ่ายภาพดาวเทียม",
        "Computing satellite ground tracks...": "กำลังคำนวณแนวการเคลื่อนที่ภาคพื้นดินของดาวเทียม",
        "Processing 24-hour accumulated rainfall...": "กำลังคำนวณปริมาณฝนสะสม 24 ชั่วโมง",
        "Processing completed.": "ประมวลผลข้อมูลเสร็จสมบูรณ์",
    }
    if message in translations:
        return translations[message]
    if message and message.startswith("Generating Day "):
        return message.replace("Generating Day ", "กำลังจัดทำแผนที่ Day ").replace(
            " map with satellite footprints and ground tracks...",
            " พร้อม Satellite Footprint และ Ground Track"
        )
    return message


def render_status(status: dict):
    health = status_health(status)

    st.subheader("สถานะการประมวลผล")

    progress = int(status.get("progress", 0) or 0)
    st.progress(max(0, min(progress, 100)) / 100)

    active_step = int(status.get("step", 0) or 0)
    running = bool(status.get("running", False))

    for idx, label in enumerate(STEPS, start=1):
        if idx < active_step:
            icon = "✅"
        elif idx == active_step and running:
            icon = "🔄"
        elif idx == active_step and not running and progress == 100:
            icon = "✅"
        elif (
            not running
            and status.get("error")
            and idx == active_step
            and active_step > 0
        ):
            icon = "❌"
        else:
            icon = "○"
        st.markdown(f"{icon} **{idx}. {label}**")

    message = _thai_message(status.get("message"))
    if message:
        if health["health"] == "normal":
            st.info(message)
        elif health["health"] == "warning":
            st.warning(message)
        else:
            st.error(message)

    # Easy-to-understand process health.
    if health["health"] == "normal" and running:
        st.success("ลำดับการทำงานปกติ และยังได้รับสัญญาณตอบกลับจากระบบประมวลผล")
    elif health["health"] == "warning":
        st.warning(
            "ตรวจพบสถานะที่ควรเฝ้าระวัง: "
            + (health.get("anomaly") or "กระบวนการอัปเดตช้ากว่าปกติ")
        )
    elif health["health"] == "error":
        st.error(
            "ตรวจพบความผิดปกติของกระบวนการ: "
            + (health.get("anomaly") or status.get("error") or "ไม่ทราบสาเหตุ")
        )

    detail = status.get("detail")
    if detail and detail != status.get("message"):
        st.caption(f"กิจกรรมล่าสุด: {detail}")

    if status.get("error"):
        st.error(f"รายละเอียดข้อผิดพลาด: {status['error']}")

    if status.get("started_at"):
        st.caption(f"เริ่มประมวลผล: {format_ict(status.get('started_at'))}")
    if status.get("updated_at"):
        st.caption(f"ปรับปรุงสถานะล่าสุด: {format_ict(status.get('updated_at'))}")

    pid = health.get("pid")
    pid_alive = health.get("pid_alive")
    if running and pid is not None:
        if pid_alive is True:
            st.caption(f"Process ID {pid}: ยังทำงานอยู่")
        elif pid_alive is False:
            st.caption(f"Process ID {pid}: ไม่พบกระบวนการแล้ว")

    age = health.get("heartbeat_age_seconds")
    if running and age is not None:
        if age < 60:
            st.caption(f"ได้รับสัญญาณจากระบบล่าสุดเมื่อ {age} วินาทีที่ผ่านมา")
        else:
            mins = age // 60
            st.caption(f"ได้รับสัญญาณจากระบบล่าสุดประมาณ {mins} นาทีที่ผ่านมา")



def show_forecast_map(image, caption=None):
    """Show forecast maps centered at 90% dashboard width."""
    _, center, _ = st.columns([0.05, 0.90, 0.05], gap=None)
    with center:
        if caption:
            st.image(image, width="stretch", caption=caption)
        else:
            st.image(image, width="stretch")


def display_local_maps() -> bool:
    images = get_latest_day_images()
    if not images:
        return False

    labels = list(images.keys())
    selected = st.radio(
        "ช่วงเวลาพยากรณ์",
        labels,
        horizontal=True,
        label_visibility="collapsed",
        key="local_days",
    )
    show_forecast_map(str(images[selected]))
    return True


def display_private_drive_maps() -> bool:
    if not drive_is_configured():
        return False

    try:
        folder, files = find_latest_forecast_set()
    except Exception as exc:
        st.warning(f"ไม่สามารถเชื่อมต่อ Google Drive ได้: {exc}")
        return False

    if not files:
        return False

    available_days = sorted(files.keys())
    labels = [f"Day {d}" for d in available_days]
    selected_label = st.radio(
        "ช่วงเวลาพยากรณ์",
        labels,
        horizontal=True,
        label_visibility="collapsed",
        key="drive_days",
    )
    selected_day = int(selected_label.split()[1])
    metadata = files[selected_day]

    try:
        image_bytes = download_drive_file(metadata["id"])
        show_forecast_map(image_bytes, caption=metadata.get("name", ""))
    except Exception as exc:
        st.error(f"ไม่สามารถดาวน์โหลดภาพจาก Google Drive ได้: {exc}")
        return False

    if folder:
        st.caption(
            f"ชุดข้อมูลที่กำลังแสดง: {folder.get('name','')} "
            f"• ปรับปรุง {folder.get('modifiedTime','')}"
        )
    return True


st.title("ข้อมูลพยากรณ์ปริมาณฝนจากแบบจำลอง ECMWF")
st.markdown(
    '<div class="subtitle">ปริมาณฝนสะสม 24 ชั่วโมง และแผนการถ่ายภาพดาวเทียม SAR</div>',
    unsafe_allow_html=True,
)


@st.fragment(run_every="10s")
def status_auto_refresh():
    """Check process status every 10 seconds without adding visible dashboard text."""
    current = load_status()
    active = bool(current.get("running")) or LOCK_FILE.exists()
    if active:
        _ = status_health(current)

# Recover only a clearly stale (>45 min) lock.
clear_stale_lock_if_safe()

status = load_status()
health = status_health(status)
is_running = bool(status.get("running", False)) or LOCK_FILE.exists()
remaining = cooldown_remaining()
in_cooldown = remaining > 0
disable_run = is_running or in_cooldown

left, right = st.columns([1, 1])

with left:
    run_clicked = st.button(
        "▶ ดำเนินการประมวลผลข้อมูล",
        type="primary",
        width="stretch",
        disabled=disable_run,
    )

with right:
    if is_running:
        if health["health"] == "normal":
            st.warning("ระบบกำลังประมวลผลข้อมูล ไม่สามารถดำเนินการซ้ำในขณะนี้")
        elif health["health"] == "warning":
            st.warning("ระบบยังอยู่ระหว่างประมวลผล แต่ตรวจพบการอัปเดตที่ล่าช้าหรือผิดลำดับ")
        else:
            st.error("ตรวจพบสถานะการประมวลผลผิดปกติ กรุณาตรวจสอบรายละเอียดด้านล่าง")
    elif in_cooldown:
        minutes, seconds = divmod(remaining, 60)
        st.info(
            f"สามารถดำเนินการประมวลผลครั้งถัดไปได้ใน "
            f"{minutes} นาที {seconds:02d} วินาที"
        )
    elif status.get("progress") == 100 and not status.get("error"):
        st.success("การประมวลผลข้อมูลล่าสุดเสร็จสมบูรณ์")
    elif status.get("error"):
        st.error("การประมวลผลครั้งล่าสุดไม่สำเร็จ กรุณาดูสถานะด้านล่าง")
    else:
        st.caption("สามารถดำเนินการประมวลผลได้โดยไม่ต้องเข้าสู่ระบบ")


# ONE status panel only. It is visible to every viewer, not only the person
# who pressed Run.
if (
    is_running
    or status.get("progress", 0) > 0
    or status.get("error")
    or status.get("anomaly")
):
    render_status(status)


if run_clicked:
    status_area = st.empty()

    def callback(step: int, progress: int, message: str):
        # Update the same conceptual status panel during this session.
        # The persistent status file lets other users see the same state.
        with status_area.container():
            render_status(load_status())

    try:
        run_forecast(callback=callback)
        clear_drive_cache()
        st.success("ประมวลผลข้อมูลเสร็จสมบูรณ์")
        time.sleep(0.5)
        st.rerun()
    except Exception as exc:
        st.error(f"เกิดข้อผิดพลาดในการประมวลผล: {exc}")
        time.sleep(0.5)
        st.rerun()


st.divider()
status_auto_refresh()

st.header("ปริมาณฝนสะสม 24 ชั่วโมง")

# Make it explicit when maps are from the previous completed run.
if is_running:
    st.info(
        "กำลังประมวลผลข้อมูลรอบใหม่ แผนที่ด้านล่างเป็นผลจากรอบที่เสร็จสมบูรณ์ล่าสุด "
        "และจะเปลี่ยนเป็นรอบใหม่เมื่อจัดทำผลครบถ้วน"
    )

st.caption(
    "แสดงผลแผนที่พยากรณ์พร้อม Satellite Footprint และ Ground Track "
    "ตามผลผลิตจากกระบวนการประมวลผล"
)

shown = display_private_drive_maps()
if not shown:
    shown = display_local_maps()

if not shown:
    st.info(
        "ยังไม่พบผลการพยากรณ์ปริมาณฝนสะสม 24 ชั่วโมง "
        "กรุณาดำเนินการประมวลผลข้อมูล"
    )

st.divider()
