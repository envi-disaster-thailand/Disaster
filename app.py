from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st

from forecast_runner import (
    STATUS_FILE,
    LOCK_FILE,
    LAST_RUN_FILE,
    COOLDOWN_MINUTES,
    run_forecast,
    get_latest_day_images,
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


def load_status() -> dict:
    if not STATUS_FILE.exists():
        return {
            "running": False,
            "step": 0,
            "message": "พร้อมดำเนินการ",
            "progress": 0,
            "updated_at": None,
            "error": None,
        }
    try:
        return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {
            "running": False,
            "step": 0,
            "message": "พร้อมดำเนินการ",
            "progress": 0,
            "updated_at": None,
            "error": None,
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


def render_status(status: dict):
    st.subheader("สถานะการประมวลผล")

    progress = int(status.get("progress", 0))
    st.progress(max(0, min(progress, 100)) / 100)

    active_step = int(status.get("step", 0))
    running = bool(status.get("running", False))

    for idx, label in enumerate(STEPS, start=1):
        if idx < active_step:
            icon = "✅"
        elif idx == active_step and running:
            icon = "🔄"
        elif idx == active_step and not running and progress == 100:
            icon = "✅"
        else:
            icon = "○"
        st.markdown(f"{icon} **{idx}. {label}**")

    message = status.get("message")
    if message:
        st.info(message)

    if status.get("error"):
        st.error(status["error"])

    if status.get("updated_at"):
        st.caption(f"ปรับปรุงสถานะล่าสุด: {status['updated_at']}")


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
    st.image(str(images[selected]), width='stretch')
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
        st.image(
            image_bytes,
            width='stretch',
            caption=metadata.get("name", ""),
        )
    except Exception as exc:
        st.error(f"ไม่สามารถดาวน์โหลดภาพจาก Google Drive ได้: {exc}")
        return False

    if folder:
        st.caption(
            f"ชุดข้อมูลล่าสุดจาก Google Drive: {folder.get('name','')} "
            f"• ปรับปรุง {folder.get('modifiedTime','')}"
        )
    else:
        st.caption("แสดงผลจากโฟลเดอร์ Google Drive ที่กำหนด")

    return True


st.title("ข้อมูลพยากรณ์ปริมาณฝนจากแบบจำลอง ECMWF")
st.markdown(
    '<div class="subtitle">ปริมาณฝนสะสม 24 ชั่วโมง และแผนการถ่ายภาพดาวเทียม SAR</div>',
    unsafe_allow_html=True,
)

status = load_status()
is_running = bool(status.get("running", False)) or LOCK_FILE.exists()
remaining = cooldown_remaining()
in_cooldown = remaining > 0
disable_run = is_running or in_cooldown

left, right = st.columns([1, 2])

with left:
    run_clicked = st.button(
        "▶ ดำเนินการประมวลผลข้อมูล",
        type="primary",
        width='stretch',
        disabled=disable_run,
    )

with right:
    if is_running:
        st.warning("ระบบกำลังประมวลผลข้อมูล ไม่สามารถดำเนินการซ้ำในขณะนี้")
    elif in_cooldown:
        minutes, seconds = divmod(remaining, 60)
        st.info(
            f"สามารถดำเนินการประมวลผลครั้งถัดไปได้ใน "
            f"{minutes} นาที {seconds:02d} วินาที"
        )
    elif status.get("progress") == 100 and not status.get("error"):
        st.success("การประมวลผลข้อมูลล่าสุดเสร็จสมบูรณ์")
    else:
        st.caption("สามารถดำเนินการประมวลผลได้โดยไม่ต้องเข้าสู่ระบบ")

if run_clicked:
    status_area = st.empty()

    def callback(step: int, progress: int, message: str):
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


st.divider()
st.header("ปริมาณฝนสะสม 24 ชั่วโมง")
st.caption(
    "แสดงผลแผนที่พยากรณ์พร้อม Satellite Footprint และ Ground Track "
    "ตามผลผลิตจากกระบวนการประมวลผล"
)

# Dashboard displays only private Shared Drive 24-hour Day 1-Day 10 products.
shown = display_private_drive_maps()
if not shown:
    shown = display_local_maps()

if not shown:
    st.info(
        "ยังไม่พบผลการพยากรณ์ปริมาณฝนสะสม 24 ชั่วโมง "
        "กรุณาดำเนินการประมวลผลข้อมูล"
    )

st.divider()
st.caption("Dashboard แสดงเฉพาะข้อมูลปริมาณฝนสะสม 24 ชั่วโมง Day 1–Day 10")
