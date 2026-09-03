from __future__ import annotations

import io
import re
from typing import Dict, Optional, Tuple

import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from drive_auth import (
    build_credentials,
    credentials_available,
    credentials_mode,
    secret as _secret,
)

FOLDER_MIME = "application/vnd.google-apps.folder"

# A finished run folder holds Day 1 to Day 10.
EXPECTED_DAYS = 10

# The run folder being written is always the newest one, so during a run the
# newest folder is incomplete. Look past it, but keep the number of Drive
# calls bounded.
MAX_FOLDERS_TO_INSPECT = 12


def drive_is_configured() -> bool:
    return bool(_secret("GOOGLE_DRIVE_FOLDER_ID")) and credentials_available()


@st.cache_resource(show_spinner=False)
def get_drive_service():
    if not drive_is_configured():
        raise RuntimeError(
            "ยังไม่ได้กำหนดค่าการเชื่อมต่อ Google Drive ใน Streamlit Secrets"
        )
    return build(
        "drive", "v3",
        credentials=build_credentials(),
        cache_discovery=False,
    )


def _list_children(parent_id: str):
    service = get_drive_service()
    items = []
    page_token = None

    while True:
        response = service.files().list(
            q=f"'{parent_id}' in parents and trashed = false",
            fields="nextPageToken, files(id,name,mimeType,modifiedTime,createdTime)",
            orderBy="modifiedTime desc",
            pageSize=1000,
            pageToken=page_token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()

        items.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return items


# All four product sets use the same "dayN" token in their filenames:
#   ecmwf-24hr-day1-...   the Day 1-Day 10 maps this dashboard shows
#   ecmwf-12hr-day1-...   ecmwf-06hr-day1-...   ecmwf-03hr-day1-...
# The 3-hour set is uploaded last, so matching on "dayN" alone picked the
# newest file for each day and displayed a 3-hour map under the 24-hour
# heading. The 24-hour marker is required.
DAY_FILE_RE = re.compile(
    r"(?:^|[-_])24-?hr?[-_]?day[\s_-]?(\d{1,2})(?:[-_.]|$)",
    flags=re.I,
)


def _extract_day(name: str) -> Optional[int]:
    """Day number of a 24-hour accumulation map, or None for anything else."""
    m = DAY_FILE_RE.search(name)
    if not m:
        return None
    day = int(m.group(1))
    return day if 1 <= day <= 10 else None


def _day_pngs_in_folder(folder_id: str) -> Dict[int, dict]:
    result: Dict[int, dict] = {}
    for item in _list_children(folder_id):
        name = item.get("name", "")
        if item.get("mimeType") == FOLDER_MIME:
            continue
        if not name.lower().endswith(".png"):
            continue
        day = _extract_day(name)
        if day is None:
            continue

        # children are already sorted by modifiedTime desc,
        # so the first match for each day is the newest file.
        result.setdefault(day, item)

    return result


@st.cache_data(ttl=30, show_spinner=False)
def find_forecast_sets():
    """Return (newest_folder, newest_files, complete_folder, complete_files).

    newest_*   the most recent folder holding any 24-hour day map, including a
               run still in progress, so Day 1 is visible as soon as it lands.
    complete_* the most recent folder holding all ten days.

    They are usually the same folder. While a run is uploading they differ, and
    the dashboard says which one it is showing.
    """
    root_id = _secret("GOOGLE_DRIVE_FOLDER_ID")
    if not root_id:
        return None, {}, None, {}

    # The root folder may hold the day PNGs directly.
    root_files = _day_pngs_in_folder(root_id)
    if len(root_files) >= EXPECTED_DAYS:
        return None, root_files, None, root_files

    children = _list_children(root_id)
    folders = [x for x in children if x.get("mimeType") == FOLDER_MIME]

    newest_folder, newest_files = None, {}
    complete_folder, complete_files = None, {}

    for folder in folders[:MAX_FOLDERS_TO_INSPECT]:
        day_files = _day_pngs_in_folder(folder["id"])
        if not day_files:
            continue

        if newest_folder is None:
            newest_folder = dict(folder, complete=len(day_files) >= EXPECTED_DAYS)
            newest_files = day_files

        if len(day_files) >= EXPECTED_DAYS:
            complete_folder = dict(folder, complete=True)
            complete_files = day_files
            break

    if not newest_files and root_files:
        newest_files = root_files

    return newest_folder, newest_files, complete_folder, complete_files


def find_latest_forecast_set():
    """Backward-compatible view: the newest complete set, else the newest set."""
    newest_folder, newest_files, complete_folder, complete_files = find_forecast_sets()
    if complete_files:
        return complete_folder, complete_files
    return newest_folder, newest_files


@st.cache_data(ttl=300, show_spinner=False)
def download_drive_file(file_id: str) -> bytes:
    service = get_drive_service()
    request = service.files().get_media(
        fileId=file_id,
        supportsAllDrives=True,
    )

    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)

    done = False
    while not done:
        _, done = downloader.next_chunk()

    return buffer.getvalue()


def clear_drive_cache():
    find_forecast_sets.clear()
    download_drive_file.clear()
    # The cached service holds a Credentials object. After the refresh token is
    # replaced in Secrets, the old one would keep failing until the app process
    # restarts unless it is dropped here too.
    get_drive_service.clear()


def is_credential_error(exc: Exception) -> bool:
    """True when Drive refused the stored refresh token itself."""
    text = str(exc).lower()
    return "invalid_grant" in text or "expired or revoked" in text
