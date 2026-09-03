from __future__ import annotations

import io
import re
from typing import Dict, Optional, Tuple

import streamlit as st
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

DRIVE_READONLY_SCOPE = "https://www.googleapis.com/auth/drive"
FOLDER_MIME = "application/vnd.google-apps.folder"

# A finished run folder holds Day 1 to Day 10.
EXPECTED_DAYS = 10

# The run folder being written is always the newest one, so during a run the
# newest folder is incomplete. Look past it, but keep the number of Drive
# calls bounded.
MAX_FOLDERS_TO_INSPECT = 12


def _secret(name: str) -> str:
    value = st.secrets.get(name, "")
    if value is None:
        return ""
    return str(value).strip()


def drive_is_configured() -> bool:
    required = (
        "GOOGLE_CLIENT_ID",
        "GOOGLE_CLIENT_SECRET",
        "GOOGLE_REFRESH_TOKEN",
        "GOOGLE_DRIVE_FOLDER_ID",
    )
    return all(_secret(k) for k in required)


@st.cache_resource(show_spinner=False)
def get_drive_service():
    if not drive_is_configured():
        raise RuntimeError("ยังไม่ได้กำหนดค่า OAuth สำหรับ Google Drive ใน Streamlit Secrets")

    creds = Credentials(
        token=None,
        refresh_token=_secret("GOOGLE_REFRESH_TOKEN"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=_secret("GOOGLE_CLIENT_ID"),
        client_secret=_secret("GOOGLE_CLIENT_SECRET"),
        scopes=[DRIVE_READONLY_SCOPE],
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


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


@st.cache_data(ttl=60, show_spinner=False)
def find_latest_forecast_set() -> Tuple[Optional[dict], Dict[int, dict]]:
    """
    GOOGLE_DRIVE_FOLDER_ID can point either to:
      1) the PNG parent containing dated run folders, or
      2) a folder that directly contains Day 1-Day 10 PNG files.

    Returns:
      latest_folder_metadata (or None for the root itself),
      {1: file_metadata, ..., 10: file_metadata}
    """
    root_id = _secret("GOOGLE_DRIVE_FOLDER_ID")
    if not root_id:
        return None, {}

    # First: allow root itself to contain the PNG files.
    root_files = _day_pngs_in_folder(root_id)
    if len(root_files) >= EXPECTED_DAYS:
        return None, root_files

    # Otherwise inspect newest child folders. Prefer a complete Day 1-Day 10
    # set: the newest folder is the one currently being written during a run,
    # and returning it would show a half-finished forecast as the latest one.
    children = _list_children(root_id)
    folders = [x for x in children if x.get("mimeType") == FOLDER_MIME]

    best_folder = None
    best_files: Dict[int, dict] = {}

    for folder in folders[:MAX_FOLDERS_TO_INSPECT]:
        day_files = _day_pngs_in_folder(folder["id"])
        if len(day_files) >= EXPECTED_DAYS:
            return dict(folder, complete=True), day_files
        if len(day_files) > len(best_files):
            best_folder, best_files = folder, day_files

    # No complete run folder found. Show the most complete thing available so
    # the page is not empty before the first run has ever finished.
    if root_files and len(root_files) >= len(best_files):
        return None, root_files
    if best_files:
        return dict(best_folder, complete=False), best_files

    return None, {}


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
    find_latest_forecast_set.clear()
    download_drive_file.clear()
