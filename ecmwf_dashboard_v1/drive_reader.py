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


# The engine uploads 24h, 12h, 06h and 03h products into the SAME run
# folder, and they all contain "dayN" in the name. Matching "dayN" alone
# would let a 3-hour panel win the Day 6 slot, because the 3-hour files
# are uploaded last and _day_pngs_in_folder keeps the newest match.
DAY_PNG_RE = re.compile(r"^ecmwf-24hr-day(\d{1,2})[-_.]", re.I)


def _extract_day(name: str) -> Optional[int]:
    m = DAY_PNG_RE.match(name)
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
    if root_files:
        return None, root_files

    # Otherwise inspect newest child folders until a forecast set is found.
    children = _list_children(root_id)
    folders = [x for x in children if x.get("mimeType") == FOLDER_MIME]

    for folder in folders:
        day_files = _day_pngs_in_folder(folder["id"])
        if day_files:
            return folder, day_files

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
