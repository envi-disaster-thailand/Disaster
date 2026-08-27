from __future__ import annotations

import mimetypes
import os
from pathlib import Path

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"


def _secret(name: str) -> str:
    """Environment first, Streamlit secrets second.

    forecast_engine.py runs as a plain python subprocess with a changed
    working directory, so st.secrets is not reliably resolvable there.
    forecast_runner.py forwards the values through the environment instead.
    """
    value = os.environ.get(name, "")
    if value:
        return value.strip()

    try:
        import streamlit as st

        value = st.secrets.get(name, "")
    except Exception:
        value = ""

    return "" if value is None else str(value).strip()


def get_drive_write_service():
    required = (
        "GOOGLE_CLIENT_ID",
        "GOOGLE_CLIENT_SECRET",
        "GOOGLE_REFRESH_TOKEN",
        "GOOGLE_DRIVE_FOLDER_ID",
    )
    missing = [k for k in required if not _secret(k)]
    if missing:
        raise RuntimeError(f"Missing Streamlit Secrets: {', '.join(missing)}")

    creds = Credentials(
        token=None,
        refresh_token=_secret("GOOGLE_REFRESH_TOKEN"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=_secret("GOOGLE_CLIENT_ID"),
        client_secret=_secret("GOOGLE_CLIENT_SECRET"),
        scopes=[DRIVE_SCOPE],
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _escape_q(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")


def find_child_folder(parent_id: str, name: str):
    service = get_drive_write_service()
    q = (
        f"'{parent_id}' in parents and trashed = false and "
        "mimeType = 'application/vnd.google-apps.folder' and "
        f"name = '{_escape_q(name)}'"
    )
    resp = service.files().list(
        q=q,
        fields="files(id,name,modifiedTime)",
        pageSize=100,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    files = resp.get("files", [])
    return files[0] if files else None


def ensure_run_folder(parent_id: str, folder_name: str) -> str:
    existing = find_child_folder(parent_id, folder_name)
    if existing:
        return existing["id"]

    service = get_drive_write_service()
    created = service.files().create(
        body={
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id],
        },
        fields="id,name",
        supportsAllDrives=True,
    ).execute()
    return created["id"]


def find_file_in_folder(folder_id: str, name: str):
    service = get_drive_write_service()
    q = (
        f"'{folder_id}' in parents and trashed = false and "
        "mimeType != 'application/vnd.google-apps.folder' and "
        f"name = '{_escape_q(name)}'"
    )
    resp = service.files().list(
        q=q,
        fields="files(id,name,modifiedTime,size)",
        pageSize=100,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    files = resp.get("files", [])
    return files[0] if files else None


def upload_file(local_path: str | Path, folder_id: str):
    local_path = Path(local_path)
    service = get_drive_write_service()
    mime = mimetypes.guess_type(local_path.name)[0] or "application/octet-stream"
    media = MediaFileUpload(str(local_path), mimetype=mime, resumable=True)

    existing = find_file_in_folder(folder_id, local_path.name)
    if existing:
        return service.files().update(
            fileId=existing["id"],
            media_body=media,
            fields="id,name,modifiedTime,size",
            supportsAllDrives=True,
        ).execute()

    return service.files().create(
        body={"name": local_path.name, "parents": [folder_id]},
        media_body=media,
        fields="id,name,modifiedTime,size",
        supportsAllDrives=True,
    ).execute()
