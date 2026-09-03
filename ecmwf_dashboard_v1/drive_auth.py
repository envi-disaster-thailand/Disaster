from __future__ import annotations

import json
import os
from pathlib import Path

try:
    import streamlit as st
except Exception:  # the engine subprocess can run without Streamlit
    st = None

from google.oauth2 import service_account
from google.oauth2.credentials import Credentials as OAuthCredentials

DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"

SERVICE_ACCOUNT_KEY = "GOOGLE_SERVICE_ACCOUNT_JSON"
SERVICE_ACCOUNT_FILE_KEY = "GOOGLE_APPLICATION_CREDENTIALS"
OAUTH_KEYS = ("GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "GOOGLE_REFRESH_TOKEN")


def raw_secret(name: str):
    """Environment first (the engine runs as a subprocess), then Streamlit Secrets."""
    value = os.getenv(name)
    if value not in (None, ""):
        return value
    if st is None:
        return None
    try:
        return st.secrets.get(name, None)
    except Exception:
        return None


def secret(name: str) -> str:
    value = raw_secret(name)
    return "" if value is None else str(value).strip()


def _service_account_info():
    """Service-account key from Secrets, as a JSON string or a TOML table."""
    raw = raw_secret(SERVICE_ACCOUNT_KEY)
    if raw is None or raw == "":
        return None

    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        info = None
        for kwargs in ({}, {"strict": False}):
            try:
                info = json.loads(text, **kwargs)
                break
            except json.JSONDecodeError:
                info = None
        if not isinstance(info, dict):
            return None
    else:
        try:
            info = dict(raw)
        except Exception:
            return None

    key = info.get("private_key")
    if isinstance(key, str):
        # A key pasted on one line keeps a literal backslash-n.
        info["private_key"] = key.replace("\\n", "\n")

    if not info.get("private_key") or not info.get("client_email"):
        return None
    return info


def _service_account_file():
    path = secret(SERVICE_ACCOUNT_FILE_KEY)
    if path and Path(path).exists():
        return path
    return None


def credentials_mode() -> str | None:
    """'service_account', 'oauth', or None when nothing usable is configured."""
    if _service_account_info() or _service_account_file():
        return "service_account"
    if all(secret(k) for k in OAUTH_KEYS):
        return "oauth"
    return None


def credentials_available() -> bool:
    return credentials_mode() is not None


def build_credentials():
    """Service-account credentials when available, otherwise the OAuth token.

    A service account never expires and needs no consent screen, so it is
    preferred. The OAuth path stays as a fallback so an existing deployment
    keeps working until its secrets are migrated.
    """
    info = _service_account_info()
    if info:
        return service_account.Credentials.from_service_account_info(
            info, scopes=[DRIVE_SCOPE]
        )

    path = _service_account_file()
    if path:
        return service_account.Credentials.from_service_account_file(
            path, scopes=[DRIVE_SCOPE]
        )

    missing = [k for k in OAUTH_KEYS if not secret(k)]
    if missing:
        raise RuntimeError(
            "Missing Google Drive credentials. Provide "
            f"{SERVICE_ACCOUNT_KEY}, or all of: {', '.join(missing)}"
        )

    return OAuthCredentials(
        token=None,
        refresh_token=secret("GOOGLE_REFRESH_TOKEN"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=secret("GOOGLE_CLIENT_ID"),
        client_secret=secret("GOOGLE_CLIENT_SECRET"),
        scopes=[DRIVE_SCOPE],
    )
