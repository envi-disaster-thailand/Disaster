"""Print the Streamlit Secrets block for a service-account key file.

    python make_secrets_toml.py path\\to\\key.json

The key file is only read locally and the output is printed to this terminal.
Copy the printed block into Streamlit Secrets. Never commit the key file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

FIELD_ORDER = [
    "type",
    "project_id",
    "private_key_id",
    "private_key",
    "client_email",
    "client_id",
    "auth_uri",
    "token_uri",
    "auth_provider_x509_cert_url",
    "client_x509_cert_url",
    "universe_domain",
]

REQUIRED = ("client_email", "private_key", "token_uri")


def toml_string(value: str) -> str:
    """Quote a value as a TOML basic string."""
    escaped = (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )
    return f'"{escaped}"'


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        return 1

    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Not a valid JSON key file: {exc}")
        return 1

    missing = [k for k in REQUIRED if not info.get(k)]
    if missing:
        print(f"This does not look like a service-account key. Missing: {missing}")
        return 1

    if info.get("type") != "service_account":
        print(f"Warning: 'type' is {info.get('type')!r}, expected 'service_account'.")

    keys = [k for k in FIELD_ORDER if k in info]
    keys += [k for k in info if k not in FIELD_ORDER]

    print()
    print("# ---- copy everything below into Streamlit Secrets ----")
    print()
    print('GOOGLE_DRIVE_FOLDER_ID = "PUT_YOUR_EXISTING_FOLDER_ID_HERE"')
    print()
    print("[GOOGLE_SERVICE_ACCOUNT_JSON]")
    for key in keys:
        print(f"{key} = {toml_string(info[key])}")
    print()
    print("# ---- end ----")
    print()
    print(f"Service account: {info['client_email']}")
    print("This address must be a Content manager on the destination Shared Drive.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
