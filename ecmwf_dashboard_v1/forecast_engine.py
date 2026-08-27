"""
ECMWF forecast engine adapter
=============================

This file is intentionally small in Dashboard V1.

Paste / move the processing functions from the existing Colab notebook here,
then keep the same PNG generation logic, especially PART A. The dashboard
expects the same map style and satellite overlays.

Server-side secrets must NOT be written directly in this file.
Use environment variables or Google Secret Manager.

Required PART A output filename pattern:
    ecmwf-24hr-day1-...-Local-time.png
    ...
    ecmwf-24hr-day10-...-Local-time.png

The dashboard reads these PNG files from DASHBOARD_OUTPUT_DIR.

When integrating the original notebook, call status() at the indicated places.
"""

from __future__ import annotations

import os
from pathlib import Path

OUTPUT_DIR = Path(os.getenv("DASHBOARD_OUTPUT_DIR", "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def status(step: int, progress: int, message: str):
    print(f"STATUS|{step}|{progress}|{message}", flush=True)


def main():
    status(1, 5, "Preparing system...")

    # ---------------------------------------------------------
    # STEP 2 — ECMWF download
    # Move the notebook ECMWF Client.retrieve(...) section here.
    # ---------------------------------------------------------
    status(2, 15, "Downloading ECMWF forecast...")
    # TODO: integrate notebook download code

    # ---------------------------------------------------------
    # STEP 3 — Read GRIB + Thailand mask
    # ---------------------------------------------------------
    status(3, 30, "Reading and preparing rainfall data...")
    # TODO: integrate load_grib(), Natural Earth boundaries,
    #       Thailand mask, etc.

    # ---------------------------------------------------------
    # STEP 4 — Satellite acquisition plans + ground tracks
    # THIS IS REQUIRED. DO NOT REMOVE.
    # ---------------------------------------------------------
    status(4, 45, "Loading satellite acquisition plans and ground tracks...")
    # TODO: integrate:
    #   fetch_sat_plan(...)
    #   Sentinel-1 fallback logic
    #   ground-track / TLE logic
    #
    # Preserve:
    #   Sentinel-1C / Sentinel-1D
    #   RADARSAT-2
    #   COSMO-SkyMed
    # footprints and ground tracks.

    # ---------------------------------------------------------
    # STEP 5 — PART A rainfall accumulation
    # ---------------------------------------------------------
    status(5, 58, "Processing 24-hour accumulated rainfall...")
    # TODO: integrate panels_24h calculation exactly from notebook.

    # ---------------------------------------------------------
    # STEP 6 — Generate Day 1–Day 10 maps
    # Keep original draw_panel() styling unchanged.
    # ---------------------------------------------------------
    status(6, 70, "Generating Day 1 map with satellite overlays...")

    # Example for actual implementation:
    #
    # for idx, p in enumerate(panels_24h, start=1):
    #     progress = 70 + int((idx / 10) * 20)
    #     status(
    #         6,
    #         progress,
    #         f"Generating Day {idx} map with satellite footprints and ground tracks..."
    #     )
    #
    #     output_name = OUTPUT_DIR / p["fn"]
    #     draw_panel(
    #         ax,
    #         p["data"],
    #         RAIN_LEVELS_24H,
    #         RAIN_COLORS_24H,
    #         tl1,
    #         tl2,
    #         "",
    #         footprints=p["fps"],
    #         ground_tracks=p["gts"],
    #     )
    #     fig.savefig(output_name, dpi=150, bbox_inches="tight",
    #                 facecolor="white")

    # ---------------------------------------------------------
    # STEP 7 — Google Shared Drive
    # Use a service account / workload identity on the server.
    # Do NOT request email/password from web users.
    # ---------------------------------------------------------
    status(7, 95, "Saving results to Google Drive...")
    # TODO: upload PART A/B/C/D outputs to Shared Drive.

    # PART B / C / D may still be calculated and uploaded,
    # but the web dashboard displays PART A only.

    # Safety guard in V1:
    raise RuntimeError(
        "Dashboard UI is ready, but forecast_engine.py still needs the "
        "existing notebook processing code integrated before RUN is enabled in production."
    )


if __name__ == "__main__":
    main()
