from __future__ import annotations

import io
import os
from pathlib import Path
from drive_writer import ensure_run_folder, upload_file
from drive_writer import _secret as _drive_secret

import matplotlib
matplotlib.use("Agg")

# Dashboard runtime output directory only.
# Data sources and processing logic below are taken from the original notebook.
OUTPUT_DIR = Path(os.getenv("DASHBOARD_OUTPUT_DIR", "outputs")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(OUTPUT_DIR)

def status(step: int, progress: int, message: str):
    print(f"STATUS|{step}|{progress}|{message}", flush=True)

status(1, 5, "Preparing system...")

import os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from ecmwf.opendata import Client
from datetime import datetime, timedelta


GRIB_FILE_24H = 'tp_thailand_24h.grib2'
GRIB_FILE_12H = 'tp_thailand_12h.grib2'
GRIB_FILE_6H  = 'tp_thailand_6h.grib2'
GRIB_FILE_3H  = 'tp_thailand_3h.grib2'

LON_MIN, LON_MAX = 92.0, 110.0
LAT_MIN, LAT_MAX = 2.0,  22.0

TZ_OFFSET = timedelta(hours=7)

STEPS_24H = list(range(24, 241, 24))
STEPS_12H = list(range(12, 241, 12))
STEPS_6H  = list(range(6, 241, 6))
STEPS_3H  = list(range(3, 145, 3))

RAIN_LEVELS_24H = [1, 5, 10, 20, 35, 60, 100, 150, 200]
RAIN_LEVELS_12H = [0.5, 2, 4, 10, 25, 50, 100, 150]
RAIN_LEVELS_6H  = [0.2, 1, 2, 5, 10, 20, 40, 75]
RAIN_LEVELS_3H  = [0.1, 0.5, 1, 3, 7, 15, 30, 60]

RAIN_COLORS_24H = [
    '#00ffff', '#0099ff', '#0000ff',
    '#cc00ff', '#ff00cc', '#ff6600',
    '#ff0000', '#8b0000',
]
RAIN_COLORS_12H = [
    '#00ffff', '#0099ff', '#0000ff',
    '#cc00ff', '#ff00cc', '#ff6600',
    '#ff0000',
]
RAIN_COLORS_6H = [
    '#00ffff', '#0099ff', '#0000ff',
    '#cc00ff', '#ff00cc', '#ff6600',
    '#ff0000',
]
RAIN_COLORS_3H = [
    '#00ffff', '#0099ff', '#0000ff',
    '#cc00ff', '#ff00cc', '#ff6600',
    '#ff0000',
]

SMOOTH_SIGMA = 1.2
OCEAN_COLOR  = '#cde8f5'
LAND_COLOR   = '#f2ede4'
# --- Satellite acquisition-plan configuration ---
SAT_PLAN_URL = 'https://disaster.gistda.or.th/sat-plan'
# Fallback: ESA Copernicus publishes Sentinel-1 acquisition plans as
# public KML files. Used only when GISTDA returns no Sentinel-1 data.
ESA_S1_PLAN_PAGE = ('https://sentinels.copernicus.eu/copernicus/'
                    'sentinel-1/acquisition-plans')
ESA_FALLBACK_SATS = {'sentinel-1c', 'sentinel-1d'}

# SAR satellites (cloud-penetrating, best for monitoring during rain)
SAT_NAMES = ['sentinel-1c', 'sentinel-1d', 'radarsat-2', 'cosmo-skymed']

SAT_STYLE = {
    'sentinel-1c':  {'color': '#fff176', 'label': 'Sentinel-1C', 'lw': 2.0},
    'sentinel-1d':  {'color': '#fff176', 'label': 'Sentinel-1D', 'lw': 2.0},
    'radarsat-2':   {'color': '#39ff14', 'label': 'RADARSAT-2',  'lw': 2.0},
    'cosmo-skymed': {'color': '#ffd600', 'label': 'COSMO-SkyMed','lw': 2.0},
}

SAT_SHORT = {
    'sentinel-1c': 'S-1C',
    'sentinel-1d': 'S-1D',
    'radarsat-2':  'RS2',
    'cosmo-skymed': 'CSK',
}

# Only keep footprints overlapping this bounding box (Thailand region)
SAT_BBOX = (LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)

SHARED_DRIVE_PATH = (
    '/content/drive/Shareddrives/New-Disaster-Water'
    '/Colab_ECMWF_Export/PNG'
)



# Fail fast: verifying credentials after a 15-minute download wastes the run.
_MISSING_DRIVE = [
    _k for _k in (
        "GOOGLE_CLIENT_ID",
        "GOOGLE_CLIENT_SECRET",
        "GOOGLE_REFRESH_TOKEN",
        "GOOGLE_DRIVE_FOLDER_ID",
    )
    if not _drive_secret(_k)
]
if _MISSING_DRIVE:
    raise RuntimeError(
        "Missing Google Drive credentials: " + ", ".join(_MISSING_DRIVE)
    )

status(2, 15, "Downloading ECMWF forecast...")

# @title
client = Client(source='ecmwf')


def load_grib(path):
    ds = xr.open_dataset(path, engine='cfgrib',
                         backend_kwargs={'indexpath': ''})
    var = [v for v in ds.data_vars
           if 'tp' in v.lower() or 'precip' in v.lower()][0]
    da = ds[var].sel(
        latitude=slice(LAT_MAX, LAT_MIN),
        longitude=slice(LON_MIN, LON_MAX),
    ) * 1000.0
    step_dim = [d for d in da.dims if 'step' in d][0]
    return da, step_dim


def to_hours(v, fb):
    try:
        return int(np.timedelta64(int(v), 'ns') / np.timedelta64(1, 'h'))
    except Exception:
        pass
    try:
        return int(v / np.timedelta64(1, 'h'))
    except Exception:
        return fb


def fetch_product(label, steps, target, fallback_steps):
    """Download one step set, materialize the Thailand subset, drop the GRIB.

    Each product is fetched only when it is about to be plotted. The
    dashboard needs the 24-hour set alone, so the 100 MB of 12h/6h/3h data
    must not stand between the run starting and Day 1-10 being published --
    especially when ECMWF answers 429 and multiurl sleeps 120 s per retry.
    """
    print(f'\nDownloading {label} steps...', flush=True)
    result = client.retrieve(
        type='fc', param='tp', step=steps, target=target,
    )
    size_mb = os.path.getsize(target) / 1e6
    print(f'{label} file size : {size_mb:.1f} MB', flush=True)

    da, sdim = load_grib(target)
    da.load()
    try:
        Path(target).unlink()
        print(f'[cleanup] Removed temporary GRIB: {target}', flush=True)
    except FileNotFoundError:
        pass

    hours = [to_hours(v, fallback_steps[k])
             for k, v in enumerate(da[sdim].values)]
    print(f'{label} steps : {hours}', flush=True)
    return da, sdim, hours, result


da_24h, sdim_24h, hours_24h, _result_24h = fetch_product(
    '24h', STEPS_24H, GRIB_FILE_24H, STEPS_24H)

run_utc = _result_24h.datetime
if not isinstance(run_utc, datetime):
    run_utc = datetime.utcfromtimestamp(run_utc.timestamp())
run_ict = run_utc + TZ_OFFSET

print(f'Model run (UTC)  : {run_utc:%Y-%m-%d %H:%M}')
print(f'Model run (ICT)  : {run_ict:%Y-%m-%d %H:%M} Local Time')

status(3, 30, "Reading and preparing rainfall data...")

lats  = da_24h.latitude.values
lons  = da_24h.longitude.values
lon2d, lat2d = np.meshgrid(lons, lats)


shpfile = shpreader.natural_earth(
    resolution='50m', category='cultural', name='admin_0_countries')
reader_c = shpreader.Reader(shpfile)

thailand_geom = None
neighbor_geoms = []
for rec in reader_c.records():
    name  = rec.attributes.get('NAME', '').strip()
    lon_c = rec.attributes.get('LABEL_X', 0)
    lat_c = rec.attributes.get('LABEL_Y', 0)
    if name == 'Thailand':
        thailand_geom = rec.geometry
    elif (LON_MIN - 5 <= lon_c <= LON_MAX + 5 and
          LAT_MIN - 5 <= lat_c <= LAT_MAX + 5):
        neighbor_geoms.append(rec.geometry)

shp_adm1 = shpreader.natural_earth(
    resolution='10m', category='cultural',
    name='admin_1_states_provinces')
reader_adm1 = shpreader.Reader(shp_adm1)
province_geoms = [
    rec.geometry for rec in reader_adm1.records()
    if rec.attributes.get('admin', '') == 'Thailand'
]

print(f'Thailand border    : {thailand_geom is not None}')
print(f'Neighbor countries : {len(neighbor_geoms)}')
print(f'Thailand provinces : {len(province_geoms)}')


from shapely.geometry import Point
from shapely.prepared import prep

def build_thailand_mask(lats, lons, geom):
    """Return boolean 2-D mask True where grid point is inside Thailand."""
    if geom is None:
        return None
    grid_lon, grid_lat = np.meshgrid(lons, lats)
    try:
        import shapely
        return shapely.contains_xy(geom, grid_lon, grid_lat)
    except AttributeError:
        prepared = prep(geom)
        mask = np.zeros((len(lats), len(lons)), dtype=bool)
        for r, lat in enumerate(lats):
            for c, lon in enumerate(lons):
                if prepared.contains(Point(lon, lat)):
                    mask[r, c] = True
        return mask

print('Building Thailand grid mask...')
thailand_mask = build_thailand_mask(lats, lons, thailand_geom)
if thailand_mask is not None:
    print(f'Mask ready — {thailand_mask.sum()} grid points inside Thailand.')
else:
    print('Thailand geometry not available; mask skipped.')


status(4, 42, "Loading satellite acquisition plans...")

# @title
import json as _json
import urllib.request
import urllib.parse
from collections import defaultdict
from datetime import datetime as _dt
from shapely.geometry import shape as _shape, box as _box


def _normalize_records(data):
    # The API may return a JSON list, a JSON string that itself encodes a
    # list (double-encoded), or a dict wrapping the list under a key.
    for _ in range(3):
        if isinstance(data, str):
            try:
                data = _json.loads(data)
            except _json.JSONDecodeError:
                return []
            continue
        if isinstance(data, dict):
            for key in ('data', 'results', 'features', 'items'):
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
            else:
                return [data]
            continue
        break
    if not isinstance(data, list):
        return []
    records = []
    for item in data:
        if isinstance(item, str):
            try:
                item = _json.loads(item)
            except _json.JSONDecodeError:
                continue
        if isinstance(item, dict):
            records.append(item)
    return records


def fetch_sat_plan(sat_name, from_date, to_date, timeout=60):
    params = urllib.parse.urlencode({
        'sat_name': sat_name,
        'from_date': from_date,
        'to_date': to_date,
    })
    url = f'{SAT_PLAN_URL}?{params}'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode('utf-8')
    try:
        data = _json.loads(raw)
    except _json.JSONDecodeError:
        print(f'  [warn] {sat_name}: response was not valid JSON')
        return []
    return _normalize_records(data)


def parse_file_time(file_name, fallback_date):
    # file_name pattern example: S1V_20260703_1035 -> UTC yyyymmdd_HHMM
    try:
        parts = file_name.split('_')
        ymd = parts[-2]
        hm = parts[-1][:4]
        return _dt.strptime(ymd + hm, '%Y%m%d%H%M')
    except Exception:
        try:
            return _dt.strptime(fallback_date, '%Y-%m-%d')
        except Exception:
            return None


def collect_footprints(sat_names, from_date, to_date, bbox):
    region = _box(bbox[0], bbox[1], bbox[2], bbox[3])
    all_fps = []
    total_raw = 0
    total_kept = 0
    for sat in sat_names:
        # GISTDA API uses generic 'sentinel-1' for all S1 variants
        _api_name = 'sentinel-1' if sat.startswith('sentinel-1') else sat
        try:
            records = fetch_sat_plan(_api_name, from_date, to_date)
        except Exception as exc:
            print(f'  [warn] {sat}: download failed ({exc})')
            continue
        kept = 0
        for rec in records:
            total_raw += 1
            if not isinstance(rec, dict):
                continue
            geom_raw = rec.get('geom')
            if not geom_raw:
                continue
            try:
                gj = _json.loads(geom_raw) if isinstance(geom_raw, str) else geom_raw
                poly = _shape(gj)
            except Exception:
                continue
            if not poly.is_valid or not poly.intersects(region):
                continue
            t_utc = parse_file_time(rec.get('file_name', ''), rec.get('date', ''))
            if t_utc is None:
                continue
            t_ict = t_utc + TZ_OFFSET
            _fn = rec.get('file_name', '')
            _sat_key = sat
            if sat.startswith('sentinel-1'):
                _fn_up = _fn.upper()
                if _fn_up.startswith('S1D') or '-1D' in _fn_up:
                    _sat_key = 'sentinel-1d'
                else:
                    _sat_key = 'sentinel-1c'
            all_fps.append({
                'sat': _sat_key,
                'geom': poly,
                't_utc': t_utc,
                't_ict': t_ict,
                'file_name': _fn,
            })
            kept += 1
        total_kept += kept
        _lbl = SAT_STYLE.get(sat, {}).get('label', sat)
        print(f'  {_lbl:<14}: {kept} footprints over region')
    all_fps.sort(key=lambda f: f['t_ict'])
    n_days = len({f['t_ict'].strftime('%Y-%m-%d') for f in all_fps})
    print(f'Total: kept {total_kept} of {total_raw} footprints '
          f'across {n_days} day(s).')
    return all_fps


plan_from = run_ict.strftime('%Y-%m-%d')
plan_to = (run_utc + timedelta(days=11)).strftime('%Y-%m-%d')
print(f'Fetching SAR acquisition plans {plan_from} to {plan_to} ...')
all_footprints = collect_footprints(
    SAT_NAMES, plan_from, plan_to, SAT_BBOX)

def footprints_in_window(t0_ict, t1_ict):
    return [f for f in all_footprints
            if t0_ict <= f['t_ict'] < t1_ict]


# @title
import re
import xml.etree.ElementTree as _ET
from shapely.geometry import Polygon as _Polygon


def _kml_strip_ns(tag):
    return tag.split('}')[-1] if '}' in tag else tag


def _kml_coords(text):
    pts = []
    for tok in text.replace('\n', ' ').split():
        parts = tok.split(',')
        if len(parts) >= 2:
            try:
                pts.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    return pts


def _kml_time(s):
    if not s:
        return None
    s = s.strip().replace('Z', '')
    for fmt in ('%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M'):
        try:
            return _dt.strptime(s, fmt)
        except ValueError:
            continue
    return None


_START_KEYS = {'observationtimestart', 'start time', 'starttime',
               'begintime', 'begin time'}


def find_esa_kml_urls(page_html, from_dt, to_dt):
    # KML file names embed their coverage window:
    # ..._MP_YYYYMMDDThhmmss_YYYYMMDDThhmmss.kml
    urls = re.findall(r'https?://[^"\'>) ]+?\.kml', page_html)
    urls = list(dict.fromkeys(urls))
    picked = []
    for u in urls:
        stamps = re.findall(r'(\d{8}T\d{6})', u)
        if len(stamps) >= 2:
            try:
                s0 = _dt.strptime(stamps[0], '%Y%m%dT%H%M%S')
                s1 = _dt.strptime(stamps[1], '%Y%m%dT%H%M%S')
            except ValueError:
                picked.append(u)
                continue
            if s1 >= from_dt and s0 <= to_dt:
                picked.append(u)
        elif re.search(r'\d{2}-\w+-\d{4}', u) or 'sentinel-1d' in u.lower():
            # date range in URL text form or S1D specific
            picked.append(u)
        else:
            picked.append(u)
    return picked


def parse_esa_kml(kml_text, region, from_dt, to_dt):
    _KML_NS = 'http://www.opengis.net/kml/2.2'
    try:
        root = _ET.fromstring(kml_text)
    except _ET.ParseError:
        return []
    out = []
    for pm in root.iter(f'{{{_KML_NS}}}Placemark'):
        # ESA KML uses <Data> elements (not SimpleData)
        fields = {}
        for d in pm.iter(f'{{{_KML_NS}}}Data'):
            v = d.find(f'{{{_KML_NS}}}value')
            if v is not None and v.text:
                fields[d.get('name', '')] = v.text.strip()
        # Also handle SimpleData for backward compatibility
        for sd in pm.iter(f'{{{_KML_NS}}}SimpleData'):
            if sd.text:
                fields[sd.get('name', '')] = sd.text.strip()
        # Time: ObservationTimeStart is authoritative; fallback to <name>
        t_str = fields.get('ObservationTimeStart', '')
        if not t_str:
            _nm = pm.find(f'{{{_KML_NS}}}name')
            t_str = _nm.text.strip() if _nm is not None and _nm.text else ''
        t_start = _kml_time(t_str)
        if t_start is None or t_start < from_dt or t_start > to_dt:
            continue
        # Satellite: SatelliteId field is authoritative (S1C / S1D)
        sat_id = fields.get('SatelliteId', '').upper()
        _esat = 'sentinel-1d' if 'S1D' in sat_id else 'sentinel-1c'
        # Geometry
        _ce = pm.find(f'.//{{{_KML_NS}}}coordinates')
        if _ce is None:
            continue
        coords = _kml_coords(_ce.text)
        if len(coords) < 3:
            continue
        try:
            poly = _Polygon(coords)
        except Exception:
            continue
        if not poly.is_valid or not poly.intersects(region):
            continue
        out.append({
            'sat': _esat,
            'geom': poly,
            't_utc': t_start,
            't_ict': t_start + TZ_OFFSET,
            'file_name': fields.get('DatatakeId', ''),
            'source': 'esa',
        })
    return out


def fetch_esa_sentinel1(from_date, to_date, bbox, timeout=90):
    region = _box(bbox[0], bbox[1], bbox[2], bbox[3])
    from_dt = _dt.strptime(from_date, '%Y-%m-%d') - timedelta(days=1)
    to_dt = _dt.strptime(to_date, '%Y-%m-%d') + timedelta(days=2)
    req = urllib.request.Request(ESA_S1_PLAN_PAGE,
                                 headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            page = resp.read().decode('utf-8', errors='ignore')
    except Exception as exc:
        print(f'  [warn] ESA plan page unreachable ({exc})')
        return []
    kml_urls = find_esa_kml_urls(page, from_dt, to_dt)
    if not kml_urls:
        print('  [warn] no matching ESA KML files found on plan page')
        return []
    collected = []
    print(f'  Found {len(kml_urls)} KML URL(s) on ESA page')
    for u in kml_urls[:8]:
        try:
            r = urllib.request.Request(u, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(r, timeout=timeout) as resp:
                kml_text = resp.read().decode('utf-8', errors='ignore')
        except Exception as exc:
            print(f'  [warn] ESA KML download failed ({exc})')
            continue
        collected.extend(parse_esa_kml(kml_text, region, from_dt, to_dt))
    return collected


# --- Fallback trigger: only for satellites GISTDA did not cover ---
# Check which S1 satellites GISTDA already has
_s1_present = {f['sat'] for f in all_footprints
               if f['sat'] in ('sentinel-1c', 'sentinel-1d')}
_s1_missing = {'sentinel-1c', 'sentinel-1d'} - _s1_present

if _s1_missing:
    print(f'GISTDA missing: {_s1_missing} — fetching from ESA ...')
    _esa_all = fetch_esa_sentinel1(plan_from, plan_to, SAT_BBOX)
    # Only keep footprints for satellites that GISTDA did not have
    _esa_fps = [f for f in _esa_all if f['sat'] in _s1_missing]
    if _esa_fps:
        all_footprints.extend(_esa_fps)
        all_footprints.sort(key=lambda f: f['t_ict'])
        _added = {f['sat'] for f in _esa_fps}
        print(f'  ESA added {len(_esa_fps)} footprint(s) for: {_added}')
    else:
        print(f'  ESA had no footprints for: {_s1_missing}')
else:
    print('GISTDA has both Sentinel-1C and Sentinel-1D. ESA fallback not needed.')


# @title
# Built once and reused by every panel. Creating a NaturalEarthFeature per
# panel re-reads the shapefile ~120 times per run and leaks memory.
LAND_FEATURE = cfeature.NaturalEarthFeature(
    'physical', 'land', '50m',
    facecolor=LAND_COLOR, edgecolor='none')
COASTLINE_FEATURE = cfeature.COASTLINE.with_scale('50m')


def draw_panel(ax, data, rain_levels, rain_colors, title_line1, title_line2, stat_label,
               footprints=None):
    cmap = mcolors.ListedColormap(rain_colors)
    norm = mcolors.BoundaryNorm(rain_levels, cmap.N)

    ax.set_facecolor(OCEAN_COLOR)
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
                  crs=ccrs.PlateCarree())

    ax.add_feature(LAND_FEATURE, zorder=1)

    smooth = gaussian_filter(data, sigma=SMOOTH_SIGMA)
    masked = np.where(smooth < rain_levels[0], np.nan, smooth)

    cf = ax.contourf(
        lon2d, lat2d, masked,
        levels=rain_levels, colors=rain_colors,
        transform=ccrs.PlateCarree(),
        zorder=2, alpha=0.83, extend='max',
    )
    ax.contour(
        lon2d, lat2d, masked,
        levels=rain_levels, colors='white',
        linewidths=0.35,
        transform=ccrs.PlateCarree(),
        zorder=3, alpha=0.5,
    )

    if province_geoms:
        ax.add_geometries(province_geoms, crs=ccrs.PlateCarree(),
                          facecolor='none', edgecolor='#555555',
                          linewidth=0.3, zorder=4)
    if neighbor_geoms:
        ax.add_geometries(neighbor_geoms, crs=ccrs.PlateCarree(),
                          facecolor='none', edgecolor='#777777',
                          linewidth=0.7, zorder=5)
    if thailand_geom:
        ax.add_geometries([thailand_geom], crs=ccrs.PlateCarree(),
                          facecolor='none', edgecolor='#111111',
                          linewidth=1.8, zorder=6)

    ax.add_feature(COASTLINE_FEATURE,
                   edgecolor='#555555', linewidth=0.5, zorder=7)

    gl = ax.gridlines(draw_labels=True, linewidth=0.3,
                      linestyle=':', color='#bbbbbb', zorder=8)
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlocator = mticker.MultipleLocator(4)
    gl.ylocator = mticker.MultipleLocator(4)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 6.5, 'color': '#333333'}
    gl.ylabel_style = {'size': 6.5, 'color': '#333333'}
    ax._panel_gridliner = gl

    ax.set_title(f'{title_line1}\n{title_line2}',
                 fontsize=8, color='#111111', pad=3, loc='center')

    if thailand_mask is not None:
        th_data = np.where(thailand_mask, data, np.nan)
        th_max  = np.nanmax(th_data)
        if np.isfinite(th_max) and th_max >= rain_levels[0]:
            # Top-5 locations in Thailand
            flat = th_data.flatten()
            top5_flat = np.argsort(flat)[::-1]
            seen = []
            count = 0
            for fi in top5_flat:
                if not np.isfinite(flat[fi]):
                    continue
                r_idx, c_idx = np.unravel_index(fi, th_data.shape)
                pt_lon, pt_lat = lons[c_idx], lats[r_idx]
                too_close = any(
                    abs(pt_lon - s[0]) < 1.0 and abs(pt_lat - s[1]) < 1.0
                    for s in seen
                )
                if too_close:
                    continue
                seen.append((pt_lon, pt_lat))
                color = '#cc0000' if count == 0 else '#444444'
                ax.plot(pt_lon, pt_lat,
                        marker='v', markersize=7 if count == 0 else 5,
                        color=color,
                        markeredgecolor='white', markeredgewidth=0.6,
                        transform=ccrs.PlateCarree(), zorder=25)
                txt = ax.text(pt_lon, pt_lat - 0.6,
                        f'{flat[fi]:.1f} mm',
                        transform=ccrs.PlateCarree(),
                        ha='center', va='top',
                        fontsize=7.5, color='white', fontweight='bold',
                        zorder=26)
                txt.set_path_effects([
                    pe.withStroke(linewidth=2.0,
                                  foreground='#111111')])
                count += 1
                if count >= 5:
                    break
            # Top-right label: Thailand max only
            ax.text(0.985, 0.985,
                    f'Thailand max: {th_max:.1f} mm',
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=7, color='#cc0000', fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.82,
                              edgecolor='#dddddd', pad=2),
                    zorder=20)
        else:
            ax.text(0.985, 0.985, 'Thailand: < threshold',
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=7, color='#888888',
                    bbox=dict(facecolor='white', alpha=0.75,
                              edgecolor='#dddddd', pad=2),
                    zorder=20)

    if footprints:
        drawn_sats = set()
        for fp in footprints:
            style = SAT_STYLE.get(fp['sat'], {'color': '#000000'})
            geom = fp['geom']
            geoms = geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]
            for g in geoms:
                xs, ys = g.exterior.xy
                ax.plot(list(xs), list(ys),
                        transform=ccrs.PlateCarree(),
                        color=style['color'],
                        linewidth=style.get('lw', 2.0),
                        alpha=1.0, zorder=25)
                cen = g.centroid
                short = SAT_SHORT.get(fp['sat'], fp['sat'])
                lbl = ax.text(cen.x, cen.y, short,
                              transform=ccrs.PlateCarree(),
                              ha='center', va='center',
                              fontsize=7.5, fontweight='bold',
                              color='white', zorder=26)
                lbl.set_path_effects([
                    pe.withStroke(linewidth=2.5,
                                  foreground='black')])
            drawn_sats.add(fp['sat'])
        if drawn_sats:
            lines = [f"{len(footprints)} SAR pass(es)"]
            for s in SAT_NAMES:
                if s in drawn_sats and s in SAT_STYLE:
                    lbl = SAT_STYLE[s]['label']
                    cnt = sum(1 for fp in footprints if fp['sat'] == s)
                    lines.append(f'{lbl}: {cnt}')
            ax.text(0.015, 0.985, '\n'.join(lines),
                    transform=ax.transAxes, ha='left', va='top',
                    fontsize=6.5, color='#111111',
                    bbox=dict(facecolor='white', alpha=0.82,
                              edgecolor='#dddddd', pad=2),
                    zorder=21)

    cbar = plt.colorbar(cf, ax=ax, orientation='horizontal',
                        pad=0.04, fraction=0.046, extend='max')
    cbar.set_ticks(rain_levels)
    cbar.set_ticklabels([str(v) for v in rain_levels])
    cbar.set_label('Precipitation (mm)', fontsize=7, color='#111111')
    cbar.ax.tick_params(labelsize=6.5, color='#333333')

    return cf


def _disable_gridline_labels(ax):
    gl = getattr(ax, '_panel_gridliner', None)
    if gl is None:
        return False
    gl.top_labels = False
    gl.bottom_labels = False
    gl.left_labels = False
    gl.right_labels = False
    gl.geo_labels = False
    gl.inline_labels = False
    gl.x_inline = False
    gl.y_inline = False
    gl._drawn = False
    for label in getattr(gl, '_all_labels', []):
        label.set_visible(False)
    return True


def save_map(fig, ax, filename, dpi=150):
    """Save one map, tolerating the Cartopy gridliner boundary failure.

    Cartopy builds a shapely Polygon from the geo spine to place gridline
    labels. On some matplotlib/shapely combinations that polygon comes out
    invalid and raises GEOSException during draw, which would otherwise kill
    the whole run. Retry once with the labels turned off.
    """
    try:
        fig.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white')
        return True
    except Exception as exc:
        print(f'[warn] {filename}: {type(exc).__name__}: {exc}', flush=True)

    if _disable_gridline_labels(ax):
        try:
            fig.savefig(filename, dpi=dpi,
                        bbox_inches='tight', facecolor='white')
            print(f'[warn] {filename}: saved without gridline labels',
                  flush=True)
            return True
        except Exception as exc:
            print(f'[warn] {filename}: retry failed: {exc}', flush=True)

    try:
        fig.savefig(filename, dpi=dpi, facecolor='white')
        print(f'[warn] {filename}: saved without tight bbox', flush=True)
        return True
    except Exception as exc:
        print(f'[error] {filename}: skipped: {exc}', flush=True)
        return False


SAT_LEGEND_HANDLES = [
    Line2D([0], [0], color=SAT_STYLE[s]['color'], linewidth=1.5,
           label=SAT_STYLE[s]['label'] + ' footprint')
    for s in SAT_NAMES if s in SAT_STYLE
]

LEGEND_HANDLES = [
    Line2D([0],[0], color='#111111', linewidth=1.8, label='Thailand border'),
    Line2D([0],[0], color='#777777', linewidth=0.7, label='Country borders'),
    Line2D([0],[0], color='#555555', linewidth=0.3, label='Thailand provinces'),
    Line2D([0],[0], marker='v', color='w', markerfacecolor='#cc0000',
           markeredgecolor='white', markersize=8,
           label='Top-5 max rainfall (Thailand)'),
]

def page_suptitle(fig, run_ict, run_utc, subtitle):
    fig.suptitle(
        f'ECMWF IFS — {subtitle}\n'
        f'Model run: {run_ict:%d %b %Y  %H:%M} Local Time '
        f'(= {run_utc:%Y-%m-%d %H:%M} UTC)',
        fontsize=10, color='#111111', y=0.999, va='top',
    )
    fig.text(0.01, 0.002,
             'Data: ECMWF Open Data (CC BY 4.0)  |  '
             'Map: Natural Earth  |  Python / Cartopy',
             fontsize=5.5, color='#999999', va='bottom')

print('Helper functions ready.')



# ---------------- Dashboard export helper ----------------
# Preserve the original Shared Drive structure:
# New-Disaster-Water/Colab_ECMWF_Export/PNG/YYYY-MM-DD_HHMM_ICT

DRIVE_PNG_PARENT_ID = _drive_secret("GOOGLE_DRIVE_FOLDER_ID")
if not DRIVE_PNG_PARENT_ID:
    raise RuntimeError(
        "GOOGLE_DRIVE_FOLDER_ID is missing. It must point to the existing PNG folder."
    )

RUN_FOLDER_NAME = run_ict.strftime("%Y-%m-%d_%H%M_ICT")
RUN_FOLDER_ID = ensure_run_folder(DRIVE_PNG_PARENT_ID, RUN_FOLDER_NAME)
print(f"Shared Drive run folder ready: {RUN_FOLDER_NAME}", flush=True)


def _upload(path, remove_after=False):
    p = Path(path)
    upload_file(p, RUN_FOLDER_ID)
    print(f"[drive] Uploaded: {p.name}", flush=True)
    if remove_after:
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def _export(panels, summary_src, run_ict):
    _upload(summary_src, remove_after=True)


print("Export helper ready (Shared Drive streaming mode).", flush=True)


# The notebook assembled its summary page by drawing every panel a second
# time into one giant figure of GeoAxes. That is what raised the Cartopy
# gridliner GEOSException on Streamlit, and it doubled the rendering work.
#
# Here the page is composed instead: the already-rendered panel PNGs are
# tiled with PIL one at a time, and only the title bar and the legend bar
# are drawn with matplotlib. Output stays at the notebook's full 150 dpi,
# but peak memory is one tile plus the canvas rather than a 240-inch Agg
# figure holding every decoded panel at once.
SUMMARY_DPI = 150
SUMMARY_MARGIN = 18


def _render_strip(width_in, height_in, draw_fn, dpi=SUMMARY_DPI):
    """Render a full-width header/footer band to an RGB array."""
    from PIL import Image

    fig = plt.figure(figsize=(width_in, height_in), facecolor='white')
    draw_fn(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, facecolor='white')
    plt.close(fig)
    buf.seek(0)
    with Image.open(buf) as im:
        return im.convert('RGB').copy()


def _make_summary_page(image_files, output_name, subtitle,
                       ncols=2, with_satellite_legend=False):
    from PIL import Image

    files = [Path(f) for f in image_files if Path(f).exists()]
    if not files:
        print(f"[warn] {output_name}: no panels to assemble", flush=True)
        return

    sizes = []
    for f in files:
        with Image.open(f) as im:
            sizes.append(im.size)
    tile_w = max(w for w, _ in sizes)
    tile_h = max(h for _, h in sizes)
    nrows = (len(files) + ncols - 1) // ncols

    grid_w = ncols * tile_w + (ncols + 1) * SUMMARY_MARGIN
    grid_h = nrows * tile_h + (nrows + 1) * SUMMARY_MARGIN
    width_in = grid_w / SUMMARY_DPI

    def _draw_header(fig):
        # Only the title belongs in the top band. page_suptitle() also draws
        # the credit line, which the notebook places at the foot of the page,
        # so the footer band renders that instead.
        fig.suptitle(
            f'ECMWF IFS - {subtitle}\n'
            f'Model run: {run_ict:%d %b %Y  %H:%M} Local Time '
            f'(= {run_utc:%Y-%m-%d %H:%M} UTC)',
            fontsize=10, color='#111111', y=0.88, va='top')

    def _draw_footer(fig):
        if with_satellite_legend:
            sats_present = {fp['sat'] for fp in all_footprints}
            fp_handles = [
                Line2D([0], [0], color=SAT_STYLE[s]['color'], linewidth=1.5,
                       label=SAT_STYLE[s]['label'] + ' footprint')
                for s in SAT_NAMES if s in sats_present and s in SAT_STYLE
            ]
            if fp_handles:
                sat_leg = fig.legend(
                    handles=fp_handles,
                    loc='upper center',
                    bbox_to_anchor=(0.5, 0.95),
                    ncol=len(fp_handles), fontsize=7.5,
                    framealpha=0.9, edgecolor='#cccccc',
                    title='Satellite acquisition footprints',
                    title_fontsize=7.5,
                    bbox_transform=fig.transFigure)
                fig.add_artist(sat_leg)
        fig.legend(handles=LEGEND_HANDLES,
                   loc='lower center',
                   bbox_to_anchor=(0.5, 0.06),
                   ncol=4,
                   fontsize=7.5, framealpha=0.9,
                   edgecolor='#cccccc',
                   bbox_transform=fig.transFigure)
        fig.text(0.01, 0.02,
                 'Data: ECMWF Open Data (CC BY 4.0)  |  '
                 'Map: Natural Earth  |  Python / Cartopy',
                 fontsize=5.5, color='#999999', va='bottom')

    header = _render_strip(width_in, 0.85, _draw_header)
    footer = _render_strip(width_in, 1.35 if with_satellite_legend else 0.95,
                           _draw_footer)

    canvas = Image.new(
        'RGB',
        (grid_w, header.height + grid_h + footer.height),
        'white')
    canvas.paste(header, (0, 0))

    y_grid = header.height
    for idx, f in enumerate(files):
        r, c = divmod(idx, ncols)
        with Image.open(f) as im:
            im = im.convert('RGB')
            x = SUMMARY_MARGIN + c * (tile_w + SUMMARY_MARGIN) \
                + (tile_w - im.width) // 2
            y = y_grid + SUMMARY_MARGIN + r * (tile_h + SUMMARY_MARGIN) \
                + (tile_h - im.height) // 2
            canvas.paste(im, (x, y))

    canvas.paste(footer, (0, header.height + grid_h))
    canvas.save(output_name, dpi=(SUMMARY_DPI, SUMMARY_DPI))
    print(f"Saved: {output_name} "
          f"({canvas.width}x{canvas.height} px @ {SUMMARY_DPI} dpi)",
          flush=True)
    canvas.close()


status(5, 58, "Processing 24-hour accumulated rainfall...")
# @title
panels_24h = []
for i in range(len(hours_24h)):
    h1 = hours_24h[i]
    h0 = hours_24h[i - 1] if i > 0 else 0
    curr = da_24h.isel({sdim_24h: i}).values
    prev = (da_24h.isel({sdim_24h: i - 1}).values if i > 0
            else np.zeros_like(curr))
    incr = np.maximum(curr - prev, 0.0)
    t0_ict = run_utc + timedelta(hours=h0) + TZ_OFFSET
    t1_ict = run_utc + timedelta(hours=h1) + TZ_OFFSET
    day_n  = i + 1
    _fn = (f'ecmwf-24hr-day{day_n}'
           f'-{t0_ict:%Y%m%dt%H%M}'
           f'-{t1_ict:%Y%m%dt%H%M}'
           f'-Local-time.png')
    panels_24h.append({
        'data': incr, 'h0': h0, 'h1': h1,
        't0': t0_ict, 't1': t1_ict, 'day': day_n,
        'fps': footprints_in_window(t0_ict, t1_ict),
        'fn': _fn,
    })



lat_span = LAT_MAX - LAT_MIN
lon_span = LON_MAX - LON_MIN
cell_w   = 6.2
cell_h   = cell_w * (lat_span / lon_span) + 1.8

for idx, p in enumerate(panels_24h):
    status(
        6,
        65 + int(((idx + 1) / max(len(panels_24h), 1)) * 20),
        f"Generating Day {p['day']} map with satellite footprints and ground tracks..."
    )
    tl1 = (f"Day {p['day']}  —  "
           f"{p['t0']:%d %b %H:%M} to {p['t1']:%d %b %H:%M}  "
           f"Local Time (ICT)")
    tl2 = f"+{p['h0']}h to +{p['h1']}h  |  24-hour accumulated"

    _fig_p, _ax_p = plt.subplots(
        1, 1, figsize=(cell_w, cell_h),
        subplot_kw={'projection': ccrs.PlateCarree()},
        facecolor='white')

    draw_panel(
        _ax_p, p['data'],
        RAIN_LEVELS_24H, RAIN_COLORS_24H,
        tl1, tl2, '',
        footprints=p['fps'])

    save_map(_fig_p, _ax_p, p['fn'], dpi=150)
    plt.close(_fig_p)
    if Path(p['fn']).exists():
        _upload(p['fn'], remove_after=False)

_make_summary_page(
    [p['fn'] for p in panels_24h],
    'ecmwf_24h_accumulated.png',
    '24-Hour Accumulated Precipitation Forecast (Day 1 - Day 10)',
    ncols=2,
    with_satellite_legend=True,
)
_export(panels_24h, 'ecmwf_24h_accumulated.png', run_ict)
# The 24-hour day maps are deliberately kept on disk: forecast_runner
# .get_latest_day_images() serves them when Google Drive is unreachable.

# Everything the dashboard needs is now on Drive and on disk. The 12h, 6h
# and 3h products are extra Drive deliverables, so a failure there must not
# invalidate a run whose Day 1-10 maps already succeeded.
status(6, 85, "Day 1-10 maps ready. Generating additional products...")
print("[part-a] Day 1-10 24-hour products complete and uploaded.", flush=True)


def _optional_part(label, fn):
    try:
        fn()
    except Exception as exc:
        print(f"[warn] {label} skipped: {type(exc).__name__}: {exc}",
              flush=True)



def _part_12_hour():
    status(6, 88, "Generating additional 12-hour products...")
    da_12h, sdim_12h, hours_12h, _ = fetch_product(
        '12h', STEPS_12H, GRIB_FILE_12H, STEPS_12H)
    # @title
    panels_12h = []
    for i in range(1, len(hours_12h)):
        h0   = hours_12h[i - 1]
        h1   = hours_12h[i]
        curr = da_12h.isel({sdim_12h: i}).values
        prev = da_12h.isel({sdim_12h: i - 1}).values
        incr = np.maximum(curr - prev, 0.0)
        t0_ict = run_utc + timedelta(hours=h0) + TZ_OFFSET
        t1_ict = run_utc + timedelta(hours=h1) + TZ_OFFSET
        _pn = (i - 1) // 2 + 1
        _fn = (f'ecmwf-12hr-day{_pn}'
               f'-{t0_ict:%Y%m%dt%H%M}'
               f'-{t1_ict:%Y%m%dt%H%M}'
               f'-Local-time.png')
        panels_12h.append({
            'data': incr, 'h0': h0, 'h1': h1,
            't0': t0_ict, 't1': t1_ict, 'fn': _fn,
        })



    lat_span = LAT_MAX - LAT_MIN
    lon_span = LON_MAX - LON_MIN
    cell_w   = 6.2
    cell_h   = cell_w * (lat_span / lon_span) + 1.8

    for idx, p in enumerate(panels_12h):
        tl1 = (f"{p['t0']:%d %b %H:%M} to {p['t1']:%d %b %H:%M}  "
               f"Local Time (ICT)")
        tl2 = f"+{p['h0']}h to +{p['h1']}h  |  12-hour accumulated"

        _fig_p, _ax_p = plt.subplots(
            1, 1, figsize=(cell_w, cell_h),
            subplot_kw={'projection': ccrs.PlateCarree()},
            facecolor='white')

        draw_panel(
            _ax_p, p['data'],
            RAIN_LEVELS_12H, RAIN_COLORS_12H,
            tl1, tl2, '')

        save_map(_fig_p, _ax_p, p['fn'], dpi=150)
        plt.close(_fig_p)
        if Path(p['fn']).exists():
            _upload(p['fn'], remove_after=False)

    _make_summary_page(
        [p['fn'] for p in panels_12h],
        'ecmwf_12h_incremental.png',
        '12-Hour Incremental Precipitation Forecast (10-day)',
        ncols=2,
        with_satellite_legend=False,
    )
    _export(panels_12h, 'ecmwf_12h_incremental.png', run_ict)
    for _p in panels_12h:
        try:
            Path(_p['fn']).unlink()
        except FileNotFoundError:
            pass


_optional_part('12-hour products', _part_12_hour)


def _part_6_hour():
    status(6, 91, "Generating additional 6-hour products...")
    da_6h, sdim_6h, hours_6h, _ = fetch_product(
        '6h', STEPS_6H, GRIB_FILE_6H, STEPS_6H)
    # @title
    # PART C — 6-Hour Incremental Precipitation
    panels_6h = []
    for i in range(1, len(hours_6h)):
        h0   = hours_6h[i - 1]
        h1   = hours_6h[i]
        curr = da_6h.isel({sdim_6h: i}).values
        prev = da_6h.isel({sdim_6h: i - 1}).values
        incr = np.maximum(curr - prev, 0.0)
        t0_ict = run_utc + timedelta(hours=h0) + TZ_OFFSET
        t1_ict = run_utc + timedelta(hours=h1) + TZ_OFFSET
        _pn = (i - 1) // 4 + 1
        _fn = (f'ecmwf-06hr-day{_pn}'
               f'-{t0_ict:%Y%m%dt%H%M}'
               f'-{t1_ict:%Y%m%dt%H%M}'
               f'-Local-time.png')
        panels_6h.append({
            'data': incr, 'h0': h0, 'h1': h1,
            't0': t0_ict, 't1': t1_ict, 'fn': _fn,
        })



    lat_span = LAT_MAX - LAT_MIN
    lon_span = LON_MAX - LON_MIN
    cell_w   = 6.2
    cell_h   = cell_w * (lat_span / lon_span) + 1.8

    for idx, p in enumerate(panels_6h):
        tl1 = (f"{p['t0']:%d %b %H:%M} to {p['t1']:%d %b %H:%M}  "
               f"Local Time (ICT)")
        tl2 = f"+{p['h0']}h to +{p['h1']}h  |  6-hour accumulated"

        _fig_p, _ax_p = plt.subplots(
            1, 1, figsize=(cell_w, cell_h),
            subplot_kw={'projection': ccrs.PlateCarree()},
            facecolor='white')

        draw_panel(
            _ax_p, p['data'],
            RAIN_LEVELS_6H, RAIN_COLORS_6H,
            tl1, tl2, '')

        save_map(_fig_p, _ax_p, p['fn'], dpi=150)
        plt.close(_fig_p)
        if Path(p['fn']).exists():
            _upload(p['fn'], remove_after=False)

    _make_summary_page(
        [p['fn'] for p in panels_6h],
        'ecmwf_6h_incremental.png',
        '6-Hour Incremental Precipitation Forecast (10-day)',
        ncols=2,
        with_satellite_legend=False,
    )
    _export(panels_6h, 'ecmwf_6h_incremental.png', run_ict)
    for _p in panels_6h:
        try:
            Path(_p['fn']).unlink()
        except FileNotFoundError:
            pass


_optional_part('6-hour products', _part_6_hour)


def _part_3_hour():
    status(6, 94, "Generating additional 3-hour products...")
    da_3h, sdim_3h, hours_3h, _ = fetch_product(
        '3h', STEPS_3H, GRIB_FILE_3H, STEPS_3H)
    # @title
    # PART D — 3-Hour Incremental Precipitation (first 6 days only)
    panels_3h = []
    for i in range(1, len(hours_3h)):
        h0   = hours_3h[i - 1]
        h1   = hours_3h[i]
        curr = da_3h.isel({sdim_3h: i}).values
        prev = da_3h.isel({sdim_3h: i - 1}).values
        incr = np.maximum(curr - prev, 0.0)
        t0_ict = run_utc + timedelta(hours=h0) + TZ_OFFSET
        t1_ict = run_utc + timedelta(hours=h1) + TZ_OFFSET
        _pn = (i - 1) // 8 + 1
        _fn = (f'ecmwf-03hr-day{_pn}'
               f'-{t0_ict:%Y%m%dt%H%M}'
               f'-{t1_ict:%Y%m%dt%H%M}'
               f'-Local-time.png')
        panels_3h.append({
            'data': incr, 'h0': h0, 'h1': h1,
            't0': t0_ict, 't1': t1_ict, 'fn': _fn,
        })



    lat_span = LAT_MAX - LAT_MIN
    lon_span = LON_MAX - LON_MIN
    cell_w   = 6.2
    cell_h   = cell_w * (lat_span / lon_span) + 1.8

    for idx, p in enumerate(panels_3h):
        tl1 = (f"{p['t0']:%d %b %H:%M} to {p['t1']:%d %b %H:%M}  "
               f"Local Time (ICT)")
        tl2 = f"+{p['h0']}h to +{p['h1']}h  |  3-hour accumulated"

        _fig_p, _ax_p = plt.subplots(
            1, 1, figsize=(cell_w, cell_h),
            subplot_kw={'projection': ccrs.PlateCarree()},
            facecolor='white')

        draw_panel(
            _ax_p, p['data'],
            RAIN_LEVELS_3H, RAIN_COLORS_3H,
            tl1, tl2, '')

        save_map(_fig_p, _ax_p, p['fn'], dpi=150)
        plt.close(_fig_p)
        if Path(p['fn']).exists():
            _upload(p['fn'], remove_after=False)

    _make_summary_page(
        [p['fn'] for p in panels_3h],
        'ecmwf_3h_incremental.png',
        '3-Hour Incremental Precipitation Forecast (Day 1 - Day 6)',
        ncols=2,
        with_satellite_legend=False,
    )
    _export(panels_3h, 'ecmwf_3h_incremental.png', run_ict)
    for _p in panels_3h:
        try:
            Path(_p['fn']).unlink()
        except FileNotFoundError:
            pass


_optional_part('3-hour products', _part_3_hour)


status(7, 100, "Processing completed.")
