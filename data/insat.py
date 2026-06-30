"""
INSAT / MOSDAC satellite ingestion (real data, bring-your-own-file).

MOSDAC (https://www.mosdac.gov.in) requires a free login to download INSAT-3D/3DR
Level-2B products, so files cannot be fetched unattended. This module is a REAL
loader for the actual HDF5 product format: drop a downloaded `.h5` into
`data/insat/` and the dashboard automatically ingests and regrids it onto the
national IMD grid. Nothing is synthesised - if no file is present the satellite
layer is simply unavailable (clearly indicated in the UI).

Supported products (MOSDAC product id -> our key):
  3RIMG_*_L2B_LST  -> 'lst'   Land Surface Temperature (degC)
  3RIMG_*_L2B_SST  -> 'sst'   Sea Surface Temperature (degC)
  3RIMG_*_L2B_IMC  -> 'rain'  INSAT Multi-spectral rainfall (mm/hr)

Download steps are printed by `instructions()`.
"""
from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402

INSAT_DIR = os.path.join(C.DATA_DIR, "insat")
os.makedirs(INSAT_DIR, exist_ok=True)

# Candidate dataset names inside the HDF5 (products vary slightly by version).
_VAR_CANDIDATES = {
    "lst": ["LST", "Land_Surface_Temperature", "lst"],
    "sst": ["SST", "Sea_Surface_Temperature", "sst"],
    "rain": ["IMR", "HEM", "RAIN", "Rainfall", "rain"],
}
_LAT_CANDIDATES = ["Latitude", "latitude", "lat", "Y"]
_LON_CANDIDATES = ["Longitude", "longitude", "lon", "X"]


def available_files():
    return sorted(glob.glob(os.path.join(INSAT_DIR, "*.h5")) +
                  glob.glob(os.path.join(INSAT_DIR, "*.hdf")) +
                  glob.glob(os.path.join(INSAT_DIR, "*.nc")))


# INSAT-3D/3DR product names embed the acquisition stamp as e.g. 29JUN2026_0815
_DATE_RE = re.compile(r"(\d{2})([A-Z]{3})(\d{4})(?:[_-](\d{4}))?", re.IGNORECASE)
_MONTHS = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
           "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}


def product_datetime(filename):
    """Parse the acquisition timestamp from an INSAT product filename.

    Handles the standard MOSDAC naming, e.g. ``3RIMG_29JUN2026_0815_L2B_LST``
    -> pandas.Timestamp('2026-06-29 08:15'). Returns None if no stamp is found.
    """
    import pandas as pd

    m = _DATE_RE.search(os.path.basename(filename))
    if not m:
        return None
    day, mon, year, hhmm = m.group(1), m.group(2).upper(), m.group(3), m.group(4)
    if mon not in _MONTHS:
        return None
    hh, mm = (int(hhmm[:2]), int(hhmm[2:])) if hhmm else (0, 0)
    try:
        return pd.Timestamp(int(year), _MONTHS[mon], int(day), hh, mm)
    except ValueError:
        return None


def product_doy(filename):
    """Day-of-year (1..366) for an INSAT product filename, or None."""
    ts = product_datetime(filename)
    return None if ts is None else min(int(ts.dayofyear), 366)


def has_data():
    return len(available_files()) > 0


def _find(h5, names):
    keys = {}

    def visit(name, obj):
        import h5py
        if isinstance(obj, h5py.Dataset):
            keys[name.split("/")[-1].lower()] = name
    h5.visititems(visit)
    for n in names:
        if n.lower() in keys:
            return keys[n.lower()]
    return None


def _decode(ds):
    """Apply scale_factor/add_offset and mask _FillValue."""
    arr = np.asarray(ds[()], dtype="float64")
    attrs = ds.attrs
    fill = attrs.get("_FillValue", attrs.get("fill_value", None))
    if fill is not None:
        arr = np.where(arr == np.asarray(fill).ravel()[0], np.nan, arr)
    sf = attrs.get("scale_factor", None)
    off = attrs.get("add_offset", None)
    if sf is not None:
        arr = arr * np.asarray(sf).ravel()[0]
    if off is not None:
        arr = arr + np.asarray(off).ravel()[0]
    return arr


def load_to_national_grid(path, key, grid):
    """Read product `key` from an INSAT HDF5 and regrid to the national grid.

    Handles the INSAT-3DR L2B full-disk format: per-pixel Latitude/Longitude
    (int16, scale 0.01, fill 32767) and a (1,H,W) geophysical field. The full
    disk is ~7.9M pixels, so we crop to the India bounding box BEFORE
    interpolating. Temperature products (LST/SST) are converted Kelvin -> degC.

    Returns a (H,W) field on the IMD 0.25 grid (NaN where no coverage).
    """
    import h5py
    from scipy.interpolate import griddata

    lat_t, lon_t = grid["lat"], grid["lon"]
    with h5py.File(path, "r") as f:
        vk = _find(f, _VAR_CANDIDATES[key])
        if vk is None:
            raise KeyError(f"variable for '{key}' not found in {os.path.basename(path)}")
        units = f[vk].attrs.get("units", b"")
        units = units.decode() if isinstance(units, bytes) else str(units)
        val = np.squeeze(_decode(f[vk]))                 # (H,W)
        lak = _find(f, _LAT_CANDIDATES)
        lok = _find(f, _LON_CANDIDATES)
        if not (lak and lok):
            raise KeyError("geolocation (lat/lon) not found in product")
        la = np.squeeze(_decode(f[lak]))
        lo = np.squeeze(_decode(f[lok]))

    # Kelvin -> Celsius for temperature products
    if key in ("lst", "sst") and np.nanmedian(val) > 150:
        val = val - 273.15

    # crop to India bbox (+1 deg margin) BEFORE interpolation for speed
    pad = 1.0
    m = (np.isfinite(val) & np.isfinite(la) & np.isfinite(lo)
         & (la >= lat_t.min() - pad) & (la <= lat_t.max() + pad)
         & (lo >= lon_t.min() - pad) & (lo <= lon_t.max() + pad))
    if m.sum() == 0:
        return np.full((len(lat_t), len(lon_t)), np.nan, dtype="float32")

    pts = np.column_stack([lo[m].ravel(), la[m].ravel()])
    lon2d, lat2d = np.meshgrid(lon_t, lat_t)
    out = griddata(pts, val[m].ravel(), (lon2d, lat2d), method="linear")
    return out.astype("float32")


def latest(key, grid):
    """Load the most recent available INSAT file for `key`, or None."""
    files = available_files()
    if not files:
        return None
    for path in reversed(files):
        try:
            return {"field": load_to_national_grid(path, key, grid),
                    "file": os.path.basename(path)}
        except Exception:
            continue
    return None


def instructions() -> str:
    return (
        "INSAT/MOSDAC satellite layer is inactive (no file found).\n"
        "To enable real satellite data:\n"
        "  1. Register (free) at https://www.mosdac.gov.in\n"
        "  2. Download an INSAT-3D/3DR L2B product, e.g. 3RIMG_*_L2B_LST (HDF5)\n"
        f"  3. Place the .h5 file in: {INSAT_DIR}\n"
        "The dashboard ingests and regrids it onto the national grid automatically."
    )
