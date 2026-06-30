"""
Extreme-event analytics derived from forecasts / observations + climatology.

These turn the climate twin into an early-warning tool for the events the
problem statement calls out: heatwaves, dry spells / drought, heavy-rain
events and monsoon activity. All thresholds follow IMD-style operational
definitions where possible.
"""
from __future__ import annotations

import numpy as np

# --- Heatwave (IMD-style, departure-based) ---
# Heatwave when Tmax >= 37C over plains/hills AND departure from normal >= 4.5C
# (severe when departure >= 6.5C); or Tmax >= 45C in absolute terms.
HW_DEP_MOD = 4.5
HW_DEP_SEV = 6.5
HW_TMAX_MIN = 37.0
HW_TMAX_ABS = 45.0


def heatwave_severity(tmax_field, tmax_clim):
    """0 = none, 1 = heatwave, 2 = severe. Element-wise on a 2-D field."""
    dep = tmax_field - tmax_clim
    base = (tmax_field >= HW_TMAX_MIN) & (dep >= HW_DEP_MOD)
    sev = (dep >= HW_DEP_SEV) | (tmax_field >= HW_TMAX_ABS)
    out = np.zeros_like(tmax_field, dtype="float32")
    out[base] = 1.0
    out[base & sev] = 2.0
    return out


# --- Rainfall categories (IMD daily classes, mm) ---
def rain_category(rain_field):
    """0 dry/light, 1 moderate(>=15), 2 heavy(>=64.5), 3 very heavy(>=115.5),
    4 extremely heavy(>=204.5)."""
    r = np.asarray(rain_field)
    out = np.zeros_like(r, dtype="float32")
    out[r >= 15.0] = 1
    out[r >= 64.5] = 2
    out[r >= 115.5] = 3
    out[r >= 204.5] = 4
    return out


def dry_spell_index(rain_seq, dry_thresh=2.5):
    """Given (T,H,W) daily rain, return the length of the trailing dry run per
    cell (consecutive days with rain < dry_thresh up to the last day)."""
    wet = rain_seq >= dry_thresh
    T = rain_seq.shape[0]
    run = np.zeros(rain_seq.shape[1:], dtype="float32")
    for k in range(T - 1, -1, -1):
        still_dry = ~wet[k]
        run = np.where(still_dry & (run == (T - 1 - k)), run + 1, run)
    return run


def rainfall_deficit_pct(rain_accum, clim_accum):
    """Percent departure of accumulated rain from climatology (drought signal)."""
    denom = np.where(clim_accum > 1e-3, clim_accum, np.nan)
    return 100.0 * (rain_accum - clim_accum) / denom


def heat_severity_band(v):
    return {0: ("None", "#2DD4BF"), 1: ("Heatwave", "#FF7B00"),
            2: ("Severe heatwave", "#FF2D55")}.get(int(round(v)), ("None", "#2DD4BF"))


def summarize_forecast_hazards(frames, clim_tmax_seq, landmask):
    """Per-lead hazard exposure across a forecast horizon (land-masked).

    Parameters
    ----------
    frames : dict
        ``{'tmax': (HORIZON,H,W), 'rain': (HORIZON,H,W), ...}`` forecast fields.
    clim_tmax_seq : np.ndarray
        ``(HORIZON,H,W)`` climatological Tmax for each lead day (heatwave departure).
    landmask : np.ndarray
        ``(H,W)`` boolean land mask.

    Returns
    -------
    dict of lists (length = HORIZON):
        ``leads``, ``heatwave_area_pct``, ``severe_heatwave_area_pct``,
        ``heavy_rain_area_pct``, ``very_heavy_rain_area_pct``,
        ``dry_spell_mean_days``.
    """
    tmax = np.asarray(frames["tmax"])
    rain = np.asarray(frames["rain"])
    clim = np.asarray(clim_tmax_seq)
    horizon = tmax.shape[0]
    lm = landmask.astype(bool)

    out = {"leads": list(range(1, horizon + 1)),
           "heatwave_area_pct": [], "severe_heatwave_area_pct": [],
           "heavy_rain_area_pct": [], "very_heavy_rain_area_pct": [],
           "dry_spell_mean_days": []}

    for k in range(horizon):
        sev = heatwave_severity(tmax[k], clim[k])[lm]
        out["heatwave_area_pct"].append(float(np.mean(sev >= 1) * 100))
        out["severe_heatwave_area_pct"].append(float(np.mean(sev >= 2) * 100))

        cat = rain_category(rain[k])[lm]
        out["heavy_rain_area_pct"].append(float(np.mean(cat >= 2) * 100))
        out["very_heavy_rain_area_pct"].append(float(np.mean(cat >= 3) * 100))

        dsi = dry_spell_index(rain[:k + 1])[lm]
        out["dry_spell_mean_days"].append(float(np.mean(dsi)))

    return out
