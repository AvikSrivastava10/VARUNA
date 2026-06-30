# AI Digital Twin of India's Climate — Build Plan

**ISRO Bharatiya Antariksh Hackathon (BAH) 2026 · Problem Statement #5**
*An indigenous, AI-powered digital twin of India's climate built entirely on real national (IMD) data.*

This document is the engineering plan. It is intentionally specific so the build is reproducible end-to-end inside this single repository — every dataset is downloaded here, every model is trained here, no synthetic or placeholder data anywhere.

---

## 0. Guiding principles
- **Real data only.** All fields come from IMD gridded products via `imdlib`. No synthetic fallback.
- **Self-contained.** Download → preprocess → train → evaluate → serve, all in this repo.
- **Honest + verifiable.** Every headline number is produced by a script in `evaluation/` and reproducible.
- **High-end where it counts.** A modern anomaly-based, direct-multi-horizon CNN forecaster that structurally beats the old ConvLSTM's divergence, trained on the GPU (RTX 4050).

---

## 1. Data foundation (REAL IMD)
**Sources** (https://imdpune.gov.in/cmpg/Griddata/):
| Variable | Resolution | Native grid | Period |
|---|---|---|---|
| Rainfall | 0.25° | 135 × 129 (lon×lat), 66.5–100.0E / 6.5–38.5N | 1901–2024 |
| Max temp | 1.0° | 31 × 31, 67.5–97.5E / 7.5–37.5N | 1951–2024 |
| Min temp | 1.0° | 31 × 31, same as max | 1951–2024 |

**Pipeline** (`data/`):
- `download_imd.py` — `imdlib` ingest, fill-value masking, offline cache in `data/raw/`.
- `prepare.py` — regrid temperature → 0.25° rain grid (one common national grid), build land mask, smoothed day-of-year **climatology from TRAIN years only**, anomaly statistics. Saves `data/processed/{obs.nc, clim_rain.nc, clim_temp.nc, stats.json, landmask.npy, grid.npz, meta.json}`.
- `climatology.py` — circular-smoothed day-of-year climatology + anomaly conversions.

**Span & split (no leakage):** train 1981–2018 (38 yr) · val 2019–2020 · test 2021–2024.

---

## 2. The model — ClimateUNet (`models/architecture.py`, PyTorch/CUDA)
**Why this design wins:**
- **Anomaly targets** vs climatology → rollouts relax toward climate, never diverge (the old ConvLSTM hit ±6σ by day 21).
- **Direct multi-horizon** — one forward pass emits all 10 lead days; no autoregressive feedback loop.
- **Residual U-Net + squeeze-excite attention**, time-as-channels (SimVP insight). 7.4M params, fits the 6 GB GPU.
- **Physics constraints** — rainfall reconstructed non-negative; area/latitude-weighted, land-masked loss.

Input `(7×3, 129, 135)` → Output `(10×3, 129, 135)` scaled anomalies.

**Training** (`models/train.py`): AdamW + cosine LR, AMP mixed precision, gradient clipping, Huber loss, early stopping on val. Checkpoint → `models/checkpoints/climate_unet.pt`.

---

## 3. Evaluation (`evaluation/`) — weather-standard, real units
Per variable, per lead day, over test years, land-masked + area-weighted:
- **RMSE / MAE** (mm/day, °C)
- **ACC** — anomaly correlation coefficient
- **Skill vs persistence** and **vs persistence-of-anomaly** (the references that matter)

Benchmarked against three baselines (`models/baseline.py`): persistence, persistence-of-anomaly, climatology. Outputs `outputs/eval_metrics.json` + `outputs/skill_curves.png`.

---

## 4. Connected applications (the PS#5 "applications" requirement)
One twin → three connected apps, all driven by the AI climate state:
- **Climate state & AI forecast** — national + pilot, date-driven, AI vs baseline vs obs.
- **Urban heat** — NWS heat-index + LST/UHI proxy (`scenario/engine.py`).
- **Air quality** — CPCB PM2.5→AQI proxy responding to rain washout, greening, wind, urban form.
- **Digital-twin assimilation** — observation nudging + bias correction (`twin/assimilate.py`).

---

## 5. Dashboard (`app.py`) — better than the reference
Original dark "orbital" theme (`viz/theme.py`), interactive national maps with raster overlays on a dark basemap (`viz/maps.py`), AI-forecast tab with live skill metrics, what-if scenario simulator, and an honest "About/methods" view. Streamlit + folium/plotly, no API tokens.

---

## 6. Milestones
1. ✅ Real IMD download (1981–2024) + processed cache.
2. ⏳ Train ClimateUNet on GPU; lock checkpoint.
3. ⏳ Run evaluation; record headline skill numbers.
4. ⏳ Build dashboard (theme, maps, forecast, what-if, assimilation).
5. ⏳ README + deck + demo polish.

---

## 7. Honest framing (for judges)
- AI short-term **forecast** on real IMD analyses (not a full GCM).
- Heat/AQI/LST are **physics-informed proxies** (NWS heat index, CPCB AQI, surface-energy LST) — clearly labeled.
- Assimilation is **Newtonian nudging + bias correction** (not 4D-Var/EnKF).
- Scale-up path: fine-tune a foundation model (Prithvi-WxC / Pangu-Weather on IMDAA/BharatBench), add INSAT live feeds.
