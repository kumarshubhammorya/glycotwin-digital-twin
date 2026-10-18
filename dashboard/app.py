"""
GlycoTwin: doctor-facing dashboard for a synthetic Type 2 diabetes digital twin.
Happiest Health Digital Twin Challenge 2026 - research prototype.

Synthetic data only. Research prototype, not a medical device. Not for dosing decisions.

How this file stays honest:
- Every prediction goes through the helpers in models/train_fusion_model.py
  (compute_wearable_features -> prepare_fused_dataset -> encode_features), so the
  dashboard builds and encodes model inputs exactly as training did.
- The alert comes from the classifier plus the alert thresholds stored in the
  model bundle (apply_alert_thresholds). The +2 h glucose value comes from the
  regressor. If the model cannot be loaded or run, the page shows an error and
  no forecast; it never falls back to the dataset's future labels.
- Every performance number shown is read at runtime from models/metrics.json,
  which models/evaluate.py writes.

Run from anywhere:  streamlit run dashboard/app.py   (or: python run_dashboard.py)
"""

import html
import io
import json
import math
import sys
import warnings
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "glycotwin_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

st.set_page_config(
    page_title="GlycoTwin | T2D Digital Twin (research prototype)",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    import train_fusion_model as tfm
    from train_fusion_model import (
        CATEGORY_LEVELS,
        EHR_FEATURES,
        FEATURE_COLS,
        HYPER_MG_DL,
        HYPO_MG_DL,
        WEARABLE_FEATURES,
        apply_alert_thresholds,
        compute_wearable_features,
        encode_features,
        prepare_fused_dataset,
    )
    IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - shown in the UI
    IMPORT_ERROR = exc

# ------------------------------------------------------------------------------
# Clinical constants (ADA / international consensus CGM targets)
# ------------------------------------------------------------------------------
LEVEL2_HYPO_MG_DL = 54.0      # ADA Level 2 hypoglycaemia
TARGET_HIGH_MG_DL = 180.0     # upper bound of time-in-range 70-180 mg/dL
CV_TARGET_PCT = 36            # consensus glycaemic variability target
STEP_MIN = 15                 # wearable sampling interval
HORIZON_STEPS = 120 // STEP_MIN
HISTORY_STEPS = 8 * 60 // STEP_MIN
DEFAULT_PATIENT = "IND-PAT-196"
DISCLAIMER = "Synthetic data only. Research prototype, not a medical device."
NAV_ITEMS = ["Live Monitor", "What-If Sandbox", "Data Ingest", "Cohort & Risk Board", "Model Card"]

# Wearable stream columns a device export needs before the model can score it.
WEARABLE_INPUT_COLS = ["timestamp", "glucose_cgm", "heart_rate", "hrv", "steps_15m",
                       "sleep_stage", "meal_carbs_g", "insulin_units"]
# Plausibility limits used by the ingest validator (sensor-artefact screens, not clinical targets).
GLUCOSE_BOUNDS = (40.0, 400.0)
OTHER_BOUNDS = {"heart_rate": (25, 230), "hrv": (5, 300), "steps_15m": (0, 4000),
                "meal_carbs_g": (0, 300), "insulin_units": (0, 50)}
MAX_ROC_MG_DL_PER_MIN = 4.0

RED, RED_DARK, AMBER, GREEN, TEAL, NAVY, SLATE, MUTED = (
    "#EF4444", "#B91C1C", "#F59E0B", "#10B981", "#0D9488", "#0F172A", "#64748B", "#94A3B8")

# ------------------------------------------------------------------------------
# Design system CSS (navy sidebar, light canvas, white 12px cards, teal accent)
# ------------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], [class*="st-"]:not([data-testid="stIconMaterial"]) {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
/* Keep Streamlit's Material icon font (otherwise icon names render as text). */
[data-testid="stIconMaterial"], .material-symbols-rounded, span[class*="material-symbols"] {
    font-family: 'Material Symbols Rounded' !important;
}
.tabular-nums, .metric-value, .kpi-num, [data-testid="stMetricValue"] {
    font-variant-numeric: tabular-nums;
    font-feature-settings: "tnum";
}

/* Keep the header (it holds the sidebar re-open control); hide only menu and deploy chrome. */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton, [data-testid="stAppDeployButton"] {display: none;}
header[data-testid="stHeader"] {background: rgba(248,250,252,0.85);}
.block-container {padding-top: 2.6rem;}

/* Sidebar theme */
[data-testid="stSidebar"] {background-color: #0F172A !important; border-right: 1px solid #1E293B;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label {color: #94A3B8 !important;}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {color: #F8FAFC !important;}
[data-testid="stSidebar"] div[data-baseweb="select"] {background-color: #1E293B !important; border-color: #334155 !important;}
[data-testid="stSidebar"] div[data-baseweb="select"] * {color: #F8FAFC !important;}
[data-testid="stSidebar"] [data-testid="stRadio"] > div {gap: 6px;}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background-color: #1E293B; border-radius: 8px; padding: 8px 12px;
    border: 1px solid #334155; transition: all 0.2s ease; cursor: pointer; width: 100%;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {border-color: #0D9488; background-color: #24344D;}
[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] * ,
[data-testid="stSidebar"] [data-testid="stSidebarHeader"] button * {color: #CBD5E1 !important;}

/* Teal accent buttons */
.stButton > button, .stDownloadButton > button {
    background-color: #0D9488 !important; color: #FFFFFF !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important; padding: 6px 14px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background-color: #14B8A6 !important; box-shadow: 0 4px 12px rgba(13,148,136,0.25) !important;
}
.stButton > button:disabled {background-color: #CBD5E1 !important; color: #64748B !important;}

/* Teal instead of Streamlit red for sliders and radios (red means hypoglycaemia on this page) */
[data-testid="stSlider"] [role="group"] > div[data-orientation] > div[style*="translate"] {background-color: #0D9488 !important;}
[data-testid="stSlider"] [role="group"] > div[data-orientation] > div:first-child {filter: hue-rotate(172deg) saturate(0.85);}
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] {color: #0F766E !important;}
[data-testid="stSlider"] div[style*="translate"][data-focused="true"] {box-shadow: 0 0 0 0.2rem rgba(13,148,136,0.35) !important;}
label[data-testid="stRadioOption"][data-selected="true"] > div > div:first-child {background-color: #0D9488 !important; border-color: #0D9488 !important;}
[data-testid="stCheckbox"] label[data-selected="true"] > div:first-of-type {background-color: #0D9488 !important; border-color: #0D9488 !important;}
[data-testid="stSidebar"] {min-width: 320px !important; max-width: 320px !important;}

/* Cards */
.gt-card {
    background: #FFFFFF; border-radius: 12px; border: 1px solid #E2E8F0;
    box-shadow: 0 1px 3px rgba(15,23,42,0.06), 0 1px 2px rgba(15,23,42,0.04);
    padding: 14px 16px; margin-bottom: 12px;
}
.gt-label {font-size: 11px; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.04em;}
.gt-big {font-size: 24px; font-weight: 800; color: #0F172A; margin-top: 4px; font-variant-numeric: tabular-nums;}
.gt-sub {font-size: 12px; color: #64748B; margin-top: 2px;}
.gt-title {margin: 0; font-size: 20px; font-weight: 800; color: #0F172A; line-height: 1.3;}
.gt-lede {margin: 4px 0 12px 0; font-size: 13px; color: #64748B; line-height: 1.5;}

/* Disclaimer strip shown on every page */
.gt-disclaimer {
    background: #FFF7ED; border: 1px solid #FED7AA; color: #9A3412; border-radius: 8px;
    padding: 6px 12px; font-size: 12px; font-weight: 600; margin-bottom: 12px;
}

/* Chips */
.chip {display: inline-block; font-size: 10.5px; font-weight: 700; padding: 2px 8px; border-radius: 10px; white-space: nowrap;}
.chip-heldout {background: #CCFBF1; color: #0F766E;}
.chip-train {background: #E2E8F0; color: #334155;}
.chip-synth {background: #EDE9FE; color: #5B21B6;}

/* Patient 360 card in sidebar */
.patient-360-card {
    background: #1E293B; border-radius: 12px; border: 1px solid #334155;
    padding: 12px 14px; margin-top: 8px; font-size: 12px; color: #CBD5E1;
}
.patient-360-row {display: flex; justify-content: space-between; gap: 8px; padding: 4px 0; border-bottom: 1px solid #334155;}
.patient-360-label {color: #94A3B8;}
.patient-360-value {color: #F8FAFC; font-weight: 600; text-align: right;}
.patient-360-col {flex-direction: column; gap: 2px;}
.patient-360-col .patient-360-value {text-align: left; font-size: 11px;}
.p360-note {color: #94A3B8; font-size: 10.5px; line-height: 1.35; font-weight: 400;}


/* Range bar (40-360 mg/dL scale) */
.range-bar-track {
    width: 100%; height: 10px; border-radius: 5px; position: relative; margin-top: 10px; margin-bottom: 4px;
    background: linear-gradient(to right,
        #B91C1C 0%, #B91C1C 4.4%, #EF4444 4.4%, #EF4444 9.4%,
        #10B981 9.4%, #10B981 43.8%, #F59E0B 43.8%, #F59E0B 65.6%, #EF4444 65.6%, #EF4444 100%);
}
.range-bar-needle {
    position: absolute; top: -4px; width: 6px; height: 18px; background-color: #0F172A;
    border: 2px solid #FFFFFF; border-radius: 3px; box-shadow: 0 2px 4px rgba(0,0,0,0.35); transform: translateX(-50%);
}

/* Forecast cards */
.forecast-safe, .forecast-hyper, .forecast-hypo {
    border-radius: 12px; padding: 14px 18px; box-shadow: 0 1px 3px rgba(15,23,42,0.05); margin-bottom: 10px;
}
.forecast-safe {background: #F0FDF4; border-left: 6px solid #10B981;}
.forecast-hyper {background: #FFFBEB; border-left: 6px solid #F59E0B;}
.forecast-hypo {background: #FEF2F2; border-left: 6px solid #EF4444;}
.score-box {background: rgba(255,255,255,0.75); border: 1px solid #E2E8F0; border-radius: 10px; padding: 8px 12px; min-width: 250px;}
.score-row {display: flex; justify-content: space-between; gap: 10px; font-size: 12px; color: #334155; margin-top: 3px;}
.score-track {height: 6px; background: #E2E8F0; border-radius: 3px; position: relative; margin-top: 2px;}
.score-fill {height: 6px; border-radius: 3px; position: absolute; left: 0; top: 0;}
.score-thr {position: absolute; top: -3px; width: 2px; height: 12px; background: #0F172A;}
.action-box {margin-top: 10px; font-size: 13px; line-height: 1.5; background: rgba(255,255,255,0.6); border-radius: 8px; padding: 8px 12px;}

/* Validation checklist */
.check-step {
    display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 8px 12px;
    background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; margin-bottom: 7px; font-size: 13px;
}
.badge {font-size: 11px; font-weight: 700; padding: 2px 9px; border-radius: 10px; white-space: nowrap;}
.badge-PASS {background: #DCFCE7; color: #15803D;}
.badge-WARN {background: #FEF3C7; color: #B45309;}
.badge-FAIL {background: #FEE2E2; color: #DC2626;}
.badge-SKIPPED {background: #E2E8F0; color: #475569;}

/* Model card tables */
.mc-table {width: 100%; border-collapse: collapse; font-size: 12.5px; font-variant-numeric: tabular-nums;}
.mc-table th {text-align: left; color: #64748B; font-weight: 700; font-size: 11px; text-transform: uppercase; padding: 6px 8px; border-bottom: 1px solid #E2E8F0;}
.mc-table td {padding: 6px 8px; border-bottom: 1px solid #F1F5F9; color: #0F172A;}
.mc-table tr.hl td {background: #F0FDFA; font-weight: 700;}
</style>
""", unsafe_allow_html=True)

st.markdown(f"<div class='gt-disclaimer'>⚠ {DISCLAIMER}</div>", unsafe_allow_html=True)

if IMPORT_ERROR is not None:
    st.error(f"Could not import models/train_fusion_model.py ({type(IMPORT_ERROR).__name__}: {IMPORT_ERROR}). "
             "The dashboard needs its feature and encoding helpers.")
    st.stop()


# ==============================================================================
# Loaders (cached; keyed on file modification times so a retrain is picked up)
# ==============================================================================
def _mtime(path):
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


DATA_KEY = (_mtime(DATA_DIR / "synthetic_ehr.csv"), _mtime(DATA_DIR / "synthetic_wearables.csv"))
MODEL_KEY = _mtime(MODEL_PATH)
METRICS_KEY = _mtime(METRICS_PATH)


@st.cache_resource(show_spinner="Loading synthetic EHR and wearable data ...")
def get_data(data_key):
    # tfm.load_data reads the EHR with keep_default_na=False so "None" stays a string.
    ehr, wear = tfm.load_data(str(DATA_DIR))
    wear["timestamp"] = pd.to_datetime(wear["timestamp"])
    wear = wear.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    return ehr, wear


@st.cache_resource(show_spinner="Loading trained model ...")
def get_model(model_key):
    if not MODEL_PATH.exists():
        return None, f"{MODEL_PATH.name} not found. Run: python models/train_fusion_model.py"
    try:
        bundle = joblib.load(MODEL_PATH)
    except Exception as exc:
        return None, f"Could not load {MODEL_PATH.name} ({type(exc).__name__}: {exc})."
    missing = [k for k in ("classifier", "regressor", "feature_cols", "alert_thresholds", "test_patients")
               if k not in bundle]
    if missing:
        return None, f"Model bundle is missing {missing}. Re-run python models/train_fusion_model.py."
    if bundle.get("category_levels") not in (None, CATEGORY_LEVELS):
        return None, "Category levels in the bundle differ from train_fusion_model.py. Retrain the model."
    if list(getattr(bundle["classifier"], "classes_", [])) != [0, 1, 2]:
        return None, "Classifier classes are not [0, 1, 2]; the alert rule cannot be applied."
    return bundle, None


@st.cache_data(show_spinner=False)
def get_metrics(metrics_key):
    try:
        return json.loads(METRICS_PATH.read_text()), None
    except FileNotFoundError:
        return None, "models/metrics.json not found. Run: python models/evaluate.py"
    except Exception as exc:
        return None, f"Could not read models/metrics.json ({type(exc).__name__}: {exc})."


@st.cache_data(show_spinner=False)
def patient_fused(pid, data_key):
    """The patient's fused EHR + wearable feature rows, built with the training helpers."""
    ehr, wear = get_data(data_key)
    feats = compute_wearable_features(wear[wear["patient_id"] == pid])
    fused, _ = prepare_fused_dataset(ehr[ehr["patient_id"] == pid], feats)
    return fused.sort_values("timestamp").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def patient_predictions(pid, data_key, model_key):
    """Classifier scores, thresholded alert and +2 h regressor value for every window."""
    bundle, err = get_model(model_key)
    if bundle is None:
        raise RuntimeError(err)
    X = encode_features(patient_fused(pid, data_key), bundle["feature_cols"])
    proba = bundle["classifier"].predict_proba(X)
    return pd.DataFrame({
        "p_hypo": proba[:, 1],
        "p_hyper": proba[:, 2],
        "alert": apply_alert_thresholds(proba, bundle["alert_thresholds"]),
        "forecast_2h": bundle["regressor"].predict(X),
    })


@st.cache_resource(show_spinner="Computing features for every patient ...")
def all_fused(data_key):
    ehr, wear = get_data(data_key)
    fused, _ = prepare_fused_dataset(ehr, compute_wearable_features(wear))
    return fused.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)


@st.cache_data(show_spinner="Scoring every patient ...")
def risk_board(as_of_iso, data_key, model_key):
    """One batched predict over each patient's latest window at or before `as_of`."""
    bundle, err = get_model(model_key)
    if bundle is None:
        raise RuntimeError(err)
    fused = all_fused(data_key)
    latest = fused[fused["timestamp"] <= pd.Timestamp(as_of_iso)].groupby("patient_id", sort=False).tail(1)
    X = encode_features(latest, bundle["feature_cols"])
    proba = bundle["classifier"].predict_proba(X)
    alert = apply_alert_thresholds(proba, bundle["alert_thresholds"])
    test_set = set(bundle["test_patients"])
    board = pd.DataFrame({
        "patient_id": latest["patient_id"].to_numpy(),
        "name": latest["name"].to_numpy(),
        "set": ["held-out" if p in test_set else "training" for p in latest["patient_id"]],
        "risk_tier": latest["clinical_risk_tier"].to_numpy(),
        "medication": latest["primary_medication"].to_numpy(),
        "window": latest["timestamp"].dt.strftime("%d %b %H:%M").to_numpy(),
        "cgm": latest["glucose_cgm"].to_numpy(),
        "trend": latest["glucose_velocity_15m"].to_numpy(),
        "forecast": bundle["regressor"].predict(X),
        "p_hypo": proba[:, 1],
        "p_hyper": proba[:, 2],
        "alert": alert,
    })
    board["max_score"] = board[["p_hypo", "p_hyper"]].max(axis=1)
    return board.sort_values("max_score", ascending=False).reset_index(drop=True)


# ==============================================================================
# Small helpers
# ==============================================================================
def esc(x):
    return html.escape(str(x))


def mget(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def fnum(x, fmt="{:.1f}", na="n/a"):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return na
        return fmt.format(x)
    except (TypeError, ValueError):
        return na


def fpct(frac, digits=1, na="n/a"):
    """Formats a 0-1 fraction as a percentage."""
    return fnum(None if frac is None else frac * 100.0, "{:." + str(digits) + "f}%", na)


def is_insulin(ehr_row):
    return str(ehr_row.get("insulin_regimen", "None")).strip() not in ("", "None", "nan")


def hypo_drug_hint(ehr_row):
    """Names the patient's hypoglycaemia-causing drug class, if any."""
    med = str(ehr_row.get("primary_medication", ""))
    parts = []
    if "Sulfonylurea" in med:
        inner = med.split("(")[-1].rstrip(")") if "(" in med else ""
        parts.append(f"sulfonylurea ({inner.lower()})" if inner else "sulfonylurea")
    if is_insulin(ehr_row):
        parts.append("insulin")
    return " and ".join(parts)


def glucose_state(g):
    """(label, colour) for a current CGM value."""
    if g < LEVEL2_HYPO_MG_DL:
        return f"Level 2 hypoglycaemia (<{LEVEL2_HYPO_MG_DL:.0f})", RED_DARK
    if g < HYPO_MG_DL:
        return f"Level 1 hypoglycaemia ({LEVEL2_HYPO_MG_DL:.0f}–{HYPO_MG_DL - 1:.0f})", RED
    if g <= TARGET_HIGH_MG_DL:
        return f"In target range ({HYPO_MG_DL:.0f}–{TARGET_HIGH_MG_DL:.0f})", GREEN
    if g <= HYPER_MG_DL:
        return f"Above range ({TARGET_HIGH_MG_DL + 1:.0f}–{HYPER_MG_DL:.0f})", AMBER
    return f"Marked hyperglycaemia (>{HYPER_MG_DL:.0f})", RED


def glucose_colour(g):
    return glucose_state(g)[1]


MODEL_LOOKBACK_ROWS = 8   # the longest feature window in compute_wearable_features (2 h = 8 rows)


def to_model_windows(up):
    """Puts an uploaded wearable stream on the model's time base before scoring.

    compute_wearable_features is row-based: one row must be one 15-min window (1 h = 4 rows, 2 h = 8 rows).
    A file at a 15-min cadence is used as is. A finer file (e.g. 5-min) is aggregated into 15-min windows:
    glucose is the reading at the start of each window (as in the training data, where glucose_cgm is the
    reading at the window timestamp), HR / HRV are averaged, steps / carbs / insulin are summed and the most
    common sleep stage is kept; incomplete windows at either end are dropped. A coarser file, or one with a gap inside the latest 2 h, is not scored.
    Returns (frame, note, problem); problem is None when the frame can be scored.
    """
    up = up.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    if len(up) < 2:
        return None, "", "the file has fewer than two timestamped readings."
    med = float(up["timestamp"].diff().dropna().dt.total_seconds().median() / 60)
    note = ""
    if abs(med - STEP_MIN) <= 1:
        frame = up
    elif med < STEP_MIN - 1:
        per_window = max(1, int(round(STEP_MIN / med)))
        r = up.set_index("timestamp").resample(f"{STEP_MIN}min")
        frame = pd.DataFrame({
            "n": r["glucose_cgm"].count(),
            "glucose_cgm": r["glucose_cgm"].first(),
            "heart_rate": r["heart_rate"].mean(),
            "hrv": r["hrv"].mean(),
            "steps_15m": r["steps_15m"].sum(min_count=1),
            "sleep_stage": r["sleep_stage"].agg(lambda s: s.mode().iloc[0] if s.notna().any() else np.nan),
            "meal_carbs_g": r["meal_carbs_g"].sum(min_count=1),
            "insulin_units": r["insulin_units"].sum(min_count=1),
        }).reset_index()
        full = frame["n"] >= per_window
        if full.any():
            first, last = full.idxmax(), full[::-1].idxmax()
            frame = frame.loc[first:last]
        frame = frame[frame["n"] > 0].drop(columns="n").reset_index(drop=True)
        note = (f"Resampled from a {med:.0f}-min cadence ({len(up)} rows) to {len(frame)} {STEP_MIN}-min windows "
                "(glucose = reading at each window start, as in training; mean HR and HRV; summed steps, carbs "
                "and insulin). ")
    else:
        return None, "", (f"the median interval is {med:.0f} min, but the model needs {STEP_MIN}-min windows "
                          f"(finer data is aggregated automatically; coarser data cannot be).")
    tail = frame.tail(MODEL_LOOKBACK_ROWS)
    steps = tail["timestamp"].diff().dropna().dt.total_seconds() / 60
    if len(steps) and (steps - STEP_MIN).abs().max() > 1:
        return None, "", (f"the latest 2 h has a gap (largest interval {steps.max():.0f} min); the model's 1–2 h "
                          f"features assume {MODEL_LOOKBACK_ROWS} consecutive {STEP_MIN}-min windows.")
    if len(frame) < MODEL_LOOKBACK_ROWS:
        note += (f"Only {len(frame)} {STEP_MIN}-min windows, less than 2 h of history, so the 1–2 h sums are "
                 "partial. ")
    return frame, note, None


def fmt_clock(ts, t0):
    day = (ts.normalize() - t0.normalize()).days + 1
    return f"Day {day} · {ts.strftime('%a %d %b %H:%M')}"


def alert_runs(alert):
    """(start, end, kind) for each contiguous run of the same non-zero alert."""
    runs, start = [], None
    for j, a in enumerate(list(alert) + [0]):
        if start is not None and a != alert[start]:
            runs.append((start, j - 1, int(alert[start])))
            start = None
        if start is None and a != 0 and j < len(alert):
            start = j
    return runs


def plot_layout(title, height=380, y_range=None, legend_bottom=True):
    layout = dict(
        title=dict(text=title, font=dict(family="Inter, sans-serif", size=14, color=NAVY), x=0, xanchor="left"),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=SLATE, size=11),
        xaxis=dict(showgrid=True, gridcolor="#F1F5F9", zeroline=False, linecolor="#E2E8F0"),
        yaxis=dict(showgrid=True, gridcolor="#F1F5F9", zeroline=False, linecolor="#E2E8F0"),
        margin=dict(l=50, r=20, t=44, b=40),
        hovermode="x unified",
    )
    if y_range is not None:
        layout["yaxis"]["range"] = y_range
    if legend_bottom:
        layout["legend"] = dict(orientation="h", yanchor="top", y=-0.14, xanchor="left", x=0)
        layout["margin"]["b"] = 70
    return layout


def add_glucose_bands(fig, y_top, y_bottom=40.0, compact=False):
    """Clinical bands. compact=True (short charts) puts the Level 2 label on the right so it cannot overlap Level 1."""
    fig.add_hrect(y0=min(y_bottom, LEVEL2_HYPO_MG_DL - 4), y1=LEVEL2_HYPO_MG_DL, fillcolor="rgba(185,28,28,0.14)", line_width=0,
                  annotation_text=f"Level 2 hypo (<{LEVEL2_HYPO_MG_DL:.0f})",
                  annotation_position="bottom right" if compact else "bottom left",
                  annotation_font=dict(size=10, color=RED_DARK))
    fig.add_hrect(y0=LEVEL2_HYPO_MG_DL, y1=HYPO_MG_DL, fillcolor="rgba(239,68,68,0.10)", line_width=0,
                  annotation_text=f"Level 1 hypo (<{HYPO_MG_DL:.0f})", annotation_position="top left",
                  annotation_font=dict(size=10, color=RED))
    fig.add_hrect(y0=HYPO_MG_DL, y1=TARGET_HIGH_MG_DL, fillcolor="rgba(16,185,129,0.10)", line_width=1,
                  line_dash="dash", line_color="rgba(16,185,129,0.4)",
                  annotation_text=f"Target {HYPO_MG_DL:.0f}–{TARGET_HIGH_MG_DL:.0f}", annotation_position="top left",
                  annotation_font=dict(size=10, color="#059669"))
    fig.add_hrect(y0=HYPER_MG_DL, y1=max(y_top, HYPER_MG_DL + 10), fillcolor="rgba(245,158,11,0.10)", line_width=0,
                  annotation_text=f"Marked hyperglycaemia (>{HYPER_MG_DL:.0f})", annotation_position="top left",
                  annotation_font=dict(size=10, color="#B45309"))


def add_now_line(fig, x, label="now"):
    fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, xref="x", yref="paper",
                  line=dict(color="#334155", width=1, dash="dot"))
    fig.add_annotation(x=x, y=1, xref="x", yref="paper", text=label, showarrow=False,
                       yanchor="bottom", font=dict(size=10, color="#334155"))


def badge(status):
    return f"<span class='badge badge-{status}'>{status}</span>"


def page_title(title, lede):
    st.markdown(f"<div class='gt-title'>{title}</div><div class='gt-lede'>{lede}</div>", unsafe_allow_html=True)


# ==============================================================================
# Load everything
# ==============================================================================
try:
    df_ehr, df_wear = get_data(DATA_KEY)
except Exception as exc:
    st.error(f"Could not load data/synthetic_ehr.csv and data/synthetic_wearables.csv "
             f"({type(exc).__name__}: {exc}). Run: python data/generate_datasets.py")
    st.stop()

bundle, MODEL_ERROR = get_model(MODEL_KEY)
metrics, METRICS_ERROR = get_metrics(METRICS_KEY)
TEST_PATIENTS = set(bundle["test_patients"]) if bundle is not None else set()
THRESHOLDS = bundle["alert_thresholds"] if bundle is not None else None

pids = sorted(df_ehr["patient_id"].astype(str).unique())
wear_pids = set(df_wear["patient_id"].unique())
ehr_by_pid = df_ehr.set_index("patient_id")


def patient_label(pid):
    name = ehr_by_pid.loc[pid, "name"]
    if bundle is None:
        tag = ""
    else:
        tag = " · held-out (unseen)" if pid in TEST_PATIENTS else " · training"
    if pid not in wear_pids:
        tag += " · no wearable stream"
    return f"{pid}{tag} · {name}"


def model_error_box(context="This view"):
    st.error(f"Prediction model unavailable: {MODEL_ERROR} {context} shows no forecast.")


# ---- Session state: navigation, patient, shared replay clock -------------------
if st.session_state.get("nav") not in NAV_ITEMS:
    st.session_state["nav"] = NAV_ITEMS[0]
if st.session_state.get("patient_id") not in pids:
    if DEFAULT_PATIENT in pids:
        st.session_state["patient_id"] = DEFAULT_PATIENT
    else:
        held = [p for p in pids if p in TEST_PATIENTS and p in wear_pids]
        st.session_state["patient_id"] = held[0] if held else pids[0]


def open_patient(pid, t_idx=None):
    """Callback: jump to the Live Monitor for `pid` (used by the risk board)."""
    st.session_state["patient_id"] = pid
    st.session_state["nav"] = "Live Monitor"
    if t_idx is not None:
        st.session_state["t_idx"] = int(t_idx)


# ==============================================================================
# SIDEBAR: brand, navigation, Patient-360
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:14px;">
        <div style="width:38px; height:38px; border-radius:10px; background:#0D9488; display:flex; align-items:center; justify-content:center; color:white; font-size:20px;">🧬</div>
        <div>
            <div style="font-size:18px; font-weight:800; color:#F8FAFC; letter-spacing:-0.02em;">GlycoTwin</div>
            <div style="font-size:11px; color:#94A3B8;">T2D digital twin · research prototype</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:#64748B; margin-bottom:6px;'>Navigation</p>", unsafe_allow_html=True)
    nav_tab = st.radio("Navigation", NAV_ITEMS, key="nav", label_visibility="collapsed")

    st.markdown("<p style='font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:#64748B; margin:14px 0 6px 0;'>Patient 360° (synthetic persona)</p>", unsafe_allow_html=True)
    selected_pid = st.selectbox("Patient", pids, key="patient_id", format_func=patient_label,
                                label_visibility="collapsed",
                                help="All 200 synthetic personas. 'held-out (unseen)' patients were never used to "
                                     "train the model, so their views are an honest test.")
    pe = ehr_by_pid.loc[selected_pid]
    held_out = selected_pid in TEST_PATIENTS
    on_insulin = is_insulin(pe)

    tier = str(pe["clinical_risk_tier"])
    tier_bg = RED if "High" in tier else (AMBER if "Moderate" in tier else GREEN)
    set_chip = ("" if bundle is None else
                ("<span class='chip chip-heldout'>held-out (unseen)</span>" if held_out
                 else "<span class='chip chip-train'>training patient</span>"))
    diagnoses = [d.strip() for d in str(pe["past_diagnoses"]).split(";") if d.strip()]
    dx_html = "<br>".join(f"• {esc(d)}" for d in diagnoses) if diagnoses else "None recorded"
    alleles = pe["tcf7l2_rs7903146_risk_alleles"]
    try:
        alleles_txt = f"{int(alleles)} of 2 risk alleles (T)"
    except (TypeError, ValueError):
        alleles_txt = esc(alleles)

    st.markdown(f"""
    <div class="patient-360-card">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:6px; margin-bottom:6px;">
            <span style="font-weight:700; color:#F8FAFC; font-size:14px;">{esc(pe['name'])}</span>
            <span class="tier-badge" style="background:{tier_bg}; color:{'#451A03' if tier_bg == AMBER else '#FFFFFF'}; padding:2px 8px; border-radius:12px; font-size:10px; font-weight:700; white-space:nowrap;">{esc(tier)}</span>
        </div>
        <div style="display:flex; gap:6px; flex-wrap:wrap; margin-bottom:8px;">
            <span class="chip chip-synth">synthetic persona</span>{set_chip}
        </div>
        <div class="patient-360-row"><span class="patient-360-label">Patient ID</span><span class="patient-360-value tabular-nums">{esc(selected_pid)}</span></div>
        <div class="patient-360-row"><span class="patient-360-label">Age · sex</span><span class="patient-360-value">{esc(pe['age'])} y · {esc(pe['gender'])}</span></div>
        <div class="patient-360-row"><span class="patient-360-label">BMI</span><span class="patient-360-value tabular-nums">{esc(pe['bmi'])} kg/m²</span></div>
        <div class="patient-360-row"><span class="patient-360-label">Diabetes duration</span><span class="patient-360-value tabular-nums">{esc(pe['diabetes_duration_years'])} y</span></div>
        <div class="patient-360-row"><span class="patient-360-label">Baseline HbA1c</span><span class="patient-360-value tabular-nums" style="color:#2DD4BF;">{esc(pe['baseline_hba1c'])}%</span></div>
        <div class="patient-360-row"><span class="patient-360-label">Fasting glucose</span><span class="patient-360-value tabular-nums">{esc(pe['fasting_glucose_mg_dl'])} mg/dL</span></div>
        <div class="patient-360-row"><span class="patient-360-label">Blood pressure</span><span class="patient-360-value tabular-nums">{esc(pe['systolic_bp'])}/{esc(pe['diastolic_bp'])} mmHg</span></div>
        <div class="patient-360-row"><span class="patient-360-label">eGFR</span><span class="patient-360-value tabular-nums">{esc(pe['egfr'])} mL/min/1.73m²</span></div>
        <div class="patient-360-row"><span class="patient-360-label">LDL · TG</span><span class="patient-360-value tabular-nums">{esc(pe['ldl_cholesterol_mg_dl'])} · {esc(pe['triglycerides_mg_dl'])} mg/dL</span></div>
        <div class="patient-360-row patient-360-col"><span class="patient-360-label">Primary medication</span><span class="patient-360-value">{esc(pe['primary_medication'])}</span></div>
        <div class="patient-360-row patient-360-col"><span class="patient-360-label">Insulin regimen</span><span class="patient-360-value">{esc(pe['insulin_regimen']) if on_insulin else 'Not on insulin'}</span></div>
        <div class="patient-360-row patient-360-col"><span class="patient-360-label">Past diagnoses</span><span class="patient-360-value">{dx_html}</span></div>
        <div class="patient-360-row"><span class="patient-360-label">Hypo episodes (12 m)</span><span class="patient-360-value tabular-nums">{esc(pe['prior_hypo_episodes_12m'])}</span></div>
        <div class="patient-360-row"><span class="patient-360-label">Family history of T2D</span><span class="patient-360-value">{esc(pe['family_history_t2d'])}</span></div>
        <div class="patient-360-row patient-360-col" style="border-bottom:none;">
            <span class="patient-360-label">TCF7L2 rs7903146</span>
            <span class="patient-360-value">{alleles_txt}</span>
            <span class="p360-note">The T allele is linked to higher T2D risk and lower insulin secretion; in this synthetic data each allele raises post-meal glucose excursions.</span>
        </div>
    </div>
    <div style="margin-top:12px; font-size:11px; color:#64748B; line-height:1.45;">
        Happiest Health Digital Twin Challenge 2026<br>
        Model: scikit-learn HistGradientBoosting<br>
        <span style="color:#FDBA74;">{DISCLAIMER} Not for dosing decisions.</span>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# Patient timeline, predictions and the shared replay clock
# ==============================================================================
pw = patient_fused(selected_pid, DATA_KEY) if selected_pid in wear_pids else pd.DataFrame()
preds, PRED_ERROR = None, None
if bundle is not None and len(pw):
    try:
        preds = patient_predictions(selected_pid, DATA_KEY, MODEL_KEY)
    except Exception as exc:
        PRED_ERROR = f"{type(exc).__name__}: {exc}"

n_steps = len(pw)
times = list(pw["timestamp"]) if n_steps else []
alert_onsets = []
if preds is not None:
    alert_onsets = [s for s, _, _ in alert_runs(preds["alert"].to_numpy())]

if n_steps:
    if not isinstance(st.session_state.get("t_idx"), int) or not (0 <= st.session_state["t_idx"] < n_steps):
        # The replay opens at this patient's most recent model alert (if any), else at the latest reading.
        st.session_state["t_idx"] = int(alert_onsets[-1]) if alert_onsets else n_steps - 1


def _sync_clock():
    st.session_state["t_idx"] = int(st.session_state["_t_widget"])


def _shift_clock(delta):
    st.session_state["t_idx"] = int(np.clip(st.session_state["t_idx"] + delta, 0, n_steps - 1))


def _set_clock(idx):
    st.session_state["t_idx"] = int(idx)


def render_clock(show_alert_nav=True, caption=None):
    """Replay-time scrubber over the patient's 7-day synthetic record."""
    st.session_state["_t_widget"] = st.session_state["t_idx"]
    t0 = times[0]
    cols = st.columns([4.2, 0.75, 0.75, 1.15, 1.15, 0.85] if show_alert_nav else [5.5, 0.8, 0.8, 0.9])
    with cols[0]:
        st.select_slider("Replay time (15-min windows)", options=list(range(n_steps)), key="_t_widget",
                         format_func=lambda i: fmt_clock(times[i], t0), on_change=_sync_clock,
                         label_visibility="collapsed")
    i_now = st.session_state["t_idx"]
    with cols[1]:
        st.button("−1 h", key="clk_back", on_click=_shift_clock, args=(-4,), width="stretch", disabled=i_now == 0)
    with cols[2]:
        st.button("+1 h", key="clk_fwd", on_click=_shift_clock, args=(4,), width="stretch",
                  disabled=i_now >= n_steps - 1)
    if show_alert_nav:
        prev_a = [s for s in alert_onsets if s < i_now]
        next_a = [s for s in alert_onsets if s > i_now]
        with cols[3]:
            st.button("◀ Prev alert", key="clk_prev_alert", on_click=_set_clock,
                      args=(prev_a[-1] if prev_a else i_now,), width="stretch", disabled=not prev_a,
                      help="Jump to the start of the previous model alert for this patient")
        with cols[4]:
            st.button("Next alert ▶", key="clk_next_alert", on_click=_set_clock,
                      args=(next_a[0] if next_a else i_now,), width="stretch", disabled=not next_a,
                      help="Jump to the start of the next model alert for this patient")
        with cols[5]:
            st.button("Latest", key="clk_latest", on_click=_set_clock, args=(n_steps - 1,), width="stretch")
    else:
        with cols[3]:
            st.button("Latest", key="clk_latest", on_click=_set_clock, args=(n_steps - 1,), width="stretch")
    if caption:
        st.caption(caption)
    return st.session_state["t_idx"]


def render_patient_header(idx, extra_html=""):
    set_chip = ("" if bundle is None else
                ("<span class='chip chip-heldout'>held-out (unseen by model)</span>" if held_out
                 else "<span class='chip chip-train'>training patient</span>"))
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap; padding-bottom:6px; margin-bottom:8px; border-bottom:1px solid #E2E8F0;">
        <div>
            <div style="font-size:19px; font-weight:800; color:#0F172A; display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
                {esc(pe['name'])}
                <span style="font-size:12px; font-weight:500; color:#64748B;">{esc(selected_pid)} · {esc(pe['age'])} y · {esc(pe['gender'])}</span>
                <span class="chip chip-synth">synthetic persona</span>{set_chip}
            </div>
            <div style="font-size:12px; color:#64748B; margin-top:2px;">Selected window
                <span class="tabular-nums" style="font-weight:700; color:#0F172A;">{times[idx].strftime('%a %d %b %Y, %H:%M')}</span>
                · replay of the 7-day synthetic record; the model only sees data up to this time</div>
        </div>
        <div style="display:flex; gap:6px; flex-wrap:wrap;">{extra_html}</div>
    </div>
    """, unsafe_allow_html=True)


def pill(text, kind):
    styles = {"red": ("#FEE2E2", "#B91C1C"), "amber": ("#FEF3C7", "#B45309"), "green": ("#DCFCE7", "#15803D"),
              "grey": ("#E2E8F0", "#334155")}
    bg, fg = styles[kind]
    return (f"<span style='background:{bg}; color:{fg}; padding:3px 10px; border-radius:12px; "
            f"font-size:12px; font-weight:700;'>{text}</span>")


def no_stream_box():
    st.warning(f"{selected_pid} has an EHR record but no wearable stream in data/synthetic_wearables.csv, "
               "so there is nothing to monitor or forecast for this patient.")


# ==============================================================================
# VIEW 1: LIVE MONITOR
# ==============================================================================
if nav_tab == "Live Monitor":
    if not n_steps:
        no_stream_box()
        st.stop()

    i = render_clock(show_alert_nav=preds is not None, caption="")
    row = pw.iloc[i]
    t_now = row["timestamp"]
    g_now = float(row["glucose_cgm"])
    roc = float(row["glucose_velocity_15m"])
    roc_symbol = "↗" if roc > 2.0 else ("↘" if roc < -2.0 else "→")
    state_txt, state_col = glucose_state(g_now)
    t_fc = t_now + timedelta(minutes=STEP_MIN * HORIZON_STEPS)

    have_pred = preds is not None
    if have_pred:
        p = preds.iloc[i]
        alert_k = int(p["alert"])
        p_hypo, p_hyper, fc = float(p["p_hypo"]), float(p["p_hyper"]), float(p["forecast_2h"])

    # The current reading takes priority over the model: a CGM already out of range is never shown as "safe".
    now_low, now_high = g_now < HYPO_MG_DL, g_now > HYPER_MG_DL
    now_lvl = "Level 2" if g_now < LEVEL2_HYPO_MG_DL else "Level 1"
    pills = []
    if now_low:
        pills.append(pill(f"Hypoglycaemia now ({now_lvl})", "red"))
    elif now_high:
        pills.append(pill("Marked hyperglycaemia now", "amber"))
    if have_pred:
        if alert_k == 1:
            pills.append(pill("Model alert: hypoglycaemia risk within 2 h", "red"))
        elif alert_k == 2:
            pills.append(pill("Model alert: marked hyperglycaemia risk within 2 h", "amber"))
        elif now_low or now_high:
            pills.append(pill("Model: no further 2 h alert", "grey"))
        else:
            pills.append(pill("No model alert", "green"))
    render_patient_header(i, "".join(pills))

    if not have_pred:
        if bundle is None:
            model_error_box("The Live Monitor")
        else:
            st.error(f"Prediction failed for {selected_pid}: {PRED_ERROR}. No forecast is shown.")

    # ---- 1. CGM chart (left) + current glucose and 2-hour model output (right) ----
    col_chart, col_side = st.columns([1.8, 1.2], gap="medium")
    with col_side:
        needle = max(0.0, min(100.0, (g_now - 40.0) / 320.0 * 100.0))
        st.markdown(f"""
        <div class="gt-card">
            <div class="gt-label">Current CGM glucose</div>
            <div style="margin-top:6px;"><span style="background:{state_col}1F; color:{state_col}; font-size:11px; font-weight:700; padding:2px 8px; border-radius:10px;">{state_txt}</span></div>
            <div style="display:flex; align-items:baseline; gap:6px; margin-top:8px;">
                <div class="tabular-nums" style="font-size:44px; font-weight:800; color:#0F172A; line-height:1;">{g_now:.1f}</div>
                <div style="font-size:13px; color:#64748B; font-weight:500;">mg/dL</div>
            </div>
            <div class="tabular-nums" style="font-size:14px; font-weight:700; color:{state_col}; margin-top:4px;">{roc_symbol} {roc:+.1f} mg/dL per 15 min</div>
            <div class="range-bar-track"><div class="range-bar-needle" style="left:{needle:.1f}%;"></div></div>
            <div style="display:flex; justify-content:space-between; font-size:10px; color:#94A3B8; margin-top:4px;">
                <span>{LEVEL2_HYPO_MG_DL:.0f}</span><span>{HYPO_MG_DL:.0f}</span>
                <span style="color:#059669; font-weight:600;">target</span>
                <span>{TARGET_HIGH_MG_DL:.0f}</span><span>{HYPER_MG_DL:.0f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if have_pred:
            # One line describing the model's 2 h output (shown as the secondary line when CGM is out of range now).
            if alert_k == 1:
                model_line = (f"Model, next 2 h: <b>hypoglycaemia risk alert</b> (hypo score {p_hypo:.2f} ≥ "
                              f"{THRESHOLDS['hypo']:.2f}), a reading below {HYPO_MG_DL:.0f} mg/dL expected before "
                              f"{t_fc.strftime('%H:%M')}.")
            elif alert_k == 2:
                model_line = (f"Model, next 2 h: <b>marked hyperglycaemia risk alert</b> (hyper score {p_hyper:.2f} ≥ "
                              f"{THRESHOLDS['hyper']:.2f}), a reading above {HYPER_MG_DL:.0f} mg/dL expected before "
                              f"{t_fc.strftime('%H:%M')}.")
            else:
                model_line = "Model, next 2 h: no further alert (both risk scores are below their alert thresholds)."

            if now_low:
                card_cls, head_col, body_col = "forecast-hypo", "#B91C1C", "#7F1D1D"
                kicker = "Current CGM reading · takes priority over the model"
                headline = f"Hypoglycaemia now ({now_lvl})"
                sub = (f"CGM is {g_now:.0f} mg/dL now, below {HYPO_MG_DL:.0f} mg/dL"
                       + (f" and below {LEVEL2_HYPO_MG_DL:.0f} mg/dL" if now_lvl == "Level 2" else "")
                       + f".<br>{model_line}")
            elif now_high:
                card_cls, head_col, body_col = "forecast-hyper", "#B45309", "#78350F"
                kicker = "Current CGM reading · takes priority over the model"
                headline = "Marked hyperglycaemia now"
                sub = f"CGM is {g_now:.0f} mg/dL now, above {HYPER_MG_DL:.0f} mg/dL.<br>{model_line}"
            elif alert_k == 1:
                card_cls, head_col, body_col = "forecast-hypo", "#B91C1C", "#7F1D1D"
                kicker = "Model alert · next 2 hours"
                headline = "Hypoglycaemia risk within 2 h"
                sub = f"The classifier expects at least one CGM reading below {HYPO_MG_DL:.0f} mg/dL before {t_fc.strftime('%H:%M')}."
            elif alert_k == 2:
                card_cls, head_col, body_col = "forecast-hyper", "#B45309", "#78350F"
                kicker = "Model alert · next 2 hours"
                headline = "Marked hyperglycaemia risk within 2 h"
                sub = f"The classifier expects at least one CGM reading above {HYPER_MG_DL:.0f} mg/dL before {t_fc.strftime('%H:%M')}."
            else:
                card_cls, head_col, body_col = "forecast-safe", "#047857", "#064E3B"
                kicker = "Model output · next 2 hours"
                headline = "No event predicted within 2 h"
                sub = (f"CGM is {g_now:.0f} mg/dL now"
                       + (f" (above the {TARGET_HIGH_MG_DL:.0f} mg/dL target, not above {HYPER_MG_DL:.0f})"
                          if g_now > TARGET_HIGH_MG_DL else "")
                       + ". Neither risk score is at or above its alert threshold.")

            # Action text: rescue protocol only when CGM is low NOW; never insulin dose numbers.
            actions = []
            if now_low:
                actions.append(f"<b>CGM is {g_now:.0f} mg/dL now ({now_lvl} hypoglycaemia).</b> Standard 15-15 rule: "
                               f"15 g fast-acting carbohydrate (e.g. glucose tablets or ~150 ml juice), recheck in "
                               f"15 min, repeat if still below {HYPO_MG_DL:.0f} mg/dL."
                               + (" Level 2 is clinically significant: inform the care team." if now_lvl == "Level 2" else ""))
            elif now_high:
                actions.append(f"<b>CGM is {g_now:.0f} mg/dL now (marked hyperglycaemia).</b> Follow the patient's "
                               "hyperglycaemia plan; review the recent meal, missed doses and hydration with the "
                               "care team." + (" This tool gives no insulin dose advice." if on_insulin else ""))
            if alert_k == 1 and not now_low:
                drug = hypo_drug_hint(pe)
                actions.append("<b>Predicted low:</b> consider a snack and review medication timing"
                               + (f" ({drug})" if drug else "") + " with the care team. Watch the CGM trend.")
            elif alert_k == 2 and not now_high:
                txt = ("<b>Predicted marked hyperglycaemia:</b> review the recent meal, missed doses and hydration "
                       "with the care team.")
                if on_insulin:
                    txt += " Review bolus timing with the care team (this tool gives no dose advice)."
                actions.append(txt)
            if not actions:
                actions.append("No model alert. Continue the usual care plan.")
            actions_html = "<br>".join(actions)

            def score_row(label, score, thr, colour):
                pct = max(0.0, min(1.0, score)) * 100
                return (f'<div class="score-row"><span>{label}</span><span class="tabular-nums"><b>{score:.2f}</b>'
                        f' · alert ≥ {thr:.2f}</span></div><div class="score-track"><div class="score-fill" '
                        f'style="width:{pct:.1f}%; background:{colour};"></div><div class="score-thr" '
                        f'style="left:{thr * 100:.1f}%;"></div></div>')

            fc_txt_col = glucose_colour(fc)
            hypo_row = score_row(f"Hypo &lt;{HYPO_MG_DL:.0f} in 2 h", p_hypo, THRESHOLDS["hypo"], RED)
            hyper_row = score_row(f"Hyper &gt;{HYPER_MG_DL:.0f} in 2 h", p_hyper, THRESHOLDS["hyper"], AMBER)
            st.markdown(f"""
            <div class="{card_cls}">
                <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:16px; flex-wrap:wrap;">
                    <div style="flex:1; min-width:240px;">
                        <div style="font-size:11.5px; font-weight:800; color:{head_col}; letter-spacing:0.04em; text-transform:uppercase;">{kicker}</div>
                        <div style="font-size:20px; font-weight:800; color:{body_col}; margin-top:2px; line-height:1.25;">{headline}</div>
                        <div style="font-size:12.5px; color:{body_col}; margin-top:2px;">{sub}</div>
                        <div style="font-size:13px; color:#0F172A; margin-top:8px;">Forecast glucose at +2 h ({t_fc.strftime('%H:%M')}):
                            <b class="tabular-nums" style="font-size:16px; color:{fc_txt_col};">{fc:.0f} mg/dL</b></div>
                        <div style="font-size:11px; color:#64748B;">Regressor point forecast for exactly +2 h; the alert covers any reading in the next 2 h, so the two can differ.</div>
                    </div>
                    <div class="score-box" style="min-width:0; width:100%;">
                        <div style="font-size:10.5px; font-weight:700; color:#64748B; text-transform:uppercase;">Model risk score (uncalibrated)</div>
                        {hypo_row}{hyper_row}
                        <div style="font-size:10px; color:#64748B; margin-top:6px;">HistGradientBoosting · thresholds from grouped CV on training patients</div>
                    </div>
                </div>
                <div class="action-box" style="color:{body_col};">{actions_html}
                    <div style="font-size:11px; color:#64748B; margin-top:4px;">Decision support for clinician review only. Not for dosing decisions.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<div class='gt-card' style='color:#64748B;'>No forecast: the prediction model is "
                        "unavailable (see the error above).</div>", unsafe_allow_html=True)

    # ---- 2. CGM history + forecast chart ---------------------------------------
    hist = pw.iloc[max(0, i - HISTORY_STEPS):i + 1]
    fut = pw.iloc[i + 1:i + 1 + HORIZON_STEPS]
    in_range = hist["glucose_cgm"].between(HYPO_MG_DL, TARGET_HIGH_MG_DL).mean() * 100.0
    w_mean = hist["glucose_cgm"].mean()
    w_cv = hist["glucose_cgm"].std() / w_mean * 100.0 if len(hist) > 1 and w_mean > 0 else float("nan")
    show_actual = held_out and bundle is not None
    y_vals = list(hist["glucose_cgm"]) + ([fc] if have_pred else []) + (list(fut["glucose_cgm"]) if show_actual else [])
    y_lo, y_hi = min(40.0, min(y_vals) - 10), max(300.0, max(y_vals) + 20)

    fig = go.Figure()
    add_glucose_bands(fig, y_hi, y_lo)
    fig.add_trace(go.Scatter(x=hist["timestamp"], y=hist["glucose_cgm"], mode="lines+markers",
                             name="CGM (actual, seen by model)", line=dict(color=TEAL, width=2.5),
                             marker=dict(size=6, color=TEAL),
                             hovertemplate="%{y:.0f} mg/dL<extra>CGM</extra>"))
    if show_actual and len(fut):
        fig.add_trace(go.Scatter(x=[t_now] + list(fut["timestamp"]), y=[g_now] + list(fut["glucose_cgm"]),
                                 mode="lines+markers", name="actual (hidden from model)",
                                 line=dict(color="#475569", width=2, dash="dash"),
                                 marker=dict(size=7, color="#FFFFFF", line=dict(color="#475569", width=1.5)),
                                 hovertemplate="%{y:.0f} mg/dL<extra>actual (hidden)</extra>"))
    if have_pred:
        fc_col = glucose_colour(fc) if glucose_colour(fc) != GREEN else "#0F766E"
        fig.add_trace(go.Scatter(x=[t_now, t_fc], y=[g_now, fc], mode="lines",
                                 name="interpolated to +2 h forecast", line=dict(color=fc_col, width=2, dash="dot"),
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[t_fc], y=[fc], mode="markers", name="forecast at +2 h (regressor)",
                                 marker=dict(size=13, color=fc_col, symbol="diamond", line=dict(color="#FFFFFF", width=2)),
                                 hovertemplate="%{y:.0f} mg/dL<extra>forecast +2 h</extra>"))
    add_now_line(fig, t_now)
    fig.update_layout(**plot_layout("", height=560, y_range=[y_lo, y_hi]))
    fig.update_layout(margin=dict(l=50, r=12, t=10, b=60))
    fig.update_yaxes(title_text="mg/dL")
    with col_chart:
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:baseline; gap:10px; flex-wrap:wrap;">
            <span style="font-size:14px; font-weight:700; color:#0F172A;">CGM: last 8 h and forecast at +2 h</span>
            <span style="font-size:12px; color:#64748B;">Last 8 h: time in {HYPO_MG_DL:.0f}–{TARGET_HIGH_MG_DL:.0f}
                <b class="tabular-nums" style="color:#059669;">{in_range:.0f}%</b> · CV
                <b class="tabular-nums" style="color:#0F172A;">{fnum(w_cv, '{:.0f}%')}</b> (target ≤{CV_TARGET_PCT}%)</span>
        </div>""", unsafe_allow_html=True)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        if bundle is not None and not held_out:
            st.caption("Actual future readings are overlaid only for held-out patients.")

    with col_chart:
        # Retrospective check: only meaningful (and only shown) for held-out patients.
        if have_pred and held_out and len(fut) == HORIZON_STEPS:
            lo, hi = float(fut["glucose_cgm"].min()), float(fut["glucose_cgm"].max())
            actual_k = int(row["adverse_event_2h"])
            actual_txt = {0: "no event", 1: f"hypoglycaemia (<{HYPO_MG_DL:.0f})",
                          2: f"marked hyperglycaemia (>{HYPER_MG_DL:.0f})"}[actual_k]
            if alert_k and alert_k == actual_k:
                verdict, vcol = "alert was correct", GREEN
            elif alert_k and alert_k != actual_k:
                verdict, vcol = "false alarm", AMBER
            elif actual_k:
                verdict, vcol = "missed event", RED
            else:
                verdict, vcol = "correct: no event", GREEN
            st.markdown(f"""
            <div class="gt-card" style="padding:9px 14px; font-size:12.5px; color:#334155;">
                <b>Retrospective check</b> (held-out patient; these readings were hidden from the model):
                next 2 h ranged <b class="tabular-nums">{lo:.0f}–{hi:.0f} mg/dL</b> → actual outcome: <b>{actual_txt}</b>.
                <span style="color:{vcol}; font-weight:700;">Verdict: {verdict}.</span>
            </div>""", unsafe_allow_html=True)
        elif have_pred and not held_out:
            st.caption("Training patient: the model was fitted on this patient's data, so agreement here is not "
                       "evidence of accuracy. Choose a held-out (unseen) patient for an honest check.")

    # ---- 3. Wearable context ---------------------------------------------------
    stage = str(row["sleep_stage"])
    if stage != "Awake":
        phase = f"Asleep · {stage} sleep"
    elif t_now.hour < 6:
        phase = "Awake overnight"
    else:
        phase = "Awake"
    if on_insulin:
        regimen_line = (f"Insulin last 2 h: <b class='tabular-nums' style='color:#0F172A;'>"
                        f"{row['insulin_sum_2h']:.0f} U</b> (logged)")
    else:
        regimen_line = f"Not on insulin · {esc(pe['primary_medication'])}"
    v1, v2, v3, v4 = st.columns(4)
    vit = [
        (v1, "Heart rate &amp; HRV", f"{int(row['heart_rate'])} <span style='font-size:12px; font-weight:500; color:#64748B;'>bpm</span>",
         f"HRV <b class='tabular-nums' style='color:#0F172A;'>{int(row['hrv'])} ms</b> · 1 h mean HR {row['hr_mean_1h']:.0f}"),
        (v2, "Sleep stage (wearable)", esc(stage), phase),
        (v3, "Activity (15 min)", f"{int(row['steps_15m'])} <span style='font-size:12px; font-weight:500; color:#64748B;'>steps</span>",
         f"Last 1 h <b class='tabular-nums' style='color:#0F172A;'>{int(row['steps_sum_1h'])}</b> · last 2 h "
         f"<b class='tabular-nums' style='color:#0F172A;'>{int(row['steps_sum_2h'])}</b>"),
        (v4, "Meals &amp; medication", f"{row['carbs_sum_2h']:.0f} g <span style='font-size:12px; font-weight:500; color:#64748B;'>carbs, last 2 h</span>",
         regimen_line),
    ]
    for col, lab, big, sub in vit:
        with col:
            st.markdown(f"<div class='gt-card'><div class='gt-label'>{lab}</div><div class='gt-big'>{big}</div>"
                        f"<div class='gt-sub'>{sub}</div></div>", unsafe_allow_html=True)

    d_btn1, d_btn2, _ = st.columns([1, 1, 3])
    with d_btn1:
        if st.button("Nudge patient (Demo)", key="demo_nudge"):
            st.toast("Demo only: no message was sent.")
    with d_btn2:
        if st.button("Log to EHR (Demo)", key="demo_log"):
            st.toast("Demo only: nothing was written to any EHR.")

    # ---- 4. Risk-score history and the week's alerts ------------------------------
    if have_pred:
        ph = preds.iloc[max(0, i - HISTORY_STEPS):i + 1]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=hist["timestamp"], y=ph["p_hypo"], mode="lines", name=f"hypo <{HYPO_MG_DL:.0f} score",
                                  line=dict(color=RED, width=2), hovertemplate="%{y:.2f}<extra>hypo score</extra>"))
        fig2.add_trace(go.Scatter(x=hist["timestamp"], y=ph["p_hyper"], mode="lines", name=f"hyper >{HYPER_MG_DL:.0f} score",
                                  line=dict(color=AMBER, width=2), hovertemplate="%{y:.2f}<extra>hyper score</extra>"))
        for name, col in (("hypo", RED), ("hyper", AMBER)):
            fig2.add_hline(y=THRESHOLDS[name], line=dict(color=col, width=1, dash="dash"),
                           annotation_text=f"{name} alert threshold {THRESHOLDS[name]:.2f}",
                           annotation_position="top left", annotation_font=dict(size=10, color=SLATE))
        add_now_line(fig2, t_now)
        fig2.update_layout(**plot_layout("Model risk score over the last 8 h (uncalibrated)", height=250, y_range=[0, 1.05]))
        st.plotly_chart(fig2, width="stretch", config={"displayModeBar": False})

        runs = alert_runs(preds["alert"].to_numpy())
        with st.expander(f"This week's model alerts for {selected_pid} ({len(runs)})"):
            if not runs:
                st.write("The model raised no alert for this patient during the 7-day record.")
            else:
                rows = []
                for s, e, k in runs:
                    score_col = "p_hypo" if k == 1 else "p_hyper"
                    r = {"Start": pw["timestamp"].iloc[s].strftime("%a %d %b %H:%M"),
                         "Alert": "Hypoglycaemia risk" if k == 1 else "Marked hyperglycaemia risk",
                         "Duration (min)": (e - s + 1) * STEP_MIN,
                         "Peak score": round(float(preds[score_col].iloc[s:e + 1].max()), 2)}
                    if held_out:
                        r["Event followed? (hidden label at alert start)"] = (
                            "yes" if int(pw["adverse_event_2h"].iloc[s]) == k else "no (false alarm)")
                    rows.append(r)
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                st.caption("Use ◀ Prev alert / Next alert ▶ above to step through them."
                           + ("" if held_out else " Outcome column shown only for held-out patients."))

# ==============================================================================
# VIEW 2: WHAT-IF SANDBOX (rule-based; deliberately separate from the ML model)
# ==============================================================================
elif nav_tab == "What-If Sandbox":
    if not n_steps:
        no_stream_box()
        st.stop()
    page_title("What-If sandbox (rule-based, illustrative — not the ML model, not a dosing tool)",
               "A small hand-written physiology rule set for discussing lifestyle scenarios. It starts from the "
               "current CGM value at the replay time and uses only BMI and fasting glucose from the EHR. It ignores "
               "HR/HRV, sleep, medication and recent history, so it can disagree with the ML forecast on the Live "
               "Monitor; the ML model's output for the same window is shown alongside for comparison. The "
               "standalone web page has its own, different sandbox rules. Nothing here is a recommendation.")
    i = render_clock(show_alert_nav=False, caption="Sandbox start point: the CGM reading at this replay time.")
    row = pw.iloc[i]
    g0 = float(row["glucose_cgm"])
    bmi = float(pe["bmi"])
    fasting = float(pe["fasting_glucose_mg_dl"])

    # The ML model's output at this window, shown next to the sandbox so a reassuring rule-based curve is
    # never read on its own while the model is alerting.
    ml_alert, ml_fc = None, None
    if preds is not None:
        mp = preds.iloc[i]
        ml_alert, ml_fc = int(mp["alert"]), float(mp["forecast_2h"])
        ml_hypo, ml_hyper = float(mp["p_hypo"]), float(mp["p_hyper"])
        ml_state = {0: "no alert", 1: "hypoglycaemia risk alert", 2: "marked hyperglycaemia risk alert"}[ml_alert]
        ml_bg, ml_bd, ml_fg = {0: ("#F8FAFC", "#CBD5E1", "#334155"), 1: ("#FEF2F2", RED, "#7F1D1D"),
                               2: ("#FFFBEB", AMBER, "#78350F")}[ml_alert]
        st.markdown(f"""
        <div class="gt-card" style="border-left:5px solid {ml_bd}; background:{ml_bg}; padding:9px 14px; font-size:12.5px; color:{ml_fg};">
            <b>ML model at this window ({times[i].strftime('%a %d %b %H:%M')}): {ml_state}</b> ·
            hypo score <b class="tabular-nums">{ml_hypo:.2f}</b> (alert ≥ {THRESHOLDS['hypo']:.2f}) ·
            hyper score <b class="tabular-nums">{ml_hyper:.2f}</b> (alert ≥ {THRESHOLDS['hyper']:.2f}) ·
            forecast glucose at +2 h <b class="tabular-nums">{ml_fc:.0f} mg/dL</b>
            <span style="color:#64748B;"> · model risk scores are uncalibrated. The sandbox curves below do not use the ML model.</span>
        </div>""", unsafe_allow_html=True)
    elif bundle is None:
        st.caption(f"ML model unavailable ({MODEL_ERROR}); the rule-based sandbox still runs, with no ML comparison.")
    else:
        st.caption(f"ML model could not score this patient ({PRED_ERROR}); no ML comparison is shown.")

    # Illustrative rule parameters (not fitted to data, not personalised).
    CARB_MG_PER_G = 1.8          # rise per gram, scaled by BMI/22, absorbed over 15-75 min
    WALK_MG_PER_MIN = 0.7        # fall per minute of brisk walking
    ISF_MG_PER_U = 30.0          # illustrative fall per unit of rapid-acting insulin, spread over 30-150 min
    REVERSION = 0.05             # 5% pull towards fasting glucose every 15 min

    left, right = st.columns([1.1, 1.9])
    with left:
        with st.container(border=True):
            st.markdown("<p class='gt-label'>Scenario</p>", unsafe_allow_html=True)
            carbs = st.slider("Extra meal carbohydrate (g)", 0, 120, 50, step=5,
                              help="About 50 g is two rotis with dal and sabzi.")
            walk = st.slider("Brisk walk after the meal (min)", 0, 90, 0, step=5,
                             help="Starts 15 minutes after the meal begins.")
            if on_insulin:
                units = st.slider("Rapid-acting insulin in scenario (U), illustrative only", 0, 15, 0, step=1,
                                  help="Shows the shape of an insulin effect in the sandbox. Not dose advice.")
            else:
                units = 0
                st.caption("Insulin control hidden: this patient is not on insulin.")
            hours = st.selectbox("Horizon", [2, 4, 6], index=1, format_func=lambda h: f"{h} h")
            with st.expander("Rule assumptions"):
                st.markdown(
                    f"- Meal: +{CARB_MG_PER_G} mg/dL per g × BMI/22, absorbed evenly over the first hour\n"
                    f"- Walk: −{WALK_MG_PER_MIN} mg/dL per minute, starting at +15 min "
                    f"(a 10-min walk counts; intervals = ceil(minutes/15))\n"
                    + (f"- Insulin: −{ISF_MG_PER_U:.0f} mg/dL per U spread over 30–150 min\n" if on_insulin else "")
                    + f"- Every 15 min glucose relaxes {REVERSION:.0%} towards fasting glucose ({fasting:.0f} mg/dL)\n"
                    "- No oral-drug effect (a sulfonylurea-driven low, for example, is not modelled here)\n"
                    "- Parameters are illustrative, not fitted to any dataset\n"
                    "- The standalone web page uses a different rule set (it starts from the ML +2 h forecast and "
                    "has a sulfonylurea/insulin flag), so the two sandboxes can give different curves")

    n_sim = hours * 60 // STEP_MIN
    walk_intervals = math.ceil(walk / STEP_MIN)
    base, sim = [g0], [g0]
    gb, gs = g0, g0
    for j in range(1, n_sim + 1):
        carb_flux = carbs * CARB_MG_PER_G * (bmi / 22.0) / 4.0 if 1 <= j <= 4 else 0.0
        walk_min = min(STEP_MIN, max(0, walk - STEP_MIN * (j - 2))) if 2 <= j <= 1 + walk_intervals else 0
        ex_burn = walk_min * WALK_MG_PER_MIN
        ins_effect = units * ISF_MG_PER_U / 8.0 if 3 <= j <= 10 else 0.0
        gb = gb + (fasting - gb) * REVERSION
        gs = gs + carb_flux - ex_burn - ins_effect + (fasting - gs) * REVERSION
        gb, gs = float(np.clip(gb, 40, 400)), float(np.clip(gs, 40, 400))
        base.append(round(gb, 1))
        sim.append(round(gs, 1))

    labels = [f"+{STEP_MIN * j // 60} h" if (STEP_MIN * j) % 60 == 0 else f"+{STEP_MIN * j // 60}h{STEP_MIN * j % 60:02d}"
              for j in range(n_sim + 1)]

    def tir(curve):
        return sum(HYPO_MG_DL <= g <= TARGET_HIGH_MG_DL for g in curve) / len(curve) * 100.0

    s_tir, b_tir = tir(sim), tir(base)
    s_peak, s_min = max(sim), min(sim)
    with right:
        d1, d2, d3 = st.columns(3)
        with d1:
            col = GREEN if s_tir >= b_tir else RED
            st.markdown(f"""<div class="gt-card"><div class="gt-label">Time in {HYPO_MG_DL:.0f}–{TARGET_HIGH_MG_DL:.0f} (simulated)</div>
                <div class="gt-big" style="color:{col};">{s_tir:.0f}%</div>
                <div class="gt-sub">no-change curve {b_tir:.0f}% · Δ {s_tir - b_tir:+.0f} pp</div></div>""", unsafe_allow_html=True)
        with d2:
            st.markdown(f"""<div class="gt-card"><div class="gt-label">Simulated peak</div>
                <div class="gt-big" style="color:{glucose_colour(s_peak)};">{s_peak:.0f} <span style="font-size:12px; color:#64748B;">mg/dL</span></div>
                <div class="gt-sub">at {labels[sim.index(s_peak)]} · Δ vs no change {s_peak - max(base):+.0f}</div></div>""", unsafe_allow_html=True)
        with d3:
            st.markdown(f"""<div class="gt-card"><div class="gt-label">Simulated lowest</div>
                <div class="gt-big" style="color:{glucose_colour(s_min)};">{s_min:.0f} <span style="font-size:12px; color:#64748B;">mg/dL</span></div>
                <div class="gt-sub">{glucose_state(s_min)[0]}</div></div>""", unsafe_allow_html=True)

        sim_lo = min(40, s_min - 10, (ml_fc - 10) if ml_fc is not None else 40)
        sim_hi = max(320, s_peak + 20, (ml_fc + 20) if ml_fc is not None else 320)
        fig_sim = go.Figure()
        add_glucose_bands(fig_sim, sim_hi, sim_lo, compact=True)
        fig_sim.add_trace(go.Scatter(x=labels, y=base, name="no extra meal or activity (rule-based)",
                                     line=dict(color=MUTED, width=2, dash="dot"), mode="lines"))
        fig_sim.add_trace(go.Scatter(x=labels, y=sim, name="scenario (rule-based)", mode="lines+markers",
                                     line=dict(color=TEAL, width=2.5), marker=dict(size=6)))
        fc_label = labels[HORIZON_STEPS] if len(labels) > HORIZON_STEPS else None
        if ml_fc is not None and fc_label is not None:
            fig_sim.add_trace(go.Scatter(
                x=[fc_label], y=[ml_fc], mode="markers", name="ML model forecast at +2 h (no scenario; for comparison)",
                marker=dict(size=13, symbol="diamond", color=NAVY, line=dict(color="#FFFFFF", width=2)),
                hovertemplate="%{y:.0f} mg/dL<extra>ML forecast +2 h</extra>"))
        title = f"Rule-based sandbox: {carbs} g meal, {walk} min walk" + (f", {units} U insulin" if on_insulin else "")
        fig_sim.update_layout(**plot_layout(title, height=360, y_range=[sim_lo, sim_hi]))
        fig_sim.update_layout(hovermode="x unified")
        fig_sim.update_yaxes(title_text="mg/dL")
        fig_sim.update_xaxes(tickmode="array", tickvals=labels[::4], tickangle=0)
        st.plotly_chart(fig_sim, width="stretch", config={"displayModeBar": False})

        # The ML model's alert outranks a reassuring rule-based curve: never show green while it is alerting.
        ml_note = ""
        if ml_alert == 1:
            ml_note = (f"<b>The ML model flags hypoglycaemia risk within 2 h at this time</b> (hypo score "
                       f"{ml_hypo:.2f}, alert ≥ {THRESHOLDS['hypo']:.2f}; forecast {ml_fc:.0f} mg/dL at +2 h). These "
                       "rules ignore medication and recent history, so an in-range sandbox curve is not reassurance. "
                       + ("Consider a snack and review medication timing with the care team." if g0 >= HYPO_MG_DL else
                          f"CGM is {g0:.0f} mg/dL now: follow the hypoglycaemia protocol shown on the Live Monitor."))
        elif ml_alert == 2:
            ml_note = (f"<b>The ML model flags marked hyperglycaemia risk within 2 h at this time</b> (hyper score "
                       f"{ml_hyper:.2f}, alert ≥ {THRESHOLDS['hyper']:.2f}; forecast {ml_fc:.0f} mg/dL at +2 h). These "
                       "rules ignore recent history, so an in-range sandbox curve is not reassurance. Review with the "
                       "care team.")
        parts = [ml_note] if ml_note else []
        if s_min < HYPO_MG_DL:
            note_bg, note_bd, note_fg = "#FEF2F2", RED, "#7F1D1D"
            parts.append(f"Simulated lowest value {s_min:.0f} mg/dL (below {HYPO_MG_DL:.0f}) at "
                         f"{labels[sim.index(s_min)]}. Review this scenario with the care team.")
        elif ml_alert:
            note_bg, note_bd, note_fg = ("#FEF2F2", RED, "#7F1D1D") if ml_alert == 1 else ("#FFFBEB", AMBER, "#78350F")
            if s_peak > TARGET_HIGH_MG_DL:
                parts.append(f"Simulated peak {s_peak:.0f} mg/dL is above the {TARGET_HIGH_MG_DL:.0f} mg/dL target "
                             f"at {labels[sim.index(s_peak)]}.")
        elif s_peak > TARGET_HIGH_MG_DL:
            note_bg, note_bd, note_fg = "#FFFBEB", AMBER, "#78350F"
            parts.append(f"Simulated peak {s_peak:.0f} mg/dL is above the {TARGET_HIGH_MG_DL:.0f} mg/dL target at "
                         f"{labels[sim.index(s_peak)]}. Options to discuss with the care team: a smaller carbohydrate "
                         "portion or a walk after the meal.")
        else:
            note_bg, note_bd, note_fg = "#F0FDF4", GREEN, "#064E3B"
            parts.append(f"Simulated curve stays within {HYPO_MG_DL:.0f}–{TARGET_HIGH_MG_DL:.0f} mg/dL "
                         f"({s_min:.0f}–{s_peak:.0f}) over {hours} h"
                         + (", and the ML model raises no alert at this window." if ml_alert == 0 else "."))
        note = " ".join(parts)
        st.markdown(f"""
        <div class="gt-card" style="border-left:5px solid {note_bd}; background:{note_bg};">
            <div style="font-size:13px; font-weight:800; color:{note_fg};">{'ML model alert at this window' if ml_alert else 'Rule-based note'} (not a recommendation)</div>
            <div style="font-size:12.5px; color:{note_fg}; margin-top:4px;">{note}</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Save scenario to care plan (Demo)", key="demo_save"):
            st.toast("Demo only: nothing was saved.")


# ==============================================================================
# VIEW 3: DATA INGEST (parse + 5 validation checks + optional model forecast)
# ==============================================================================
elif nav_tab == "Data Ingest":
    page_title("Wearable data ingest and validation",
               "Upload a CGM or wearable export (CSV or JSON). The file is parsed and run through five checks: "
               "schema, continuity, physiological bounds, rate of change and missingness. If it also carries the "
               "full GlycoTwin wearable schema, the trained model scores its latest window.")
    st.markdown("""
    <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-bottom:14px;">
        <span style="font-size:12px; font-weight:700; color:#64748B;">Reads:</span>
        <span style="background:#E0F2FE; color:#0369A1; font-size:11px; font-weight:700; padding:4px 10px; border-radius:12px;">Generic CGM CSV (timestamp + glucose)</span>
        <span style="background:#E0F2FE; color:#0369A1; font-size:11px; font-weight:700; padding:4px 10px; border-radius:12px;">GlycoTwin wearable schema</span>
        <span style="background:#E0F2FE; color:#0369A1; font-size:11px; font-weight:700; padding:4px 10px; border-radius:12px;">CSV with one metadata line on top (LibreView-style)</span>
        <span style="background:#F1F5F9; color:#64748B; font-size:11px; font-weight:700; padding:4px 10px; border-radius:12px;">Roadmap, not built: native Dexcom / LibreView / Health Connect parsers</span>
    </div>""", unsafe_allow_html=True)

    def detect_columns(df):
        names = {c: str(c).strip().lower() for c in df.columns}

        def find(exact, contains, exclude=(), forbid=()):
            for e in exact:
                for c in df.columns:
                    if names[c] == e and c not in exclude:
                        return c
            for sub in contains:
                for c in df.columns:
                    if sub in names[c] and c not in exclude and not any(f in names[c] for f in forbid):
                        return c
            return None

        ts_c = find(["timestamp", "datetime", "device timestamp", "time", "date"], ["timestamp", "time", "date"],
                    forbid=("glucose", "cgm"))
        gl_c = find(["glucose_cgm", "glucose", "sgv", "cgm", "historic glucose mg/dl", "glucose value (mg/dl)"],
                    ["glucose", "cgm", "sgv"], exclude=(ts_c,), forbid=("time", "date"))
        return ts_c, gl_c

    def parse_upload(name, raw):
        notes = []
        if not raw:
            raise ValueError("the file is empty")
        if name.lower().endswith(".json"):
            try:
                df = pd.read_json(io.BytesIO(raw))
            except ValueError:
                df = pd.json_normalize(json.loads(raw.decode("utf-8-sig")))
            return df, notes
        text = raw.decode("utf-8-sig", errors="replace")
        df = pd.read_csv(io.StringIO(text))
        if detect_columns(df)[0] is None and len(text.splitlines()) > 2:
            try:
                df2 = pd.read_csv(io.StringIO(text), skiprows=1)
                if detect_columns(df2)[0] is not None:
                    df = df2
                    notes.append("Skipped one metadata line at the top of the file (LibreView-style export).")
            except Exception:
                pass
        return df, notes

    def validate(df):
        """Returns (checks, frame, info). checks: list of (name, status, detail)."""
        checks = []
        ts_c, gl_c = detect_columns(df)
        info = {"ts_col": ts_c, "gl_col": gl_c, "full_schema": all(c in df.columns for c in WEARABLE_INPUT_COLS),
                "has_pid": "patient_id" in df.columns, "notes": []}
        if df.empty or ts_c is None or gl_c is None:
            missing = ([] if ts_c else ["timestamp"]) + ([] if gl_c else ["glucose"])
            cols = ", ".join(map(str, list(df.columns)[:10])) or "none"
            checks.append(("1. Schema", "FAIL", f"Missing a distinct {' and '.join(missing) or 'data'} column. "
                                                f"Columns found: {esc(cols)}."))
            for nm in ("2. Continuity", "3. Physiological bounds", "4. Rate of change", "5. Missingness"):
                checks.append((nm, "SKIPPED", "Not run because the schema check failed."))
            return checks, None, info
        checks.append(("1. Schema", "PASS",
                       f"Timestamp column '{esc(ts_c)}', glucose column '{esc(gl_c)}'. "
                       + ("Full GlycoTwin wearable schema present, so the model can score it."
                          if info["full_schema"] else
                          "Glucose-only file: no HR/HRV/steps/sleep/carbs/insulin, so no model forecast.")))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ts = pd.to_datetime(df[ts_c], errors="coerce")
        glu = pd.to_numeric(df[gl_c], errors="coerce")
        if "mmol" in str(gl_c).lower():
            glu = glu * 18.016
            info["notes"].append("Glucose column looks like mmol/L; converted to mg/dL (× 18.016).")
        pid = df["patient_id"].astype(str) if info["has_pid"] else pd.Series("upload", index=df.index)
        frame = pd.DataFrame({"pid": pid.to_numpy(), "ts": ts.to_numpy(), "g": glu.to_numpy()})
        n = len(frame)

        # 2. Continuity
        bad_ts = int(frame["ts"].isna().sum())
        valid = frame.dropna(subset=["ts"])
        unsorted = int(sum(not g["ts"].is_monotonic_increasing for _, g in valid.groupby("pid", sort=False)))
        dups = int(valid.duplicated(subset=["pid", "ts"]).sum())
        srt = valid.sort_values(["pid", "ts"])
        gaps = srt.groupby("pid")["ts"].diff().dropna()
        gaps = gaps[gaps > pd.Timedelta(0)]
        if len(gaps) == 0:
            checks.append(("2. Continuity", "FAIL", f"Fewer than two distinct parsable timestamps ({bad_ts} of {n} unparsable)."))
        else:
            ok_share = float((gaps <= pd.Timedelta(minutes=30)).mean())
            med_min = gaps.median().total_seconds() / 60
            if bad_ts / n > 0.05 or ok_share < 0.80:
                status = "FAIL"
            elif bad_ts or dups or unsorted or ok_share < 0.95:
                status = "WARN"
            else:
                status = "PASS"
            n_pid = valid["pid"].nunique()
            per = f" across {n_pid} patients" if info["has_pid"] and n_pid > 1 else ""
            cadence_txt = ""
            if info["full_schema"] and abs(med_min - STEP_MIN) > 1:
                cadence_txt = (f" The model scores {STEP_MIN}-min windows: finer data is aggregated into {STEP_MIN}-min "
                               "windows before scoring; coarser data is validated only.")
            checks.append(("2. Continuity", status,
                           f"Median interval {med_min:.0f} min{per}; {ok_share:.0%} of intervals ≤30 min; "
                           f"{dups} duplicate timestamps; {bad_ts} unparsable; "
                           + ("in time order." if not unsorted else f"{unsorted} series out of order (sorted for analysis).")
                           + cadence_txt))

        # 3. Physiological bounds
        n_valid = int(glu.notna().sum())
        if n_valid == 0:
            checks.append(("3. Physiological bounds", "FAIL", "No numeric glucose values."))
        else:
            oob = int(((glu < GLUCOSE_BOUNDS[0]) | (glu > GLUCOSE_BOUNDS[1])).sum())
            extra, extra_bad = [], 0
            for c, (lo, hi) in OTHER_BOUNDS.items():
                if c in df.columns:
                    v = pd.to_numeric(df[c], errors="coerce")
                    k = int(((v < lo) | (v > hi)).sum())
                    extra_bad += k
                    if k:
                        extra.append(f"{c}: {k}")
            share = (oob + extra_bad) / n_valid
            status = "PASS" if oob + extra_bad == 0 else ("WARN" if share <= 0.01 else "FAIL")
            checks.append(("3. Physiological bounds", status,
                           f"{oob} of {n_valid} glucose values outside {GLUCOSE_BOUNDS[0]:.0f}–{GLUCOSE_BOUNDS[1]:.0f} mg/dL"
                           + (f"; other signals out of range ({', '.join(extra)})" if extra else "") + "."))

        # 4. Rate of change
        r = srt.dropna(subset=["g"]).copy()
        r["dg"] = r.groupby("pid")["g"].diff()
        r["dt"] = r.groupby("pid")["ts"].diff().dt.total_seconds() / 60
        r = r[(r["dt"] > 0) & r["dg"].notna()]
        if len(r) == 0:
            checks.append(("4. Rate of change", "FAIL", "Not enough consecutive numeric readings to check."))
        else:
            rate = (r["dg"] / r["dt"]).abs()
            k = int((rate > MAX_ROC_MG_DL_PER_MIN).sum())
            share = k / len(rate)
            status = "PASS" if k == 0 else ("WARN" if share <= 0.02 else "FAIL")
            checks.append(("4. Rate of change", status,
                           f"Largest change {rate.max():.1f} mg/dL/min; {k} of {len(rate)} intervals exceed "
                           f"{MAX_ROC_MG_DL_PER_MIN:.0f} mg/dL/min (likely sensor artefact)."))

        # 5. Missingness
        miss = n - n_valid
        other_miss = []
        if info["full_schema"]:
            for c in WEARABLE_INPUT_COLS[2:]:
                k = int(df[c].isna().sum() + (df[c].astype(str).str.strip() == "").sum())
                if k:
                    other_miss.append(f"{c}: {k}")
        share = miss / n if n else 1.0
        status = "PASS" if miss == 0 and not other_miss else ("WARN" if share <= 0.05 else "FAIL")
        checks.append(("5. Missingness", status,
                       f"{miss} of {n} glucose values missing or non-numeric ({share:.1%})"
                       + (f"; other columns missing ({', '.join(other_miss)})" if other_miss else "") + "."))
        return checks, frame, info

    up_col1, up_col2 = st.columns([1.5, 1])
    with up_col1:
        uploaded = st.file_uploader("Upload a wearable / CGM export", type=["csv", "json"],
                                    help="Needs a timestamp column and a glucose column (mg/dL, or mmol/L if the column name says so).")
    with up_col2:
        st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)
        if st.button(f"Load sample: first 48 readings of {selected_pid}", key="load_sample", width="stretch"):
            st.session_state["ingest_sample"] = True
        st.caption("Sample uses the GlycoTwin synthetic schema (no label columns).")

    sample_cols = ["patient_id"] + WEARABLE_INPUT_COLS
    active_df, source, notes = None, None, []
    if uploaded is not None:
        try:
            active_df, notes = parse_upload(uploaded.name, uploaded.getvalue())
            source = f"uploaded file '{esc(uploaded.name)}'"
        except Exception as exc:
            st.error(f"File rejected: could not parse '{uploaded.name}' ({type(exc).__name__}: {exc}).")
    elif st.session_state.get("ingest_sample") and n_steps:
        sample = df_wear[df_wear["patient_id"] == selected_pid].head(48)[sample_cols].copy()
        sample["timestamp"] = sample["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        active_df, source = sample.reset_index(drop=True), f"sample: first 48 readings of {selected_pid}"
        st.download_button("Download this sample as CSV", sample.to_csv(index=False).encode(),
                           file_name=f"glycotwin_sample_{selected_pid}.csv", mime="text/csv")

    if active_df is not None:
        checks, frame, info = validate(active_df)
        notes = notes + info["notes"]
        statuses = [s for _, s, _ in checks]
        if "FAIL" in statuses:
            st.error(f"Rejected: {statuses.count('FAIL')} of 5 checks failed for the {source}. Fix the file and re-upload.")
        elif "WARN" in statuses:
            st.warning(f"Accepted with warnings: {source} passed with {statuses.count('WARN')} warning(s).")
        else:
            st.success(f"Accepted: {source} passed all 5 checks.")
        for note in notes:
            st.info(note)

        left_c, right_c = st.columns([1.35, 1], gap="medium")
        with left_c:
            with st.container(border=True):
                st.markdown("<p style='font-size:13px; font-weight:700; color:#0F172A; margin-bottom:8px;'>Data quality checks (computed on this file)</p>", unsafe_allow_html=True)
                st.markdown("".join(
                    f"<div class='check-step'><div><b>{nm}</b><br><span style='font-size:11.5px; color:#64748B;'>{detail}</span></div>"
                    f"<div>{badge(status)}</div></div>" for nm, status, detail in checks), unsafe_allow_html=True)
        with right_c:
            if frame is not None:
                g = frame["g"].dropna()
                ts_ok = frame["ts"].dropna()
                tiles = [
                    ("Rows", f"{len(frame)}", f"{len(g)} numeric glucose values"),
                    ("Time span", (f"{(ts_ok.max() - ts_ok.min()).total_seconds() / 3600:.1f} h" if len(ts_ok) > 1 else "n/a"),
                     (f"{ts_ok.min():%d %b %H:%M} → {ts_ok.max():%d %b %H:%M}" if len(ts_ok) > 1 else "")),
                    ("Mean glucose", (f"{g.mean():.1f} mg/dL" if len(g) else "n/a"), "valid readings only"),
                    (f"Time in {HYPO_MG_DL:.0f}–{TARGET_HIGH_MG_DL:.0f}",
                     (f"{g.between(HYPO_MG_DL, TARGET_HIGH_MG_DL).mean() * 100:.0f}%" if len(g) else "n/a"),
                     (f"{(g < HYPO_MG_DL).mean() * 100:.0f}% below {HYPO_MG_DL:.0f}" if len(g) else "")),
                ]
                st.markdown("<div style='display:grid; grid-template-columns:1fr 1fr; gap:10px;'>" + "".join(
                    f"<div class='gt-card' style='margin-bottom:0;'><div class='gt-label'>{lab}</div>"
                    f"<div class='gt-big' style='font-size:21px;'>{val}</div><div class='gt-sub'>{sub}</div></div>"
                    for lab, val, sub in tiles) + "</div><div style='height:12px;'></div>", unsafe_allow_html=True)
            # Model forecast on the ingested stream (only when the file passes and carries the full schema)
            if frame is not None and "FAIL" not in statuses:
                if not info["full_schema"]:
                    st.info("Model forecast not run: the model needs " + ", ".join(WEARABLE_INPUT_COLS[2:])
                            + " in addition to timestamp and glucose.")
                elif bundle is None:
                    model_error_box("The ingest forecast")
                else:
                    try:
                        up = active_df.copy()
                        note = ""
                        if info["has_pid"]:
                            up_pid = str(up["patient_id"].astype(str).iloc[-1])
                            if up["patient_id"].astype(str).nunique() > 1:
                                note += f"File holds several patients; scoring the last one ({esc(up_pid)}). "
                            up = up[up["patient_id"].astype(str) == up_pid]
                        else:
                            up_pid = None
                        ehr_pid = up_pid if up_pid in ehr_by_pid.index else selected_pid
                        if ehr_pid != up_pid:
                            note += f"Fused with the selected patient's EHR ({esc(selected_pid)}). "
                        up = up[WEARABLE_INPUT_COLS].copy()
                        up["patient_id"] = ehr_pid
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            up["timestamp"] = pd.to_datetime(up["timestamp"], errors="coerce")
                        up = up.dropna(subset=["timestamp"])
                        for c in WEARABLE_INPUT_COLS[1:]:
                            if c != "sleep_stage":
                                up[c] = pd.to_numeric(up[c], errors="coerce")
                        win, cad_note, problem = to_model_windows(up)
                        if problem:
                            st.warning(f"Validation only, no model forecast: {problem}")
                        else:
                            win["patient_id"] = ehr_pid
                            feats = compute_wearable_features(win)
                            fused_up, _ = prepare_fused_dataset(df_ehr[df_ehr["patient_id"] == ehr_pid], feats)
                            last = fused_up.sort_values("timestamp").tail(1)
                            X = encode_features(last, bundle["feature_cols"])
                            proba = bundle["classifier"].predict_proba(X)
                            k = int(apply_alert_thresholds(proba, THRESHOLDS)[0])
                            fc_u = float(bundle["regressor"].predict(X)[0])
                            t_last = last["timestamp"].iloc[0]
                            g_last = float(last["glucose_cgm"].iloc[0])
                            model_state = {0: "No event predicted within 2 h", 1: "Hypoglycaemia risk within 2 h",
                                           2: "Marked hyperglycaemia risk within 2 h"}[k]
                            # As on the Live Monitor, a reading already out of range outranks the model output.
                            if g_last < HYPO_MG_DL:
                                lvl = "Level 2" if g_last < LEVEL2_HYPO_MG_DL else "Level 1"
                                state, css = f"Hypoglycaemia now ({lvl})", "forecast-hypo"
                                state_sub = f"Model, next 2 h: {model_state.lower()}."
                            elif g_last > HYPER_MG_DL:
                                state, css = "Marked hyperglycaemia now", "forecast-hyper"
                                state_sub = f"Model, next 2 h: {model_state.lower()}."
                            else:
                                state = model_state
                                css = {0: "forecast-safe", 1: "forecast-hypo", 2: "forecast-hyper"}[k]
                                state_sub = ""
                            # Keep the HTML free of blank lines: a blank line ends Markdown's HTML block and the
                            # indented remainder would render as a code block.
                            sub_html = (f'<div style="font-size:12.5px; color:#334155;">{state_sub}</div>'
                                        if state_sub else "")
                            st.markdown(f"""
                            <div class="{css}">
                                <div class="gt-label">Model on the ingested stream · latest 15-min window {t_last:%d %b %H:%M}</div>
                                <div style="font-size:19px; font-weight:800; color:#0F172A; margin-top:2px;">{state}</div>{sub_html}
                                <div style="font-size:13px; color:#334155; margin-top:4px;">
                                    CGM {g_last:.0f} mg/dL · forecast glucose at +2 h <b>{fc_u:.0f} mg/dL</b> ·
                                    model risk score (uncalibrated): hypo {proba[0, 1]:.2f} (alert ≥ {THRESHOLDS['hypo']:.2f}),
                                    hyper {proba[0, 2]:.2f} (alert ≥ {THRESHOLDS['hyper']:.2f})</div>
                                <div style="font-size:11px; color:#64748B; margin-top:4px;">{note}{cad_note}Features built with
                                    compute_wearable_features → prepare_fused_dataset → encode_features, as in training.</div>
                            </div>""", unsafe_allow_html=True)
                    except Exception as exc:
                        st.error(f"Model could not score this file ({type(exc).__name__}: {exc}).")

        if frame is not None and len(frame["g"].dropna()) > 1 and frame["ts"].notna().sum() > 1:
            pf = frame.dropna(subset=["ts", "g"]).sort_values(["pid", "ts"])
            up_lo, up_hi = min(40, pf["g"].min() - 10), max(320, pf["g"].max() + 20)
            fig_u = go.Figure()
            add_glucose_bands(fig_u, up_hi, up_lo, compact=True)
            for k, (p_id, gdf) in enumerate(pf.groupby("pid", sort=False)):
                if k >= 5:
                    break
                fig_u.add_trace(go.Scatter(x=gdf["ts"], y=gdf["g"], mode="lines+markers", name=str(p_id),
                                           line=dict(width=2), marker=dict(size=6)))
            fig_u.update_layout(**plot_layout("Ingested glucose", height=300, y_range=[up_lo, up_hi]))
            st.plotly_chart(fig_u, width="stretch", config={"displayModeBar": False})

        st.markdown("<p class='gt-label' style='margin-top:10px;'>Preview (first 10 rows)</p>", unsafe_allow_html=True)
        st.dataframe(active_df.head(10), width="stretch", hide_index=True)
    elif uploaded is None:
        st.info("Upload a CSV/JSON file, or load the 48-reading sample of the selected patient.")


# ==============================================================================
# VIEW 4: COHORT & 2-HOUR RISK BOARD
# ==============================================================================
elif nav_tab == "Cohort & Risk Board":
    page_title("Cohort overview and 2-hour risk board",
               "Every number on this page is computed from data/synthetic_*.csv and the trained model at runtime.")

    n_cohort = len(df_ehr)
    hba1c = pd.to_numeric(df_ehr["baseline_hba1c"], errors="coerce")
    tiers = df_ehr["clinical_risk_tier"].astype(str)
    n_high = int(tiers.str.contains("High").sum())
    cgm_all = df_wear["glucose_cgm"]
    k1, k2, k3, k4 = st.columns(4)
    tiles = [
        ("Synthetic personas", f"{n_cohort}",
         f"{len(wear_pids)} with 7-day wearable streams" + (f" · {len(TEST_PATIENTS)} held-out" if bundle else "")),
        ("Mean baseline HbA1c", f"{hba1c.mean():.2f}%", f"{(hba1c >= 7.0).mean() * 100:.0f}% at or above 7.0%"),
        ("High clinical risk tier", f"{n_high} ({n_high / n_cohort * 100:.0f}%)",
         f"{int(tiers.str.contains('Moderate').sum())} moderate · {int(tiers.str.contains('Low').sum())} low"),
        (f"Cohort CGM time in {HYPO_MG_DL:.0f}–{TARGET_HIGH_MG_DL:.0f}",
         f"{cgm_all.between(HYPO_MG_DL, TARGET_HIGH_MG_DL).mean() * 100:.1f}%",
         f"{(cgm_all < HYPO_MG_DL).mean() * 100:.2f}% of readings <{HYPO_MG_DL:.0f} · "
         f"{(cgm_all > HYPER_MG_DL).mean() * 100:.2f}% >{HYPER_MG_DL:.0f}"),
    ]
    for col, (lab, val, sub) in zip((k1, k2, k3, k4), tiles):
        with col:
            st.markdown(f"<div class='gt-card'><div class='gt-label'>{lab}</div><div class='gt-big'>{val}</div>"
                        f"<div class='gt-sub'>{sub}</div></div>", unsafe_allow_html=True)

    st.markdown("<p style='font-size:15px; font-weight:800; color:#0F172A; margin:8px 0 2px 0;'>2-hour risk board</p>", unsafe_allow_html=True)
    if bundle is None:
        model_error_box("The risk board")
    elif not n_steps:
        no_stream_box()
    else:
        i = render_clock(show_alert_nav=False,
                         caption="Board time uses the same replay clock as the Live Monitor. Each patient is scored "
                                 "on their latest window at or before this time, in one batched model call.")
        as_of = times[i]
        try:
            board = risk_board(as_of.isoformat(), DATA_KEY, MODEL_KEY)
        except Exception as exc:
            board = None
            st.error(f"Risk board unavailable ({type(exc).__name__}: {exc}).")
        if board is not None:
            n_hypo, n_hyper = int((board["alert"] == 1).sum()), int((board["alert"] == 2).sum())
            n_low_now = int((board["cgm"] < HYPO_MG_DL).sum())
            n_high_now = int((board["cgm"] > HYPER_MG_DL).sum())
            st.markdown(f"""
            <div style="display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin:2px 0 8px 0;">
                {pill(f"{n_hypo} hypoglycaemia-risk alerts", "red" if n_hypo else "green")}
                {pill(f"{n_hyper} marked-hyperglycaemia-risk alerts", "amber" if n_hyper else "green")}
                {pill(f"CGM now: {n_low_now} below {HYPO_MG_DL:.0f} · {n_high_now} above {HYPER_MG_DL:.0f}",
                      "red" if n_low_now else ("amber" if n_high_now else "grey"))}
                <span style="font-size:12px; color:#64748B;">of {len(board)} patients at {as_of:%a %d %b %H:%M} ·
                sorted by the higher of the two model risk scores (uncalibrated) · alert thresholds hypo ≥ {THRESHOLDS['hypo']:.2f},
                hyper ≥ {THRESHOLDS['hyper']:.2f}</span>
            </div>""", unsafe_allow_html=True)
            only_held = st.toggle("Held-out (unseen) patients only", value=False, key="board_heldout")
            view = board[board["set"] == "held-out"] if only_held else board
            alert_txt = {0: "—", 1: "HYPO risk", 2: "HYPER risk"}

            def cgm_state_now(g):
                # The current reading comes first: a patient already out of range is flagged whatever the model says.
                if g < LEVEL2_HYPO_MG_DL:
                    return f"LOW <{LEVEL2_HYPO_MG_DL:.0f} (L2)"
                if g < HYPO_MG_DL:
                    return f"LOW <{HYPO_MG_DL:.0f} (L1)"
                if g > HYPER_MG_DL:
                    return f"HIGH >{HYPER_MG_DL:.0f}"
                return (f"{TARGET_HIGH_MG_DL + 1:.0f}–{HYPER_MG_DL:.0f}" if g > TARGET_HIGH_MG_DL
                        else f"{HYPO_MG_DL:.0f}–{TARGET_HIGH_MG_DL:.0f}")
            table = pd.DataFrame({
                "Patient": view["patient_id"], "Name (synthetic)": view["name"], "Set": view["set"],
                "Model alert (2 h)": view["alert"].map(alert_txt), "Hypo score": view["p_hypo"],
                "Hyper score": view["p_hyper"], "CGM now": view["cgm"].round(0),
                "CGM state now": view["cgm"].map(cgm_state_now), "Δ 15 min": view["trend"].round(1),
                "Forecast +2 h": view["forecast"].round(0), "Risk tier": view["risk_tier"],
                "Medication": view["medication"],
            })
            st.dataframe(
                table, hide_index=True, width="stretch", height=420,
                column_config={
                    "Hypo score": st.column_config.ProgressColumn(format="%.2f", min_value=0.0, max_value=1.0,
                                                                  help="Model risk score (uncalibrated)"),
                    "Hyper score": st.column_config.ProgressColumn(format="%.2f", min_value=0.0, max_value=1.0,
                                                                   help="Model risk score (uncalibrated)"),
                    "CGM now": st.column_config.NumberColumn(format="%.0f mg/dL"),
                    "Forecast +2 h": st.column_config.NumberColumn(format="%.0f mg/dL"),
                })
            st.caption("Training patients are included for completeness, but the model has seen their data; "
                       "held-out rows are the honest test.")
            top = list(view["patient_id"].head(30))
            if top:
                oc1, oc2 = st.columns([2.2, 1])
                with oc1:
                    pick = st.selectbox("Open a patient from the board", top, key="board_pick",
                                        format_func=lambda p: f"{p} · {ehr_by_pid.loc[p, 'name']} · "
                                        + alert_txt[int(board.loc[board['patient_id'] == p, 'alert'].iloc[0])])
                with oc2:
                    st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
                    st.button("Open in Live Monitor", key="board_open", on_click=open_patient, args=(pick, i),
                              width="stretch")

    chart_c1, chart_c2 = st.columns(2)
    with chart_c1:
        fig_h = go.Figure(go.Histogram(x=hba1c, nbinsx=15, marker_color=TEAL, marker_line=dict(color="#FFFFFF", width=1),
                                       name="patients", hovertemplate="HbA1c %{x}%: %{y} patients<extra></extra>"))
        fig_h.add_vline(x=7.0, line_width=2, line_dash="dash", line_color=SLATE,
                        annotation_text="common ADA target <7.0%", annotation_position="top right",
                        annotation_font=dict(color=SLATE, size=11))
        lay = plot_layout("Baseline HbA1c (synthetic personas)", height=300, legend_bottom=False)
        lay["xaxis"]["title"] = "HbA1c (%)"
        lay["yaxis"]["title"] = "patients"
        lay["showlegend"] = False
        fig_h.update_layout(**lay)
        st.plotly_chart(fig_h, width="stretch", config={"displayModeBar": False})
    with chart_c2:
        counts = tiers.value_counts()
        order = [t for t in ("High Risk", "Moderate Risk", "Low Risk") if t in counts.index] + \
                [t for t in counts.index if t not in ("High Risk", "Moderate Risk", "Low Risk")]
        cmap = {"High Risk": RED, "Moderate Risk": AMBER, "Low Risk": GREEN}
        fig_p = go.Figure(go.Pie(labels=order, values=[int(counts[t]) for t in order], hole=0.55, sort=False,
                                 marker=dict(colors=[cmap.get(t, SLATE) for t in order], line=dict(color="#FFFFFF", width=2)),
                                 textinfo="label+value"))
        lay = plot_layout("Clinical risk tier", height=300, legend_bottom=False)
        lay["showlegend"] = False
        fig_p.update_layout(**lay)
        st.plotly_chart(fig_p, width="stretch", config={"displayModeBar": False})
        st.caption("Tier set by the synthetic generator from drug class (insulin or sulfonylurea → higher "
                   "hypoglycaemia risk) and age.")

    n_feat = len(bundle["feature_cols"]) if bundle else len(FEATURE_COLS)
    st.markdown(f"""
    <p style='font-size:14px; font-weight:700; color:#0F172A; margin:12px 0 8px 0;'>How the pieces fit</p>
    <div class="gt-card" style="padding:18px;">
        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(210px, 1fr)); gap:14px;">
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:14px; border-top:4px solid #0D9488;">
                <div class="gt-label" style="color:#0D9488;">1. Synthetic streams</div>
                <div style="font-size:14px; font-weight:700; color:#0F172A; margin:4px 0 6px 0;">EHR + wearables</div>
                <div style="font-size:12px; color:#64748B; line-height:1.5;">{len(EHR_FEATURES)} EHR features (demographics, BMI, diet, diabetes duration, HbA1c, fasting glucose, BP, eGFR, medication, insulin regimen, risk tier, prior hypos, family history, TCF7L2) and 15-min CGM, HR, HRV, steps, sleep stage, carbs, insulin. Past diagnoses, LDL and TG are shown on the patient card but are not model inputs. data/generate_datasets.py</div>
            </div>
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:14px; border-top:4px solid #0284C7;">
                <div class="gt-label" style="color:#0284C7;">2. Fusion features</div>
                <div style="font-size:14px; font-weight:700; color:#0F172A; margin:4px 0 6px 0;">Backward-looking windows</div>
                <div style="font-size:12px; color:#64748B; line-height:1.5;">{len(WEARABLE_FEATURES)} wearable features (15/30-min velocity, 1 h mean/SD, 1-2 h sums) joined to the EHR per patient: {n_feat} inputs in total.</div>
            </div>
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:14px; border-top:4px solid #8B5CF6;">
                <div class="gt-label" style="color:#8B5CF6;">3. ML models</div>
                <div style="font-size:14px; font-weight:700; color:#0F172A; margin:4px 0 6px 0;">HistGradientBoosting</div>
                <div style="font-size:12px; color:#64748B; line-height:1.5;">Classifier (class_weight="balanced") for hypo/hyper within 2 h, alert thresholds chosen by patient-grouped CV on training patients; regressor for glucose at +2 h.</div>
            </div>
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:14px; border-top:4px solid #10B981;">
                <div class="gt-label" style="color:#10B981;">4. Clinician view</div>
                <div style="font-size:14px; font-weight:700; color:#0F172A; margin:4px 0 6px 0;">This dashboard</div>
                <div style="font-size:12px; color:#64748B; line-height:1.5;">Live replay with alerts, risk board, data ingest checks, model card. The What-If sandbox is rule-based and separate from the ML models.</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)
    if st.button("Export cohort summary (Demo)", key="demo_export"):
        st.toast("Demo only: nothing was exported.")


# ==============================================================================
# VIEW 5: MODEL CARD (every number read from models/metrics.json at runtime)
# ==============================================================================
elif nav_tab == "Model Card":
    page_title("Model card",
               "Every number on this page is read at runtime from <code>models/metrics.json</code>, written by "
               "<code>python models/evaluate.py</code>. Nothing here is typed in by hand.")
    if metrics is None:
        st.error(f"Model metrics unavailable: {METRICS_ERROR}")
        st.stop()

    m = metrics
    ho = mget(m, "holdout", default={})
    cv = mget(m, "grouped_cv_5fold", default={})
    reg_m, per_m, lin_m = mget(ho, "regressor", default={}), mget(ho, "baseline_persistence", default={}), \
        mget(ho, "baseline_linear_extrapolation", default={})
    c_hypo, c_hyper = mget(ho, "classifier", "hypo_lt70", default={}), mget(ho, "classifier", "hyper_gt250", default={})
    e_hypo, e_hyper = mget(ho, "episodes_hypo", default={}), mget(ho, "episodes_hyper", default={})
    n_test = mget(ho, "test_patients")
    clarke = mget(reg_m, "clarke_pct", default={})
    ab = (clarke.get("A") or 0) + (clarke.get("B") or 0) if clarke else None

    def tile(col, label, value, sub):
        with col:
            st.markdown(f"<div class='gt-card'><div class='gt-label'>{label}</div><div class='gt-big'>{value}</div>"
                        f"<div class='gt-sub'>{sub}</div></div>", unsafe_allow_html=True)

    t1, t2, t3, t4, t5 = st.columns(5)
    tile(t1, "+2 h forecast MAE (held-out)", f"{fnum(reg_m.get('mae'))} mg/dL",
         f"persistence {fnum(per_m.get('mae'))} · linear {fnum(lin_m.get('mae'))}")
    tile(t2, "Clarke zone A (held-out)", f"{fnum(clarke.get('A'))}%", f"A+B {fnum(ab)}% · D {fnum(clarke.get('D'))}%")
    tile(t3, f"Hypo <{HYPO_MG_DL:.0f} in 2 h", f"{fpct(c_hypo.get('sensitivity'), 0)} sens.",
         f"precision {fpct(c_hypo.get('precision'), 0)} · PR-AUC {fnum(c_hypo.get('pr_auc'), '{:.2f}')}")
    tile(t4, f"Hyper >{HYPER_MG_DL:.0f} in 2 h", f"{fpct(c_hyper.get('sensitivity'), 0)} sens.",
         f"precision {fpct(c_hyper.get('precision'), 0)} · PR-AUC {fnum(c_hyper.get('pr_auc'), '{:.2f}')}")
    tile(t5, "Episodes warned 15–120 min ahead",
         f"{fnum(e_hypo.get('warned_15_to_120_min_ahead'), '{:.0f}')}/{fnum(e_hypo.get('episodes'), '{:.0f}')} hypo",
         f"{fnum(e_hyper.get('warned_15_to_120_min_ahead'), '{:.0f}')}/{fnum(e_hyper.get('episodes'), '{:.0f}')} hyper episodes")

    st.caption(f"Held-out = {fnum(n_test, '{:.0f}')} synthetic patients never used for training "
               f"({fnum(mget(ho, 'windows'), '{:,.0f}')} windows). Split: {esc(mget(ho, 'split', default='n/a'))}.")

    def html_table(headers, rows, highlight=None):
        head = "".join(f"<th>{esc(h)}</th>" for h in headers)
        body = "".join(
            f"<tr class='{'hl' if highlight is not None and k == highlight else ''}'>"
            + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for k, r in enumerate(rows))
        return f"<table class='mc-table'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    with st.container(border=True):
        st.markdown("**Glucose at +2 h: held-out patients**")
        left, right = st.columns([1.45, 1])
        with left:
            rows = []
            for name, d in (("GlycoTwin regressor", reg_m), ("Persistence (glucose stays the same)", per_m),
                            ("Linear extrapolation of the last 15 min", lin_m),
                            ("GlycoTwin, 5-fold grouped CV (all patients)", mget(cv, "regressor", default={}))):
                cz = d.get("clarke_pct", {}) or {}
                rows.append([name, fnum(d.get("mae")), fnum(d.get("rmse")), fnum(d.get("r2"), "{:.3f}"),
                             fnum(cz.get("A")), fnum(cz.get("B")), fnum(cz.get("C")), fnum(cz.get("D")), fnum(cz.get("E"))])
            st.markdown(html_table(["Model", "MAE", "RMSE", "R²", "A %", "B %", "C %", "D %", "E %"], rows, highlight=0),
                        unsafe_allow_html=True)
            st.caption("MAE/RMSE in mg/dL. A–E: Clarke error grid zones (% of windows).")
        with right:
            if clarke and per_m.get("clarke_pct"):
                zones = ["A", "B", "C", "D", "E"]
                fig_c = go.Figure()
                fig_c.add_trace(go.Bar(x=zones, y=[clarke.get(z, 0) for z in zones], name="GlycoTwin regressor",
                                       marker_color=TEAL, hovertemplate="zone %{x}: %{y:.2f}%<extra>GlycoTwin</extra>"))
                fig_c.add_trace(go.Bar(x=zones, y=[per_m["clarke_pct"].get(z, 0) for z in zones], name="persistence",
                                       marker_color=MUTED, hovertemplate="zone %{x}: %{y:.2f}%<extra>persistence</extra>"))
                lay = plot_layout("Clarke zones, held-out (% of windows)", height=250)
                lay["barmode"] = "group"
                lay["hovermode"] = "closest"
                lay["bargap"] = 0.3
                lay["bargroupgap"] = 0.08
                fig_c.update_layout(**lay)
                fig_c.update_yaxes(title_text="% of windows", range=[0, 100])
                st.plotly_chart(fig_c, width="stretch", config={"displayModeBar": False})

    with st.container(border=True):
        st.markdown(f"**2-h event alerts: held-out patients** (thresholds hypo ≥ "
                    f"{fnum(mget(ho, 'alert_thresholds', 'hypo'), '{:.2f}')}, hyper ≥ "
                    f"{fnum(mget(ho, 'alert_thresholds', 'hyper'), '{:.2f}')})")
        rows = []
        sources = [("GlycoTwin (held-out)", mget(ho, "classifier", default={})),
                   ("GlycoTwin, only windows not already in an event", mget(ho, "classifier_not_already_in_event", default={})),
                   ("Simple rule: CGM <90 and falling / >200 and rising", mget(ho, "rule_baseline_events", default={})),
                   ("GlycoTwin, 5-fold patient-grouped CV (all patients)", mget(cv, "classifier", default={}))]
        for name, src in sources:
            for key, lab in (("hypo_lt70", f"hypo <{HYPO_MG_DL:.0f}"), ("hyper_gt250", f"hyper >{HYPER_MG_DL:.0f}")):
                d = src.get(key, {}) or {}
                rows.append([name, lab, fnum(d.get("positives"), "{:,.0f}"), fpct(d.get("sensitivity")),
                             fpct(d.get("precision")), fpct(d.get("specificity")),
                             fnum(d.get("roc_auc"), "{:.3f}"), fnum(d.get("pr_auc"), "{:.3f}")])
        st.markdown(html_table(["Model", "Event", "Positive windows", "Sensitivity", "Precision", "Specificity",
                                "ROC-AUC", "PR-AUC"], rows), unsafe_allow_html=True)
        st.caption("Window-level metrics. Events are rare, so PR-AUC and precision matter more than ROC-AUC.")

    with st.container(border=True):
        st.markdown("**Episodes and alarm burden: held-out patients**")
        rows = []
        for lab, d in ((f"Hypoglycaemia (<{HYPO_MG_DL:.0f})", e_hypo), (f"Marked hyperglycaemia (>{HYPER_MG_DL:.0f})", e_hyper)):
            rows.append([lab, f"{fnum(d.get('warned_15_to_120_min_ahead'), '{:.0f}')} of {fnum(d.get('episodes'), '{:.0f}')}",
                         fpct(d.get("episode_sensitivity"), 0), fnum(d.get("median_lead_time_min"), "{:.0f} min"),
                         fnum(d.get("alarms_per_patient_day"), "{:.2f}"), fpct(d.get("false_alarm_fraction"), 0)])
        st.markdown(html_table(["Episode type", "Warned 15–120 min ahead", "Episode sensitivity", "Median lead",
                                "Alarms / patient-day", "Alarms not followed by event"], rows), unsafe_allow_html=True)
        st.caption(f"Honest reading: most alarms are not followed by an event "
                   f"({fpct(e_hypo.get('false_alarm_fraction'), 0)} of hypo alarm onsets and "
                   f"{fpct(e_hyper.get('false_alarm_fraction'), 0)} of hyper alarm onsets), at about "
                   f"{fnum(e_hypo.get('alarms_per_patient_day'), '{:.2f}')} hypo and "
                   f"{fnum(e_hyper.get('alarms_per_patient_day'), '{:.2f}')} hyper alarms per patient-day.")

    ab_m = mget(m, "ablation_grouped_cv_5fold", default={})
    swap = mget(ho, "ehr_swap_test", default={})
    lat = mget(m, "latency", default={})
    left2, right2 = st.columns([1.3, 1])
    with left2:
        with st.container(border=True):
            st.markdown("**Ablation: which inputs matter** (5-fold patient-grouped CV)")
            labels = {"fused_all_features": "All features (EHR + wearables)", "wearables_only": "Wearables only",
                      "ehr_only_plus_hour": "EHR only + hour of day", "without_cgm": "Without CGM",
                      "without_hr_hrv_sleep": "Without HR, HRV, sleep",
                      "without_history_and_genetics": "Without hypo history, family history, TCF7L2",
                      "without_hour_of_day": "Without hour of day"}
            rows = [[labels.get(k, k), fnum(d.get("features"), "{:.0f}"), fnum(d.get("regressor_mae"), "{:.2f}"),
                     fnum(d.get("hypo_pr_auc"), "{:.3f}"), fnum(d.get("hyper_pr_auc"), "{:.3f}")]
                    for k, d in ab_m.items()]
            st.markdown(html_table(["Variant", "Features", "MAE", "Hypo PR-AUC", "Hyper PR-AUC"], rows, highlight=0),
                        unsafe_allow_html=True)
            full, wo = ab_m.get("fused_all_features", {}), ab_m.get("wearables_only", {})
            hrv = ab_m.get("without_hr_hrv_sleep", {})
            if full and wo and hrv:
                st.caption(
                    f"Adding the EHR to the wearables changes MAE from {fnum(wo.get('regressor_mae'), '{:.2f}')} to "
                    f"{fnum(full.get('regressor_mae'), '{:.2f}')} and hypo PR-AUC from "
                    f"{fnum(wo.get('hypo_pr_auc'), '{:.3f}')} to {fnum(full.get('hypo_pr_auc'), '{:.3f}')}; hyper "
                    f"detection is driven by the wearables (PR-AUC {fnum(wo.get('hyper_pr_auc'), '{:.3f}')} without "
                    f"the EHR vs {fnum(full.get('hyper_pr_auc'), '{:.3f}')} with it). Dropping HR, HRV and sleep moves "
                    f"MAE by {fnum((hrv.get('regressor_mae') or 0) - (full.get('regressor_mae') or 0), '{:+.2f}')}: "
                    "they add little independent signal in this synthetic data.")
    with right2:
        with st.container(border=True):
            st.markdown("**EHR-swap test** (held-out patients scored with another patient's EHR)")
            rows = [["2-h forecast MAE", fnum(reg_m.get("mae"), "{:.2f}"), fnum(swap.get("regressor_mae"), "{:.2f}")],
                    ["Hypo PR-AUC", fnum(c_hypo.get("pr_auc"), "{:.3f}"), fnum(swap.get("hypo_pr_auc"), "{:.3f}")],
                    ["Hyper PR-AUC", fnum(c_hyper.get("pr_auc"), "{:.3f}"), fnum(swap.get("hyper_pr_auc"), "{:.3f}")]]
            st.markdown(html_table(["Metric", "Own EHR", "Swapped EHR"], rows), unsafe_allow_html=True)
            st.caption("Worse scores with a swapped EHR show the model really uses the EHR inputs.")
        with st.container(border=True):
            st.markdown("**Data and runtime**")
            dd = mget(m, "data", default={})
            lc = dd.get("label_counts", {}) or {}
            ins_total = int(df_ehr["insulin_regimen"].astype(str).str.strip().ne("None").sum())
            ins_test = (int(df_ehr[df_ehr["patient_id"].isin(TEST_PATIENTS)]["insulin_regimen"].astype(str).str.strip().ne("None").sum())
                        if bundle else None)
            rows = [["Patients · windows", f"{fnum(dd.get('patients'), '{:.0f}')} · {fnum(dd.get('windows'), '{:,.0f}')}"],
                    ["Features (wearable + EHR)", f"{fnum(mget(dd, 'features', 'total'), '{:.0f}')} "
                     f"({fnum(mget(dd, 'features', 'wearable'), '{:.0f}')} + {fnum(mget(dd, 'features', 'ehr'), '{:.0f}')})"],
                    ["Window labels: none / hypo / hyper", f"{fnum(lc.get('0'), '{:,.0f}')} / {fnum(lc.get('1'), '{:,.0f}')} / {fnum(lc.get('2'), '{:,.0f}')}"],
                    [f"CGM readings <{LEVEL2_HYPO_MG_DL:.0f} · <{HYPO_MG_DL:.0f}",
                     f"{fnum(dd.get('cgm_readings_lt54'), '{:.0f}')} · {fnum(dd.get('cgm_readings_lt70'), '{:.0f}')}"],
                    [f"Patients with any reading <{HYPO_MG_DL:.0f}", fnum(dd.get("patients_with_any_lt70"), "{:.0f}")],
                    ["Insulin-treated personas (all · held-out)", f"{ins_total} · {fnum(ins_test, '{:.0f}')}"],
                    ["Latency, one window (median)", fnum(lat.get("single_window_ms_median"), "{:.0f} ms")],
                    ["Latency, batched", fnum(lat.get("batched_us_per_window"), "{:.0f} µs per window")]]
            st.markdown(html_table(["Item", "Value"], rows), unsafe_allow_html=True)
            st.caption(esc(lat.get("hardware_note", "")))

    with st.container(border=True):
        st.markdown("**Limitations**")
        lims = list(mget(m, "notes", default=[]) or [])
        lims += [
            "Generator parameters are illustrative: population ranges loosely informed by published Indian "
            "literature, not calibrated to any dataset.",
            "Risk scores are the raw outputs of a class_weight=\"balanced\" classifier and are uncalibrated; "
            "compare them with the alert threshold and do not read them as the chance of an event.",
            f"Only {ins_total} of {len(df_ehr)} personas use insulin and {fnum(ins_test, '{:.0f}')} are in the held-out "
            "set, so performance for insulin-treated patients is not tested.",
            "The +2 h value is a single point forecast; the dotted line on the Live Monitor is interpolation, not a "
            "predicted trajectory.",
            "Alert thresholds were chosen by F2 score on out-of-fold predictions of the training patients only.",
            "The What-If sandbox is rule-based and separate from these models.",
            f"{DISCLAIMER} Not for dosing decisions.",
        ]
        st.markdown("\n".join(f"- {esc(x)}" for x in lims))
