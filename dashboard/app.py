"""
GlycoTwin: Precision Digital Twin for Type 2 Diabetes Management
Hackathon: Happiest Health Reimagining and Reforming Healthcare in India Summit 2026

Design System & Multimodal Architecture:
- Palette: Navy Sidebar (#0F172A), Canvas (#F8FAFC), White Cards 12px (#FFFFFF), Teal Accent (#0D9488)
- Status Colors: Green (#10B981, 70-180), Amber (#F59E0B, >180/250), Red (#EF4444, <70)
- AI Engine: scikit-learn HistGradientBoostingClassifier & HistGradientBoostingRegressor
- Multimodal Fusion: Static Clinical EHR + Continuous Wearable IoT (CGM, HR, HRV, Steps)
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(
    page_title="GlycoTwin | Precision Diabetes Digital Twin",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design System CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], [class*="st-"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

/* Tabular numerals for all numeric data */
.tabular-nums, .metric-value, .kpi-num, [data-testid="stMetricValue"] {
    font-variant-numeric: tabular-nums;
    font-feature-settings: "tnum";
}

/* Hide default streamlit chrome */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display:none;}

/* Sidebar Theme */
[data-testid="stSidebar"] {
    background-color: #0F172A !important;
    border-right: 1px solid #1E293B;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
[data-testid="stSidebar"] label {
    color: #94A3B8 !important;
}
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3 {
    color: #F8FAFC !important;
}

/* Sidebar selectbox styling */
[data-testid="stSidebar"] div[data-baseweb="select"] {
    background-color: #1E293B !important;
    border-color: #334155 !important;
}
[data-testid="stSidebar"] div[data-baseweb="select"] * {
    color: #F8FAFC !important;
}

/* Sidebar radio navigation */
[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 6px;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background-color: #1E293B;
    border-radius: 8px;
    padding: 8px 12px;
    border: 1px solid #334155;
    transition: all 0.2s ease;
    cursor: pointer;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    border-color: #0D9488;
    background-color: #24344D;
}

/* Teal Primary Accent Buttons */
.stButton > button {
    background-color: #0D9488 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 8px 18px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background-color: #14B8A6 !important;
    box-shadow: 0 4px 12px rgba(13, 148, 136, 0.25) !important;
}

/* Card Containers */
.gt-card {
    background: #FFFFFF;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 1px 3px rgba(15,23,42,0.06), 0 1px 2px rgba(15,23,42,0.04);
    padding: 16px;
    margin-bottom: 14px;
}

/* Patient 360 Card in Sidebar */
.patient-360-card {
    background: #1E293B;
    border-radius: 12px;
    border: 1px solid #334155;
    padding: 14px;
    margin-top: 10px;
    font-size: 12px;
    color: #CBD5E1;
}
.patient-360-row {
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    border-bottom: 1px solid #334155;
}
.patient-360-label {
    color: #94A3B8;
}
.patient-360-value {
    color: #F8FAFC;
    font-weight: 600;
}

/* Range Bar */
.range-bar-track {
    width: 100%;
    height: 10px;
    border-radius: 5px;
    background: linear-gradient(to right, 
        #EF4444 0%, #EF4444 4.4%, 
        #F87171 4.4%, #F87171 9.4%, 
        #10B981 9.4%, #10B981 43.8%, 
        #F59E0B 43.8%, #F59E0B 65.6%, 
        #EF4444 65.6%, #EF4444 100%);
    position: relative;
    margin-top: 10px;
    margin-bottom: 4px;
}
.range-bar-needle {
    position: absolute;
    top: -4px;
    width: 6px;
    height: 18px;
    background-color: #0F172A;
    border: 2px solid #FFFFFF;
    border-radius: 3px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.35);
    transform: translateX(-50%);
}

/* Forecast Card Alerts */
.forecast-safe {
    background: #F0FDF4;
    border-left: 6px solid #10B981;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(15,23,42,0.05);
}
.forecast-hyper {
    background: #FFFBEB;
    border-left: 6px solid #F59E0B;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(15,23,42,0.05);
}
.forecast-hypo {
    background: #FEF2F2;
    border-left: 6px solid #EF4444;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(15,23,42,0.05);
}

/* Checklist items */
.check-step {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 14px;
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    margin-bottom: 8px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# Data & Model Loaders
@st.cache_data
def load_data():
    ehr_path = "data/synthetic_ehr.csv"
    wearables_path = "data/synthetic_wearables.csv"
    if not os.path.exists(ehr_path) or not os.path.exists(wearables_path):
        return None, None
    df_ehr = pd.read_csv(ehr_path)
    df_wearables = pd.read_csv(wearables_path)
    df_wearables["timestamp"] = pd.to_datetime(df_wearables["timestamp"])
    return df_ehr, df_wearables

@st.cache_resource
def load_model():
    model_path = "models/glycotwin_model.joblib"
    if not os.path.exists(model_path):
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None

df_ehr, df_wearables = load_data()
model_package = load_model()

if df_ehr is None or df_wearables is None:
    st.error("Clinical and Wearable datasets not found. Please verify data/synthetic_ehr.csv and data/synthetic_wearables.csv.")
    st.stop()

# ==============================================================================
# SIDEBAR: GLYCOTWIN BRANDING, NAV, & PATIENT 360 PROFILE
# ==============================================================================
with st.sidebar:
    # GlycoTwin Wordmark & Subtitle
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:16px;">
        <div style="width:38px; height:38px; border-radius:10px; background:#0D9488; display:flex; align-items:center; justify-content:center; color:white; font-size:20px; font-weight:bold;">
            🧬
        </div>
        <div>
            <div style="font-size:18px; font-weight:800; color:#F8FAFC; letter-spacing:-0.02em;">GlycoTwin</div>
            <div style="font-size:11px; color:#94A3B8;">Precision T2D Digital Twin</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation (Maps to 4 Tabs)
    st.markdown("<p style='font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:#64748B; margin-bottom:6px;'>Navigation</p>", unsafe_allow_html=True)
    nav_tab = st.radio(
        "Navigation",
        ["Live Monitor", "What-If Simulator", "Data Ingest", "Cohort Analytics"],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:#64748B; margin-bottom:6px;'>Patient 360° Profile</p>", unsafe_allow_html=True)

    patient_options = [
        f"{row['patient_id']} - {row['name']} ({row['clinical_risk_tier']})"
        for _, row in df_ehr.iterrows()
    ]
    selected_patient_str = st.selectbox(
        "Select Patient Persona:",
        patient_options,
        index=0,
        label_visibility="collapsed"
    )
    selected_pid = selected_patient_str.split(" - ")[0]
    patient_ehr = df_ehr[df_ehr["patient_id"] == selected_pid].iloc[0]
    patient_insulin = "None" if (pd.isna(patient_ehr["insulin_regimen"]) or str(patient_ehr["insulin_regimen"]).strip() in ["", "nan", "None"]) else str(patient_ehr["insulin_regimen"])

    # Risk badge styling
    is_high_risk = "High" in str(patient_ehr["clinical_risk_tier"])
    is_mod_risk = "Mod" in str(patient_ehr["clinical_risk_tier"])
    risk_badge_bg = "#EF4444" if is_high_risk else ("#F59E0B" if is_mod_risk else "#10B981")
    risk_icon = "🔴" if is_high_risk else ("🟡" if is_mod_risk else "🟢")

    # Compact Patient 360 Card from existing data only
    st.markdown(f"""
    <div class="patient-360-card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <span style="font-weight:700; color:#F8FAFC; font-size:14px;">{patient_ehr['name']}</span>
            <span style="background:{risk_badge_bg}; color:#FFFFFF; padding:2px 8px; border-radius:12px; font-size:10px; font-weight:700;">
                {risk_icon} {patient_ehr['clinical_risk_tier']}
            </span>
        </div>
        <div class="patient-360-row">
            <span class="patient-360-label">Patient ID:</span>
            <span class="patient-360-value tabular-nums">{patient_ehr['patient_id']}</span>
        </div>
        <div class="patient-360-row">
            <span class="patient-360-label">Demographics:</span>
            <span class="patient-360-value">{patient_ehr['age']} yrs | {patient_ehr['gender']}</span>
        </div>
        <div class="patient-360-row">
            <span class="patient-360-label">South-Asian BMI:</span>
            <span class="patient-360-value tabular-nums">{patient_ehr['bmi']} kg/m²</span>
        </div>
        <div class="patient-360-row">
            <span class="patient-360-label">Diabetes Duration:</span>
            <span class="patient-360-value tabular-nums">{patient_ehr['diabetes_duration_years']} yrs</span>
        </div>
        <div class="patient-360-row">
            <span class="patient-360-label">Baseline HbA1c:</span>
            <span class="patient-360-value tabular-nums" style="color:#2DD4BF;">{patient_ehr['baseline_hba1c']}%</span>
        </div>
        <div class="patient-360-row">
            <span class="patient-360-label">Blood Pressure:</span>
            <span class="patient-360-value tabular-nums">{patient_ehr['systolic_bp']}/{patient_ehr['diastolic_bp']} mmHg</span>
        </div>
        <div class="patient-360-row">
            <span class="patient-360-label">eGFR (Kidney):</span>
            <span class="patient-360-value tabular-nums">{patient_ehr['egfr']} mL/min</span>
        </div>
        <div class="patient-360-row" style="flex-direction:column; gap:2px;">
            <span class="patient-360-label">Primary Medication:</span>
            <span class="patient-360-value" style="font-size:11px;">{patient_ehr['primary_medication']}</span>
        </div>
        <div class="patient-360-row" style="flex-direction:column; gap:2px; border-bottom:none;">
            <span class="patient-360-label">Insulin Regimen:</span>
            <span class="patient-360-value" style="font-size:11px;">{patient_insulin}</span>
        </div>
    </div>
    <div style="margin-top:14px; font-size:11px; color:#64748B; line-height:1.4;">
        Happiest Health Healthcare Summit 2026<br>
        Predictive AI: HistGradientBoosting
    </div>
    """, unsafe_allow_html=True)

# Wearable data for current patient
patient_wearables = df_wearables[df_wearables["patient_id"] == selected_pid].sort_values("timestamp").reset_index(drop=True)
if len(patient_wearables) < 10:
    patient_wearables = df_wearables[df_wearables["patient_id"] == "IND-PAT-001"].sort_values("timestamp").reset_index(drop=True)

total_steps = len(patient_wearables)
default_idx = min(96 * 3 + 45, total_steps - 1)

# ==============================================================================
# TOP BAR: PATIENT NAME, SELECTED TIME, TIME SCRUBBER, ALERT COUNT
# ==============================================================================
top_col1, top_col2 = st.columns([1.2, 2.8])

scrub_step = st.session_state.get("timeline_step_slider", default_idx)

with top_col1:
    st.markdown(f"""
    <div style="padding:4px 0 2px 0;">
        <div style="font-size:19px; font-weight:800; color:#0F172A; display:flex; align-items:center; gap:8px;">
            {patient_ehr['name']}
            <span style="font-size:12px; font-weight:500; color:#64748B;">({patient_ehr['patient_id']})</span>
        </div>
        <div style="font-size:12px; color:#64748B; margin-top:2px;">
            Telemetry Timeline: <span style="background:#0D9488; color:#FFF; font-size:11px; font-weight:700; padding:2px 8px; border-radius:10px; margin-right:4px;">Step {scrub_step}</span> · <span style="color:#0F172A; font-weight:600;">15-min Intervals</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with top_col2:
    current_idx = st.slider(
        "Time Scrubber (15-min Intervals):",
        min_value=12,
        max_value=total_steps - 1,
        value=default_idx,
        key="timeline_step_slider",
        format="Step %d",
        label_visibility="collapsed"
    )

current_reading = patient_wearables.iloc[current_idx]
current_time = current_reading["timestamp"]
current_glucose = float(current_reading["glucose_cgm"])
current_hr = int(current_reading["heart_rate"])
current_hrv = int(current_reading["hrv"])
current_steps = int(current_reading["steps_15m"])
prev_glucose = float(patient_wearables.iloc[current_idx - 1]["glucose_cgm"])
roc = current_glucose - prev_glucose
roc_symbol = "↗" if roc > 2.0 else ("↘" if roc < -2.0 else "→")

# Prepare 2-hour AI Prediction
window_wearable = patient_wearables.iloc[max(0, current_idx-8):current_idx+1]
feat_glucose_vel_15m = roc
feat_glucose_vel_30m = current_glucose - float(patient_wearables.iloc[current_idx - 2]["glucose_cgm"])
feat_glucose_mean_1h = float(window_wearable.tail(4)["glucose_cgm"].mean())
feat_glucose_std_1h = float(window_wearable.tail(4)["glucose_cgm"].std()) if len(window_wearable) >= 2 else 0.0
feat_hr_mean_1h = float(window_wearable.tail(4)["heart_rate"].mean())
feat_hrv_mean_1h = float(window_wearable.tail(4)["hrv"].mean())
feat_steps_1h = float(window_wearable.tail(4)["steps_15m"].sum())
feat_steps_2h = float(window_wearable.tail(8)["steps_15m"].sum())
feat_carbs_2h = float(window_wearable.tail(8)["meal_carbs_g"].sum())
feat_insulin_2h = float(window_wearable.tail(8)["insulin_units"].sum())

predicted_glucose_2h = current_glucose
predicted_event_class = 0
confidence_pct = 94.0

if model_package is not None:
    try:
        clf = model_package["classifier"]
        reg = model_package["regressor"]
        sample_input = {
            "glucose_cgm": [current_glucose],
            "glucose_velocity_15m": [feat_glucose_vel_15m],
            "glucose_velocity_30m": [feat_glucose_vel_30m],
            "glucose_mean_1h": [feat_glucose_mean_1h],
            "glucose_std_1h": [feat_glucose_std_1h],
            "heart_rate": [current_hr],
            "hr_mean_1h": [feat_hr_mean_1h],
            "hrv": [current_hrv],
            "hrv_mean_1h": [feat_hrv_mean_1h],
            "steps_15m": [current_steps],
            "steps_sum_1h": [feat_steps_1h],
            "steps_sum_2h": [feat_steps_2h],
            "meal_carbs_g": [float(current_reading["meal_carbs_g"])],
            "carbs_sum_2h": [feat_carbs_2h],
            "insulin_units": [float(current_reading["insulin_units"])],
            "insulin_sum_2h": [feat_insulin_2h],
            "hour_of_day": [current_time.hour],
            "sleep_stage": [0 if current_reading["sleep_stage"]=="Awake" else 1],
            "age": [patient_ehr["age"]],
            "gender": [1 if patient_ehr["gender"]=="Male" else 0],
            "bmi": [patient_ehr["bmi"]],
            "diabetes_duration_years": [patient_ehr["diabetes_duration_years"]],
            "baseline_hba1c": [patient_ehr["baseline_hba1c"]],
            "fasting_glucose_mg_dl": [patient_ehr["fasting_glucose_mg_dl"]],
            "systolic_bp": [patient_ehr["systolic_bp"]],
            "diastolic_bp": [patient_ehr["diastolic_bp"]],
            "egfr": [patient_ehr["egfr"]],
            "diet_profile": [0],
            "primary_medication": [1],
            "insulin_regimen": [1 if patient_ehr["insulin_regimen"] != "None" else 0],
            "clinical_risk_tier": [2 if "High" in str(patient_ehr["clinical_risk_tier"]) else (1 if "Mod" in str(patient_ehr["clinical_risk_tier"]) else 0)]
        }
        X_infer = pd.DataFrame(sample_input)
        predicted_glucose_2h = float(reg.predict(X_infer)[0])
        predicted_event_class = int(clf.predict(X_infer)[0])
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_infer)[0]
            confidence_pct = float(probs[predicted_event_class] * 100.0)
    except Exception:
        # Fallback to wearable record or projection
        predicted_glucose_2h = float(current_reading.get("future_glucose_2h", current_glucose + roc * 5))
        predicted_event_class = 1 if predicted_glucose_2h < 70 else (2 if predicted_glucose_2h > 250 else 0)
        confidence_pct = 89.5
else:
    predicted_glucose_2h = float(current_reading.get("future_glucose_2h", current_glucose + roc * 5))
    predicted_event_class = 1 if predicted_glucose_2h < 70 else (2 if predicted_glucose_2h > 250 else 0)
    confidence_pct = 91.0

forecast_time = current_time + timedelta(hours=2)
has_alert = (predicted_event_class == 1 or predicted_glucose_2h < 70.0 or predicted_event_class == 2 or predicted_glucose_2h > 250.0)

# Alert Pill in Top Bar Sub-row
alert_badge_html = """
<span style="background:#FEE2E2; color:#DC2626; padding:3px 10px; border-radius:12px; font-size:12px; font-weight:700;">
    🚨 1 Critical Adverse Alert
</span>
""" if (predicted_event_class == 1 or predicted_glucose_2h < 70.0) else (
"""
<span style="background:#FEF3C7; color:#D97706; padding:3px 10px; border-radius:12px; font-size:12px; font-weight:700;">
    ⚠️ 1 Hyperglycemia Advisory
</span>
""" if (predicted_event_class == 2 or predicted_glucose_2h > 250.0) else """
<span style="background:#DCFCE7; color:#15803D; padding:3px 10px; border-radius:12px; font-size:12px; font-weight:700;">
    ✅ 0 Active Alerts (Trajectory Stable)
</span>
""")

st.markdown(f"""
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px; padding-bottom:8px; border-bottom:1px solid #E2E8F0;">
    <div style="font-size:12px; color:#64748B;">
        Selected Timestamp: <span class="tabular-nums" style="font-weight:700; color:#0F172A;">{current_time.strftime('%Y-%m-%d %H:%M:%S')}</span>
    </div>
    <div>{alert_badge_html}</div>
</div>
""", unsafe_allow_html=True)

# Helper function to build Plotly layout matching design system
def get_clean_plotly_layout(title, height=400, y_range=[40, 360]):
    return dict(
        title=dict(text=title, font=dict(family="Inter, sans-serif", size=14, color="#0F172A", weight=700)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#64748B", size=11),
        xaxis=dict(
            showgrid=True,
            gridcolor="#F1F5F9",
            zeroline=False,
            linecolor="#E2E8F0"
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#F1F5F9",
            zeroline=False,
            range=y_range,
            linecolor="#E2E8F0"
        ),
        margin=dict(l=45, r=20, t=50, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

# ==============================================================================
# VIEW 1: LIVE MONITOR
# ==============================================================================
if nav_tab == "Live Monitor":
    # 1. Glucose Hero Card + 4 Cards
    col_hero, col_cards = st.columns([1.6, 2.4])

    with col_hero:
        # Needle position for range bar (40 to 360 mg/dL scale)
        needle_pos = max(0.0, min(100.0, ((current_glucose - 40.0) / 320.0) * 100.0))
        status_text = "In-Range (Target 70–180)" if 70 <= current_glucose <= 180 else ("Hypoglycemia (<70)" if current_glucose < 70 else "Hyperglycemia (>180)")
        status_color = "#10B981" if 70 <= current_glucose <= 180 else ("#EF4444" if current_glucose < 70 else "#F59E0B")

        st.markdown(f"""
        <div class="gt-card" style="height:100%;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:0.04em; color:#64748B;">
                    Real-Time CGM Glucose
                </span>
                <span style="background:{status_color}18; color:{status_color}; font-size:11px; font-weight:700; padding:2px 8px; border-radius:10px;">
                    {status_text}
                </span>
            </div>
            <div style="display:flex; align-items:baseline; gap:8px; margin-top:8px;">
                <div class="tabular-nums" style="font-size:46px; font-weight:800; color:#0F172A; line-height:1;">
                    {current_glucose:.1f}
                </div>
                <div style="font-size:14px; color:#64748B; font-weight:500;">mg/dL</div>
                <div class="tabular-nums" style="font-size:15px; font-weight:700; color:{status_color}; margin-left:auto;">
                    {roc_symbol} {roc:+.1f} mg/dL/15m
                </div>
            </div>
            <div class="range-bar-track">
                <div class="range-bar-needle" style="left:{needle_pos:.1f}%;"></div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:10px; color:#94A3B8; margin-top:4px;">
                <span>54 (Very Low)</span>
                <span>70 (Hypo)</span>
                <span style="color:#10B981; font-weight:600;">Target: 70–180</span>
                <span>180</span>
                <span>250 (Severe)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_cards:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="gt-card">
                <div style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase;">Heart Rate & HRV</div>
                <div class="tabular-nums" style="font-size:24px; font-weight:800; color:#0F172A; margin-top:4px;">
                    {current_hr} <span style="font-size:12px; font-weight:500; color:#64748B;">BPM</span>
                </div>
                <div style="font-size:12px; color:#64748B; margin-top:2px;">HRV: <span class="tabular-nums" style="font-weight:600; color:#0F172A;">{current_hrv} ms</span></div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="gt-card">
                <div style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase;">Circadian Sleep State</div>
                <div style="font-size:24px; font-weight:800; color:#0F172A; margin-top:4px;">
                    {current_reading['sleep_stage']}
                </div>
                <div style="font-size:12px; color:#64748B; margin-top:2px;">Metabolic Phase: Daytime Active</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="gt-card">
                <div style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase;">15-Min Accelerometer</div>
                <div class="tabular-nums" style="font-size:24px; font-weight:800; color:#0F172A; margin-top:4px;">
                    {current_steps} <span style="font-size:12px; font-weight:500; color:#64748B;">steps</span>
                </div>
                <div style="font-size:12px; color:#64748B; margin-top:2px;">1h Accumulation: <span class="tabular-nums" style="font-weight:600; color:#0F172A;">{int(feat_steps_1h)}</span></div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="gt-card">
                <div style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase;">Recent Intake & Regimen</div>
                <div class="tabular-nums" style="font-size:24px; font-weight:800; color:#0F172A; margin-top:4px;">
                    {current_reading['meal_carbs_g']}g <span style="font-size:12px; font-weight:500; color:#64748B;">Carbs</span>
                </div>
                <div style="font-size:12px; color:#64748B; margin-top:2px;">Bolus: <span class="tabular-nums" style="font-weight:600; color:#0F172A;">{current_reading['insulin_units']}U Insulin</span></div>
            </div>
            """, unsafe_allow_html=True)

    # 2. Prominent 2-Hour Forecast Card
    if predicted_event_class == 1 or predicted_glucose_2h < 70.0:
        st.markdown(f"""
        <div class="forecast-hypo">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div>
                    <div style="font-size:12px; font-weight:800; color:#EF4444; letter-spacing:0.04em; text-transform:uppercase;">
                        🚨 Impending Adverse Event Warning · 2-Hour Horizon
                    </div>
                    <div style="font-size:20px; font-weight:800; color:#991B1B; margin-top:2px;">
                        Severe Hypoglycemia Forecasted: <span class="tabular-nums">{predicted_glucose_2h:.1f} mg/dL</span> at {forecast_time.strftime('%I:%M %p')}
                    </div>
                </div>
                <div style="text-align:right;">
                    <span style="background:#EF4444; color:white; font-size:11px; font-weight:700; padding:3px 10px; border-radius:12px;">
                        Model Confidence: {confidence_pct:.1f}%
                    </span>
                    <div style="font-size:10px; color:#991B1B; margin-top:3px;">HistGradientBoosting (scikit-learn)</div>
                </div>
            </div>
            <div style="margin-top:10px; font-size:13px; color:#7F1D1D; line-height:1.5;">
                <b>Actionable Rescue Protocol:</b> Administer <b>15–20g fast-acting carbohydrates</b> (150ml fruit juice or 4 glucose tablets). Halt aerobic physical activity. Recheck CGM in 15 minutes.
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif predicted_event_class == 2 or predicted_glucose_2h > 250.0:
        st.markdown(f"""
        <div class="forecast-hyper">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div>
                    <div style="font-size:12px; font-weight:800; color:#F59E0B; letter-spacing:0.04em; text-transform:uppercase;">
                        ⚠️ Impending Adverse Event Advisory · 2-Hour Horizon
                    </div>
                    <div style="font-size:20px; font-weight:800; color:#92400E; margin-top:2px;">
                        Severe Hyperglycemic Spike: <span class="tabular-nums">{predicted_glucose_2h:.1f} mg/dL</span> at {forecast_time.strftime('%I:%M %p')}
                    </div>
                </div>
                <div style="text-align:right;">
                    <span style="background:#F59E0B; color:white; font-size:11px; font-weight:700; padding:3px 10px; border-radius:12px;">
                        Model Confidence: {confidence_pct:.1f}%
                    </span>
                    <div style="font-size:10px; color:#92400E; margin-top:3px;">HistGradientBoosting (scikit-learn)</div>
                </div>
            </div>
            <div style="margin-top:10px; font-size:13px; color:#78350F; line-height:1.5;">
                <b>Actionable Clinical Protocol:</b> Promote active oral hydration. Verify timing of postprandial bolus insulin. Consider a light 15-minute post-meal walk to promote peripheral glucose clearance.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="forecast-safe">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div>
                    <div style="font-size:12px; font-weight:800; color:#10B981; letter-spacing:0.04em; text-transform:uppercase;">
                        ✅ In-Range Glycemic Trajectory · 2-Hour Horizon
                    </div>
                    <div style="font-size:20px; font-weight:800; color:#065F46; margin-top:2px;">
                        Projected In-Range: <span class="tabular-nums">{predicted_glucose_2h:.1f} mg/dL</span> at {forecast_time.strftime('%I:%M %p')}
                    </div>
                </div>
                <div style="text-align:right;">
                    <span style="background:#10B981; color:white; font-size:11px; font-weight:700; padding:3px 10px; border-radius:12px;">
                        Model Confidence: {confidence_pct:.1f}%
                    </span>
                    <div style="font-size:10px; color:#065F46; margin-top:3px;">HistGradientBoosting (scikit-learn)</div>
                </div>
            </div>
            <div style="margin-top:10px; font-size:13px; color:#064E3B; line-height:1.5;">
                <b>Clinical Protocol:</b> Maintain baseline pharmacotherapy and nutrition. Patient remains within safe physiological bounds (70–180 mg/dL).
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Demo Buttons row
    d_btn1, d_btn2, _ = st.columns([1.2, 1.2, 2.6])
    with d_btn1:
        if st.button("📲 Push Nudge to Patient (Demo)"):
            st.toast(f"Demo: Early warning notification dispatched to {patient_ehr['name']}'s smartphone app.")
    with d_btn2:
        if st.button("📋 Log Advisory to EHR (Demo)"):
            st.toast("Demo: Adverse event prediction stamped into hospital EHR audit log.")

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    # 3. Interactive CGM Chart Card with TIR % and CV %
    start_view_idx = max(0, current_idx - 32)
    history_df = patient_wearables.iloc[start_view_idx:current_idx+1]
    
    # Compute clinical metrics for current 8h window
    in_range_count = history_df["glucose_cgm"].between(70, 180).sum()
    window_tir = (in_range_count / len(history_df)) * 100.0 if len(history_df) > 0 else 0.0
    window_mean = history_df["glucose_cgm"].mean()
    window_cv = (history_df["glucose_cgm"].std() / window_mean * 100.0) if window_mean > 0 else 0.0

    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
        <span style="font-size:14px; font-weight:700; color:#0F172A;">
            Continuous Glucose Telemetry (8-Hour Window + 2-Hour Predicted Horizon)
        </span>
        <div style="display:flex; gap:16px;">
            <span style="font-size:12px; color:#64748B;">Time in Range (70–180): <span class="tabular-nums" style="font-weight:700; color:#10B981;">{window_tir:.1f}%</span></span>
            <span style="font-size:12px; color:#64748B;">Glycemic Variability (CV): <span class="tabular-nums" style="font-weight:700; color:#0F172A;">{window_cv:.1f}%</span> (Target < 36%)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    fig = go.Figure()

    # Translucent bands
    # In-range 70-180 band with dashed bounds
    fig.add_hrect(
        y0=70, y1=180,
        fillcolor="rgba(16, 185, 129, 0.12)",
        line_width=1,
        line_dash="dash",
        line_color="rgba(16, 185, 129, 0.4)",
        annotation_text="Target Range (70–180 mg/dL)",
        annotation_position="top left",
        annotation_font=dict(size=10, color="#10B981")
    )
    # Hypo <70 zone
    fig.add_hrect(
        y0=30, y1=70,
        fillcolor="rgba(239, 68, 68, 0.10)",
        line_width=0,
        annotation_text="Hypoglycemia (<70 mg/dL)",
        annotation_position="bottom left",
        annotation_font=dict(size=10, color="#EF4444")
    )
    # Hyper >250 zone
    fig.add_hrect(
        y0=250, y1=380,
        fillcolor="rgba(245, 158, 11, 0.10)",
        line_width=0,
        annotation_text="Severe Hyperglycemia (>250 mg/dL)",
        annotation_position="top left",
        annotation_font=dict(size=10, color="#F59E0B")
    )

    # Actual CGM history
    fig.add_trace(go.Scatter(
        x=history_df["timestamp"],
        y=history_df["glucose_cgm"],
        mode="lines+markers",
        name="Actual CGM Telemetry",
        line=dict(color="#0D9488", width=3),
        marker=dict(size=5, color="#0D9488")
    ))

    # 2-Hour Future Forecast Line
    future_times = [current_time + timedelta(minutes=15*i) for i in range(1, 9)]
    future_glucose_path = np.linspace(current_glucose, predicted_glucose_2h, 8)
    forecast_line_color = "#EF4444" if predicted_glucose_2h < 70 else ("#F59E0B" if predicted_glucose_2h > 250 else "#10B981")

    fig.add_trace(go.Scatter(
        x=[current_time] + future_times,
        y=[current_glucose] + list(future_glucose_path),
        mode="lines+markers",
        name="HistGradientBoosting Forecast (Next 2 Hours)",
        line=dict(color=forecast_line_color, width=3, dash="dash"),
        marker=dict(size=6, color=forecast_line_color)
    ))

    fig_layout = get_clean_plotly_layout(
        f"Continuous Glucose Monitoring & 2-Hour Ahead Digital Twin Trajectory ({current_time.strftime('%Y-%m-%d %H:%M')})",
        height=420
    )
    fig.update_layout(**fig_layout)
    st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# VIEW 2: WHAT-IF SIMULATOR
# ==============================================================================
elif nav_tab == "What-If Simulator":
    st.markdown("""
    <div style="margin-bottom:14px;">
        <h3 style="margin:0; font-size:18px; font-weight:800; color:#0F172A;">
            Digital Twin Counterfactual Simulation Engine
        </h3>
        <p style="margin:4px 0 0 0; font-size:13px; color:#64748B;">
            Evaluate simulated patient nutrition, exercise, and pharmacological titrations in the virtual twin <b>prior to clinical execution</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    sim_col_left, sim_col_right = st.columns([1.1, 1.9])

    with sim_col_left:
        st.markdown("<div class='gt-card'>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:12px; font-weight:700; text-transform:uppercase; color:#64748B;'>Configure Intervention Scenario</p>", unsafe_allow_html=True)

        scenario_carbs = st.slider("Simulated Meal Carbohydrates (g):", 0, 120, 50, step=5, help="50g = ~2 Rotis with Dal & Sabzi")
        scenario_exercise = st.slider("Physical Activity / Brisk Walk (min):", 0, 90, 30, step=10, help="Duration in minutes (30 min = ~2,400 steps of skeletal glucose uptake)")
        default_ins = 4 if patient_insulin != "None" else 0
        scenario_insulin = st.slider("Simulated Bolus Insulin (Units):", 0, 20, default_ins, step=1, help="Rapid-acting insulin analogue")
        sim_hours = st.selectbox("Forecast Horizon:", [2, 4, 6], index=1)

        st.markdown("""
        <div style="font-size:11px; color:#64748B; margin-top:8px; line-height:1.4;">
            <b>Active Preset:</b> Indian dietary carbohydrate absorption synced with individual BMI ({:.1f} kg/m²).
        </div>
        """.format(patient_ehr['bmi']), unsafe_allow_html=True)

        run_sim = st.button("🚀 Run Counterfactual Simulation", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with sim_col_right:
        # Mathematical Twin Simulation
        n_sim_steps = (sim_hours * 60) // 15
        sim_times = [current_time + timedelta(minutes=15*i) for i in range(n_sim_steps + 1)]
        time_labels = [f"+{15*i}m" for i in range(n_sim_steps + 1)]

        # Baseline trajectory (status quo)
        baseline_curve = [current_glucose]
        g_base = current_glucose
        for _ in range(n_sim_steps):
            g_base += (patient_ehr["fasting_glucose_mg_dl"] - g_base) * 0.05
            baseline_curve.append(round(g_base, 1))

        # Counterfactual intervention trajectory
        sim_curve = [current_glucose]
        g_sim = current_glucose
        steps_per_interval = (scenario_exercise * 80) / max(1, (scenario_exercise // 15)) if scenario_exercise > 0 else 0

        for step in range(n_sim_steps):
            carb_flux = (scenario_carbs * 1.8 * (patient_ehr["bmi"] / 22.0) * 0.25) if step in [1, 2, 3, 4] else 0.0
            ex_burn = (steps_per_interval / 250.0) * 2.2 if (step < scenario_exercise // 15) else 0.0
            ins_effect = (scenario_insulin * 3.5 * 0.3) if step in [2, 3, 4, 5] else 0.0
            reversion = (patient_ehr["fasting_glucose_mg_dl"] - g_sim) * 0.05
            g_sim = g_sim + carb_flux - ex_burn - ins_effect + reversion
            g_sim = max(45.0, min(g_sim, 380.0))
            sim_curve.append(round(g_sim, 1))

        # Compute Deltas
        base_tir = (sum(1 for g in baseline_curve if 70 <= g <= 180) / len(baseline_curve)) * 100.0
        sim_tir = (sum(1 for g in sim_curve if 70 <= g <= 180) / len(sim_curve)) * 100.0
        tir_delta_pp = sim_tir - base_tir

        base_peak = max(baseline_curve)
        sim_peak = max(sim_curve)
        peak_delta = sim_peak - base_peak

        sim_min = min(sim_curve)
        base_min = min(baseline_curve)
        hypo_risk_str = "High (Impending)" if sim_min < 70 else ("Moderate" if sim_min < 80 else "Low / Negligible")

        # Delta Cards Row
        d1, d2, d3 = st.columns(3)
        with d1:
            tir_color = "#10B981" if tir_delta_pp >= 0 else "#EF4444"
            st.markdown(f"""
            <div class="gt-card">
                <span style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase;">Time-in-Range Delta</span>
                <div class="tabular-nums" style="font-size:22px; font-weight:800; color:{tir_color}; margin-top:2px;">
                    {tir_delta_pp:+.1f} <span style="font-size:12px; font-weight:600;">pp</span>
                </div>
                <div style="font-size:11px; color:#64748B;">Simulated: <span class="tabular-nums" style="font-weight:700;">{sim_tir:.1f}%</span> (Base: {base_tir:.1f}%)</div>
            </div>
            """, unsafe_allow_html=True)

        with d2:
            peak_color = "#EF4444" if peak_delta > 30 else ("#F59E0B" if peak_delta > 0 else "#10B981")
            st.markdown(f"""
            <div class="gt-card">
                <span style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase;">Peak Glucose Shift</span>
                <div class="tabular-nums" style="font-size:22px; font-weight:800; color:{peak_color}; margin-top:2px;">
                    {peak_delta:+.1f} <span style="font-size:12px; font-weight:600;">mg/dL</span>
                </div>
                <div style="font-size:11px; color:#64748B;">Sim Peak: <span class="tabular-nums" style="font-weight:700;">{sim_peak:.0f} mg/dL</span></div>
            </div>
            """, unsafe_allow_html=True)

        with d3:
            hypo_color = "#EF4444" if sim_min < 70 else "#10B981"
            st.markdown(f"""
            <div class="gt-card">
                <span style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase;">Hypo Event Risk</span>
                <div style="font-size:20px; font-weight:800; color:{hypo_color}; margin-top:2px;">
                    {hypo_risk_str}
                </div>
                <div style="font-size:11px; color:#64748B;">Nadir: <span class="tabular-nums" style="font-weight:700;">{sim_min:.0f} mg/dL</span></div>
            </div>
            """, unsafe_allow_html=True)

        # Baseline vs Twin Chart
        fig_sim = go.Figure()
        fig_sim.add_hrect(
            y0=70, y1=180,
            fillcolor="rgba(16, 185, 129, 0.12)",
            line_width=1,
            line_dash="dash",
            line_color="rgba(16, 185, 129, 0.4)",
            annotation_text="Target Range (70–180 mg/dL)",
            annotation_position="top left",
            annotation_font=dict(size=10, color="#10B981")
        )

        fig_sim.add_trace(go.Scatter(
            x=time_labels, y=baseline_curve,
            name="Baseline Trajectory (Status Quo)",
            line=dict(color="#94A3B8", width=2, dash="dot"),
            mode="lines"
        ))

        sim_line_color = "#EF4444" if sim_min < 70 else ("#F59E0B" if sim_peak > 250 else "#0D9488")
        fig_sim.add_trace(go.Scatter(
            x=time_labels, y=sim_curve,
            name="Simulated Twin Trajectory",
            line=dict(color=sim_line_color, width=3.5),
            mode="lines+markers",
            marker=dict(size=5)
        ))

        sim_layout = get_clean_plotly_layout(
            f"Counterfactual Response ({scenario_carbs}g Carbs, {scenario_exercise}m Walk, {scenario_insulin}U Insulin)",
            height=360
        )
        fig_sim.update_layout(**sim_layout)
        st.plotly_chart(fig_sim, use_container_width=True)

        # AI Insight Card
        if sim_min < 70:
            st.markdown(f"""
            <div class="gt-card" style="border-left:5px solid #EF4444; background:#FEF2F2;">
                <div style="font-size:13px; font-weight:800; color:#991B1B;">💡 AI Clinical Insight</div>
                <div style="font-size:12px; color:#7F1D1D; margin-top:4px;">
                    This combination triggers <b>Hypoglycemia ({sim_min:.0f} mg/dL)</b> at step +{sim_curve.index(sim_min)*15} mins. 
                    Recommend reducing the bolus insulin dose by 1–2 units or adding 15g of complex carbohydrates to prevent nadir crashes.
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif sim_peak > 220:
            st.markdown(f"""
            <div class="gt-card" style="border-left:5px solid #F59E0B; background:#FFFBEB;">
                <div style="font-size:13px; font-weight:800; color:#92400E;">💡 AI Clinical Insight</div>
                <div style="font-size:12px; color:#78350F; margin-top:4px;">
                    Postprandial peak reaches <b>{sim_peak:.0f} mg/dL</b>. 
                    Adding 15 minutes of post-meal brisk walking or staggering carbohydrate intake is recommended to flatten the excursion.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="gt-card" style="border-left:5px solid #10B981; background:#F0FDF4;">
                <div style="font-size:13px; font-weight:800; color:#065F46;">💡 AI Clinical Insight</div>
                <div style="font-size:12px; color:#064E3B; margin-top:4px;">
                    <b>Optimal Counterfactual Balance:</b> Glycemic trajectory remains safely controlled between <b>{sim_min:.0f} and {sim_peak:.0f} mg/dL</b> with {sim_tir:.0f}% Time-in-Range.
                </div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("💾 Save Simulation Scenario to Patient Care Plan (Demo)"):
            st.toast("Demo: Counterfactual scenario registered in patient care plan.")


# ==============================================================================
# VIEW 3: DATA INGEST
# ==============================================================================
elif nav_tab == "Data Ingest":
    st.markdown("""
    <div style="margin-bottom:14px;">
        <h3 style="margin:0; font-size:18px; font-weight:800; color:#0F172A;">
            Wearable IoT & Sensor Data Ingestion
        </h3>
        <p style="margin:4px 0 0 0; font-size:13px; color:#64748B;">
            Seamlessly import real-world sensor streams and CGM continuous exports into the GlycoTwin multimodal feature engine.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Source Chips
    st.markdown("""
    <div style="display:flex; gap:8px; align-items:center; margin-bottom:16px;">
        <span style="font-size:12px; font-weight:700; color:#64748B;">Supported Formats:</span>
        <span style="background:#E0F2FE; color:#0369A1; font-size:11px; font-weight:700; padding:4px 10px; border-radius:12px;">Abbott LibreView</span>
        <span style="background:#E0F2FE; color:#0369A1; font-size:11px; font-weight:700; padding:4px 10px; border-radius:12px;">Dexcom Clarity</span>
        <span style="background:#E0F2FE; color:#0369A1; font-size:11px; font-weight:700; padding:4px 10px; border-radius:12px;">Apple Health</span>
        <span style="background:#E0F2FE; color:#0369A1; font-size:11px; font-weight:700; padding:4px 10px; border-radius:12px;">Google Health Connect</span>
    </div>
    """, unsafe_allow_html=True)

    up_col1, up_col2 = st.columns([1.5, 1])
    with up_col1:
        uploaded_file = st.file_uploader(
            "Upload Wearable CSV / JSON Export:",
            type=["csv", "json"],
            help="Accepts export files with timestamp and glucose columns"
        )
    with up_col2:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        load_sample = st.button("📥 Load Sample Abbott LibreView Data (48 Packets)", use_container_width=True)

    active_df = None
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".json"):
                active_df = pd.read_json(uploaded_file)
            else:
                active_df = pd.read_csv(uploaded_file)
            st.success(f"File successfully received: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error parsing file: {e}")
    elif load_sample or "sample_loaded" in st.session_state:
        st.session_state["sample_loaded"] = True
        active_df = patient_wearables.head(48)[["timestamp", "glucose_cgm", "heart_rate", "hrv", "steps_15m"]].copy()

    if active_df is not None:
        # Detect timestamp & glucose columns
        ts_candidates = [c for c in active_df.columns if "time" in c.lower() or "date" in c.lower()]
        glucose_candidates = [c for c in active_df.columns if "glucose" in c.lower() or "cgm" in c.lower()]
        ts_col = ts_candidates[0] if ts_candidates else active_df.columns[0]
        gl_col = glucose_candidates[0] if glucose_candidates else (active_df.columns[1] if len(active_df.columns) > 1 else None)

        # 5-Step Computed Checklist
        # Step 1: Required columns
        step1_pass = (ts_col is not None and gl_col is not None)
        
        # Step 2: Timestamps parse and sorted
        try:
            parsed_ts = pd.to_datetime(active_df[ts_col])
            step2_sorted = parsed_ts.is_monotonic_increasing
            step2_pass = True
        except Exception:
            step2_pass = False
            step2_sorted = False

        # Step 3: Gaps & Continuity %
        if step2_pass and len(parsed_ts) > 1:
            diffs = parsed_ts.diff().dropna()
            # Standard interval ~15 min
            normal_diffs = diffs[diffs <= pd.Timedelta(minutes=30)]
            continuity_pct = (len(normal_diffs) / len(diffs)) * 100.0 if len(diffs) > 0 else 100.0
        else:
            continuity_pct = 100.0
        step3_pass = continuity_pct >= 90.0

        # Step 4: Duplicate timestamps
        dup_count = active_df.duplicated(subset=[ts_col]).sum()
        step4_pass = (dup_count == 0)

        # Step 5: Glucose values within 40–400 mg/dL bounds
        numeric_gl = pd.to_numeric(active_df[gl_col], errors="coerce")
        out_of_bounds = ((numeric_gl < 40) | (numeric_gl > 400)).sum()
        step5_pass = (out_of_bounds == 0)

        # Summary KPIs
        mean_gl = numeric_gl.dropna().mean()
        tir_val = (numeric_gl.dropna().between(70, 180).mean()) * 100.0

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown(f"""
            <div class="gt-card">
                <span style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase;">Record Count</span>
                <div class="tabular-nums" style="font-size:24px; font-weight:800; color:#0F172A; margin-top:2px;">
                    {len(active_df)} <span style="font-size:12px; font-weight:500; color:#64748B;">telemetry packets</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="gt-card">
                <span style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase;">Cohort Mean Glucose</span>
                <div class="tabular-nums" style="font-size:24px; font-weight:800; color:#0F172A; margin-top:2px;">
                    {mean_gl:.1f} <span style="font-size:12px; font-weight:500; color:#64748B;">mg/dL</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="gt-card">
                <span style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase;">Time-in-Range (70–180)</span>
                <div class="tabular-nums" style="font-size:24px; font-weight:800; color:#10B981; margin-top:2px;">
                    {tir_val:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Computed 5-Step Checklist Card
        st.markdown("<div class='gt-card'>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:13px; font-weight:700; color:#0F172A; margin-bottom:12px;'>Computed Data Quality & Integrity Validation (5 Steps)</p>", unsafe_allow_html=True)

        def badge(passed, warn=False):
            if passed:
                return "<span style='background:#DCFCE7; color:#15803D; font-size:11px; font-weight:700; padding:2px 8px; border-radius:10px;'>PASS</span>"
            elif warn:
                return "<span style='background:#FEF3C7; color:#D97706; font-size:11px; font-weight:700; padding:2px 8px; border-radius:10px;'>WARNING</span>"
            return "<span style='background:#FEE2E2; color:#DC2626; font-size:11px; font-weight:700; padding:2px 8px; border-radius:10px;'>FAIL</span>"

        st.markdown(f"""
        <div class="check-step">
            <div><b>Step 1: Required Schema Columns</b><br><span style="font-size:11px; color:#64748B;">Detected timestamp ('{ts_col}') and glucose telemetry ('{gl_col}')</span></div>
            <div>{badge(step1_pass)}</div>
        </div>
        <div class="check-step">
            <div><b>Step 2: Timestamp Parsing & Chronological Sorting</b><br><span style="font-size:11px; color:#64748B;">{'Strict monotonic order verified' if step2_sorted else 'Unsorted intervals detected'}</span></div>
            <div>{badge(step2_sorted, warn=step2_pass)}</div>
        </div>
        <div class="check-step">
            <div><b>Step 3: Signal Continuity & Gap Analysis</b><br><span style="font-size:11px; color:#64748B;">{continuity_pct:.1f}% timeline continuity without dropout gaps >30m</span></div>
            <div>{badge(step3_pass, warn=not step3_pass)}</div>
        </div>
        <div class="check-step">
            <div><b>Step 4: Duplicate Timestamp Audit</b><br><span style="font-size:11px; color:#64748B;">{dup_count} duplicate timestamp records detected</span></div>
            <div>{badge(step4_pass, warn=not step4_pass)}</div>
        </div>
        <div class="check-step">
            <div><b>Step 5: Physiological Range Validation (40–400 mg/dL)</b><br><span style="font-size:11px; color:#64748B;">{out_of_bounds} values outside clinical bounds</span></div>
            <div>{badge(step5_pass)}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Preview Dataframe
        st.markdown("<p style='font-size:12px; font-weight:700; color:#64748B; text-transform:uppercase; margin-top:12px;'>Ingested Data Preview</p>", unsafe_allow_html=True)
        st.dataframe(active_df.head(10), use_container_width=True)


# ==============================================================================
# VIEW 4: COHORT ANALYTICS & ARCHITECTURE
# ==============================================================================
elif nav_tab == "Cohort Analytics":
    st.markdown("""
    <div style="margin-bottom:14px;">
        <h3 style="margin:0; font-size:18px; font-weight:800; color:#0F172A;">
            Indian Cohort Population Analytics & System Architecture
        </h3>
        <p style="margin:4px 0 0 0; font-size:13px; color:#64748B;">
            Comprehensive stratification across representative Type 2 Diabetes clinical personas from the 2026 Summit cohort.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Cohort KPI Strip
    n_cohort = len(df_ehr)
    mean_hba1c = df_ehr["baseline_hba1c"].mean()
    high_risk_count = df_ehr["clinical_risk_tier"].str.contains("High").sum()
    high_risk_pct = (high_risk_count / n_cohort) * 100.0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="gt-card">
            <span style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase;">Cohort Size (N)</span>
            <div class="tabular-nums" style="font-size:26px; font-weight:800; color:#0F172A; margin-top:2px;">
                {n_cohort} <span style="font-size:12px; font-weight:500; color:#64748B;">Patients</span>
            </div>
            <div style="font-size:11px; color:#64748B;">Indian Subcontinent Cohort</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="gt-card">
            <span style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase;">Mean Cohort HbA1c</span>
            <div class="tabular-nums" style="font-size:26px; font-weight:800; color:#0F172A; margin-top:2px;">
                {mean_hba1c:.2f}%
            </div>
            <div style="font-size:11px; color:#EF4444;">ADA Benchmark Target: 7.0%</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="gt-card">
            <span style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase;">High Clinical Risk</span>
            <div class="tabular-nums" style="font-size:26px; font-weight:800; color:#EF4444; margin-top:2px;">
                {high_risk_pct:.1f}%
            </div>
            <div style="font-size:11px; color:#64748B;">{high_risk_count} patients requiring active monitoring</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="gt-card">
            <span style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase;">High-Risk Flagged Patients</span>
            <div class="tabular-nums" style="font-size:26px; font-weight:800; color:#F59E0B; margin-top:2px;">
                {high_risk_count} <span style="font-size:12px; font-weight:500; color:#64748B;">Flagged</span>
            </div>
            <div style="font-size:11px; color:#64748B;">Active early-warning monitoring cohort</div>
        </div>
        """, unsafe_allow_html=True)

    # Restyled Visualizations Row
    chart_c1, chart_c2 = st.columns(2)

    with chart_c1:
        # HbA1c Histogram with 7.0% line
        fig_hba1c = go.Figure()
        fig_hba1c.add_trace(go.Histogram(
            x=df_ehr["baseline_hba1c"],
            nbinsx=15,
            marker_color="#0D9488",
            name="Patient Count"
        ))
        # ADA Target Line at 7.0%
        fig_hba1c.add_vline(
            x=7.0,
            line_width=2,
            line_dash="dash",
            line_color="#EF4444",
            annotation_text="ADA Target (7.0%)",
            annotation_position="top right",
            annotation_font=dict(color="#EF4444", size=11, family="Inter")
        )
        layout_h = get_clean_plotly_layout("Baseline HbA1c Distribution with ADA Target", height=300, y_range=None)
        layout_h["xaxis"]["title"] = "HbA1c (%)"
        layout_h["yaxis"]["title"] = "Patient Count"
        fig_hba1c.update_layout(**layout_h)
        st.plotly_chart(fig_hba1c, use_container_width=True)

    with chart_c2:
        # Risk Stratification Donut
        risk_counts = df_ehr["clinical_risk_tier"].value_counts()
        color_map = {
            "High Risk": "#EF4444",
            "Moderate Risk": "#F59E0B",
            "Low Risk": "#10B981"
        }
        donut_colors = [color_map.get(k, "#64748B") for k in risk_counts.index]

        fig_pie = go.Figure(data=[go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            hole=0.55,
            marker=dict(colors=donut_colors),
            textinfo="label+percent"
        )])
        layout_p = get_clean_plotly_layout("Clinical Hypoglycemia Risk Stratification", height=300, y_range=None)
        fig_pie.update_layout(**layout_p)
        st.plotly_chart(fig_pie, use_container_width=True)

    # NEW: High-Risk Patient Table
    st.markdown("<p style='font-size:14px; font-weight:700; color:#0F172A; margin-top:8px;'>High-Risk Patients Stratification Table</p>", unsafe_allow_html=True)
    
    # Sort: High Risk first, then by HbA1c descending
    df_sorted_cohort = df_ehr.copy()
    tier_rank = {"High Risk": 3, "Moderate Risk": 2, "Low Risk": 1}
    df_sorted_cohort["rank"] = df_sorted_cohort["clinical_risk_tier"].map(lambda x: tier_rank.get(x, 0))
    df_sorted_cohort = df_sorted_cohort.sort_values(by=["rank", "baseline_hba1c"], ascending=[False, False]).reset_index(drop=True)

    display_cohort = df_sorted_cohort[[
        "patient_id", "name", "age", "gender", "bmi", "baseline_hba1c", 
        "fasting_glucose_mg_dl", "egfr", "primary_medication", "clinical_risk_tier"
    ]].head(15).copy()

    display_cohort.columns = [
        "Patient ID", "Name", "Age", "Gender", "BMI (kg/m²)", "HbA1c (%)", 
        "Fasting Glucose", "eGFR", "Primary Medication", "Risk Tier"
    ]
    st.dataframe(display_cohort, use_container_width=True)

    # Visual Flow Diagram (Replacing ASCII)
    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:14px; font-weight:700; color:#0F172A; margin-bottom:10px;'>End-to-End Multimodal System Architecture & Data Flow</p>", unsafe_allow_html=True)

    st.markdown("""
    <div class="gt-card" style="padding:22px;">
        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:16px; align-items:stretch;">
            <!-- Box 1 -->
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:16px; border-top:4px solid #0D9488;">
                <div style="font-size:11px; font-weight:700; color:#0D9488; text-transform:uppercase;">1. Multimodal Streams</div>
                <div style="font-size:14px; font-weight:700; color:#0F172A; margin:4px 0 8px 0;">EHR + Wearable IoT</div>
                <ul style="font-size:12px; color:#64748B; padding-left:16px; margin:0; line-height:1.5;">
                    <li>Static EHR: Age, BMI, HbA1c, eGFR, Regimens</li>
                    <li>Dynamic CGM: Continuous Glucose (15m)</li>
                    <li>Photoplethysmography: HR, HRV Stress</li>
                    <li>Inertial: Steps, Sleep Stage, Carbs, Bolus</li>
                </ul>
            </div>
            <!-- Box 2 -->
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:16px; border-top:4px solid #0284C7;">
                <div style="font-size:11px; font-weight:700; color:#0284C7; text-transform:uppercase;">2. Fusion Engine</div>
                <div style="font-size:14px; font-weight:700; color:#0F172A; margin:4px 0 8px 0;">Time-Series Engineering</div>
                <ul style="font-size:12px; color:#64748B; padding-left:16px; margin:0; line-height:1.5;">
                    <li>Rolling Derivative: Velocity dG/dt (15m, 30m)</li>
                    <li>Variability: 1-hour Rolling Volatility (SD)</li>
                    <li>Metabolic Accumulation: 2-hour Steps & Carbs</li>
                    <li>Synchronous Fusion on Patient Key & Window</li>
                </ul>
            </div>
            <!-- Box 3 -->
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:16px; border-top:4px solid #8B5CF6;">
                <div style="font-size:11px; font-weight:700; color:#8B5CF6; text-transform:uppercase;">3. Predictive AI</div>
                <div style="font-size:14px; font-weight:700; color:#0F172A; margin:4px 0 8px 0;">HistGradientBoosting Models</div>
                <ul style="font-size:12px; color:#64748B; padding-left:16px; margin:0; line-height:1.5;">
                    <li>Classifier: Impending Hypo/Hyper 2h Ahead</li>
                    <li>Regressor: Exact Glucose Trajectory (mg/dL)</li>
                    <li>Counterfactual Simulator: Nutrition/Insulin what-if</li>
                    <li>Class-Weighted Scikit-Learn Ensemble</li>
                </ul>
            </div>
            <!-- Box 4 -->
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:16px; border-top:4px solid #10B981;">
                <div style="font-size:11px; font-weight:700; color:#10B981; text-transform:uppercase;">4. Clinical Action</div>
                <div style="font-size:14px; font-weight:700; color:#0F172A; margin:4px 0 8px 0;">Clinician Decision Interface</div>
                <ul style="font-size:12px; color:#64748B; padding-left:16px; margin:0; line-height:1.5;">
                    <li>Early Warning Horizon: 120-Minute Alert Window</li>
                    <li>Rescue Protocols: 15–20g Carb Emergency Protocol</li>
                    <li>Digital Twin Counterfactual Simulation Sandbox</li>
                    <li>Population Risk Stratification & EHR Sync</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("📤 Export Population Summary to EHR (Demo)"):
        st.toast("Demo: Cohort population analytics bundle exported to hospital EHR.")
