"""
GlycoTwin standalone dashboard builder.

Writes dashboard/index.html: one static page (HTML + CSS + JS) that replays the
SHIPPED model bundle's outputs for a handful of demo patients. Nothing is
predicted in the browser. Every risk score, alert and +2 h forecast on the page
is computed here, at build time, by models/glycotwin_model.joblib through the
helpers in models/train_fusion_model.py (load_data, compute_wearable_features,
prepare_fused_dataset, encode_features, apply_alert_thresholds), i.e. exactly
the encoding used in training and evaluation. Every performance number on the
page is read from models/metrics.json (written by models/evaluate.py); the
builder stops if that file does not match the shipped bundle.

Run from the repository root:
    python dashboard/build_standalone_dashboard.py
    cp dashboard/index.html public/index.html
    cp dashboard/plotly.min.js public/plotly.min.js

The page loads Plotly from the relative path "plotly.min.js" (this script copies
it next to index.html from the installed plotly package), so it works offline
from file:// and on Firebase Hosting, with no CDN or web-font dependency.
The output is deterministic: the same data, bundle and metrics give a
byte-identical file.

Synthetic data only. Research prototype, not a medical device.
"""

import json
import os
import shutil
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "models"))
from train_fusion_model import (CATEGORY_LEVELS, EHR_FEATURES, FEATURE_COLS, HGB_PARAMS,  # noqa: E402
                                HYPER_MG_DL, HYPO_MG_DL, WEARABLE_FEATURES, apply_alert_thresholds,
                                compute_wearable_features, encode_features, load_data,
                                prepare_fused_dataset)

OUT_HTML = os.path.join(ROOT, "dashboard", "index.html")
OUT_PLOTLY = os.path.join(ROOT, "dashboard", "plotly.min.js")
BUNDLE_PATH = os.path.join(ROOT, "models", "glycotwin_model.joblib")
METRICS_PATH = os.path.join(ROOT, "models", "metrics.json")

# Held-out patients with real events in the replay (plus one stable patient).
HELD_OUT_DEMO = ["IND-PAT-196", "IND-PAT-080", "IND-PAT-077", "IND-PAT-066", "IND-PAT-046"]
# One insulin-treated patient is added as an example. No insulin-treated persona is in the
# held-out split, so this one is always a training (in-sample) patient and is tagged as such.
# The preferred id is used only if the EHR really lists an insulin regimen for it; otherwise
# the first insulin-treated patient by id is used, so the page never mislabels a patient.
INSULIN_EXAMPLE_PREFERRED = "IND-PAT-012"

# Clinical reference thresholds (ADA glycaemic targets), not model outputs.
LEVEL2_HYPO_MG_DL = 54.0
TARGET_HIGH_MG_DL = 180.0
HORIZON_STEPS = 8  # 2 h of 15-min windows

# What-If sandbox: hand-set illustrative rules. They are NOT fitted to data and are NOT the
# ML model; the page says so and lists them.
SANDBOX_RULES = {
    "carb_rise_mg_dl_per_g": 1.2,
    "carb_peak_min": 60,
    "walk_drop_mg_dl_per_min": 0.8,
    "walk_multiplier_su_or_insulin": 1.5,
    "insulin_drop_mg_dl_per_unit": 25.0,
    "insulin_time_constant_min": 55,
    "floor_mg_dl": 40,
    "ceiling_mg_dl": 400,
}

EHR_NUMERIC = ["age", "bmi", "diabetes_duration_years", "baseline_hba1c", "fasting_glucose_mg_dl",
               "systolic_bp", "diastolic_bp", "egfr", "ldl_cholesterol_mg_dl", "triglycerides_mg_dl",
               "prior_hypo_episodes_12m", "tcf7l2_rs7903146_risk_alleles"]


def native(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def r1(x):
    return round(float(x), 1)


def r3(x):
    return round(float(x), 3)


def ehr_record(row):
    rec = {}
    for k, v in row.items():
        v = native(v)
        if k in EHR_NUMERIC:
            v = float(v)
            if v.is_integer() and k in ("age", "systolic_bp", "diastolic_bp", "fasting_glucose_mg_dl",
                                        "ldl_cholesterol_mg_dl", "triglycerides_mg_dl",
                                        "prior_hypo_episodes_12m", "tcf7l2_rs7903146_risk_alleles"):
                v = int(v)
        else:
            v = str(v)
        rec[k] = v
    return rec


def check_inputs(bundle, metrics):
    """Stops the build if the bundle, the helper module and metrics.json disagree."""
    if list(bundle["feature_cols"]) != list(FEATURE_COLS):
        raise SystemExit("Bundle feature_cols differ from train_fusion_model.FEATURE_COLS; retrain the model.")
    if bundle.get("category_levels") != CATEGORY_LEVELS:
        raise SystemExit("Bundle category_levels differ from train_fusion_model.CATEGORY_LEVELS; retrain the model.")
    if metrics["holdout"]["alert_thresholds"] != bundle["alert_thresholds"]:
        raise SystemExit("models/metrics.json thresholds differ from the bundle; run models/evaluate.py.")
    if abs(metrics["holdout"]["regressor"]["mae"] - bundle["metrics"]["holdout_mae"]) > 0.01:
        raise SystemExit("models/metrics.json does not match the shipped bundle; run models/evaluate.py.")


def first_out_of_range(cgm):
    for i, g in enumerate(cgm):
        if g < HYPO_MG_DL or g > HYPER_MG_DL:
            return i
    return None


def build_payload(ehr, fused, P, A, R, bundle, metrics):
    test_set = set(bundle["test_patients"])
    ehr_by_id = {r["patient_id"]: r for r in (ehr_record(x) for x in ehr.to_dict("records"))}

    insulin_ids = sorted(p for p, r in ehr_by_id.items() if r["insulin_regimen"] != "None")
    if INSULIN_EXAMPLE_PREFERRED in insulin_ids:
        insulin_example = INSULIN_EXAMPLE_PREFERRED
    else:
        insulin_example = insulin_ids[0] if insulin_ids else None
        print(f"Notice: {INSULIN_EXAMPLE_PREFERRED} has no insulin regimen in data/synthetic_ehr.csv; "
              f"using {insulin_example} as the insulin example instead.")

    demo_ids = list(HELD_OUT_DEMO) + ([insulin_example] if insulin_example else [])
    sleep_levels = CATEGORY_LEVELS["sleep_stage"]
    patients = []
    for pid in demo_ids:
        idx = np.where(fused["patient_id"].to_numpy() == pid)[0]
        g = fused.iloc[idx]
        rec = ehr_by_id[pid]
        held_out = pid in test_set
        if pid in HELD_OUT_DEMO and not held_out:
            print(f"Notice: {pid} is not in the bundle's held-out split; it is tagged as a training patient.")
        on_insulin = rec["insulin_regimen"] != "None"
        on_su = "Sulfonylurea" in rec["primary_medication"]
        cgm = [r1(v) for v in g["glucose_cgm"]]
        k = first_out_of_range(cgm)
        if k is not None:
            default_step = max(0, k - 2)
            default_reason = "Opened 30 min before this patient's first out-of-range CGM reading."
        else:
            default_step = len(cgm) - 1
            default_reason = "Opened at the latest window (no out-of-range readings in this replay)."
        tag = "held_out" if held_out else "training"
        tag_label = "Held-out patient (unseen in training)" if held_out else "Training patient (in-sample)"
        if pid == insulin_example:
            tag_label = "Training patient (insulin example)" if not held_out else "Held-out patient (insulin example)"
        patients.append({
            "id": pid,
            "name": rec["name"],
            "tag": tag,
            "tag_label": tag_label,
            "on_insulin": on_insulin,
            "on_su": on_su,
            "ehr": rec,
            "default_step": int(default_step),
            "default_reason": default_reason,
            "ts": [t.strftime("%Y-%m-%d %H:%M") for t in g["timestamp"]],
            "cgm": cgm,
            "hr": [int(v) for v in g["heart_rate"]],
            "hrv": [int(v) for v in g["hrv"]],
            "steps": [int(v) for v in g["steps_15m"]],
            "sleep": [sleep_levels.index(str(v)) for v in g["sleep_stage"]],
            "carbs": [int(v) for v in g["meal_carbs_g"]],
            "ins": [int(v) for v in g["insulin_units"]],
            "p_hypo": [r3(v) for v in P[idx, 1]],
            "p_hyper": [r3(v) for v in P[idx, 2]],
            "alert": [int(v) for v in A[idx]],
            "f2h": [r1(v) for v in R[idx]],
            "actual2h": [r1(v) for v in g["future_glucose_2h"]],
            "actual_event": [int(v) for v in g["adverse_event_2h"]],
        })

    # 2-hour risk board: every patient's latest window, scored by the shipped bundle.
    last = fused.groupby("patient_id", sort=True).tail(1).index.to_numpy()
    demo_index = {p["id"]: i for i, p in enumerate(patients)}
    board = []
    for i in last:
        row = fused.iloc[i]
        pid = row["patient_id"]
        board.append({
            "id": pid,
            "name": str(row["name"]),
            "tier": str(row["clinical_risk_tier"]),
            "med": str(row["primary_medication"]),
            "insulin": str(row["insulin_regimen"]),
            "split": "held_out" if pid in test_set else "training",
            "ts": row["timestamp"].strftime("%Y-%m-%d %H:%M"),
            "cgm": r1(row["glucose_cgm"]),
            "f2h": r1(R[i]),
            "ph": r3(P[i, 1]),
            "pr": r3(P[i, 2]),
            "alert": int(A[i]),
            "demo": demo_index.get(pid, -1),
        })
    # Patients already out of range now come first (below 70, then above 250), then the highest risk score.
    def now_rank(b):
        return 0 if b["cgm"] < HYPO_MG_DL else (1 if b["cgm"] > HYPER_MG_DL else 2)
    board.sort(key=lambda b: (now_rank(b), -max(b["ph"], b["pr"]), b["id"]))

    h = metrics["holdout"]
    data = {
        "patients": patients,
        "board": board,
        "ehr": [ehr_by_id[p] for p in sorted(ehr_by_id)],
        "sleep_levels": sleep_levels,
        "thresholds": bundle["alert_thresholds"],
        "holdout_mae": h["regressor"]["mae"],
        "clinical": {"hypo": HYPO_MG_DL, "level2": LEVEL2_HYPO_MG_DL, "target_high": TARGET_HIGH_MG_DL,
                     "hyper": HYPER_MG_DL},
        "horizon_steps": HORIZON_STEPS,
        "sandbox": SANDBOX_RULES,
        "test_patients": len(bundle["test_patients"]),
        "mc": {
            "mae": {"GlycoTwin model": h["regressor"]["mae"],
                    "Persistence (glucose stays the same)": h["baseline_persistence"]["mae"],
                    "Linear extrapolation of last 15 min": h["baseline_linear_extrapolation"]["mae"]},
            "clarke": {"GlycoTwin model": h["regressor"]["clarke_pct"],
                       "Persistence": h["baseline_persistence"]["clarke_pct"]},
        },
    }
    return data, insulin_ids, insulin_example


# ----------------------------------------------------------------------------------------------
# Model card (all numbers read from models/metrics.json)
# ----------------------------------------------------------------------------------------------

def pct(frac, nd=1):
    return f"{float(frac) * 100:.{nd}f}%"


def num(x, nd=1):
    return f"{float(x):.{nd}f}"


def thousands(x):
    return f"{int(x):,}"


def tile(label, value, sub):
    return (f'<div class="gt-card stat"><div class="stat-label">{label}</div>'
            f'<div class="stat-value tabular-nums">{value}</div><div class="stat-sub">{sub}</div></div>')


def change_word(new, old, lower_is_better, tol):
    if abs(new - old) <= tol:
        return "about the same"
    better = (new < old) if lower_is_better else (new > old)
    return "better" if better else "worse"


def model_card_html(metrics, bundle, ehr, insulin_ids):
    d = metrics["data"]
    h = metrics["holdout"]
    hc = h["classifier"]
    hypo, hyper = hc["hypo_lt70"], hc["hyper_gt250"]
    reg, pers, lin = h["regressor"], h["baseline_persistence"], h["baseline_linear_extrapolation"]
    eh, ey = h["episodes_hypo"], h["episodes_hyper"]
    rb = h["rule_baseline_events"]
    ne = h["classifier_not_already_in_event"]
    cv = metrics["grouped_cv_5fold"]
    ab = metrics["ablation_grouped_cv_5fold"]
    sw = h["ehr_swap_test"]
    lat = metrics["latency"]
    thr = h["alert_thresholds"]
    test_set = set(bundle["test_patients"])
    n_ins = len(insulin_ids)
    n_ins_test = len([p for p in insulin_ids if p in test_set])
    ab_sum = reg["clarke_pct"]["A"] + reg["clarke_pct"]["B"]

    def lead(ep):
        v = ep.get("median_lead_time_min")
        return "n/a" if v is None else f"{num(v, 0)} min"

    tiles = "".join([
        tile("2 h forecast MAE", f'{num(reg["mae"])} <span class="unit">mg/dL</span>',
             f'Persistence {num(pers["mae"])} · linear extrapolation {num(lin["mae"])} · R² {num(reg["r2"], 3)}'),
        tile("Clarke error grid", f'A {num(reg["clarke_pct"]["A"])}%',
             f'A+B {num(ab_sum)}% · D {num(reg["clarke_pct"]["D"], 2)}% · persistence A {num(pers["clarke_pct"]["A"])}%'),
        tile("Hypo &lt;70 within 2 h", f'{pct(hypo["sensitivity"])} <span class="unit">sensitivity</span>',
             f'Precision {pct(hypo["precision"])} · PR-AUC {num(hypo["pr_auc"], 3)} · ROC-AUC {num(hypo["roc_auc"], 3)}'),
        tile("Hyper &gt;250 within 2 h", f'{pct(hyper["sensitivity"])} <span class="unit">sensitivity</span>',
             f'Precision {pct(hyper["precision"])} · PR-AUC {num(hyper["pr_auc"], 3)} · ROC-AUC {num(hyper["roc_auc"], 3)}'),
        tile("Hypo episodes warned ahead", f'{eh["warned_15_to_120_min_ahead"]} <span class="unit">of {eh["episodes"]}</span>',
             f'Alert 15–120 min before the first reading &lt;70 · {num(eh["alarms_per_patient_day"], 2)} alarms per patient-day'),
        tile("Hyper episodes warned ahead", f'{ey["warned_15_to_120_min_ahead"]} <span class="unit">of {ey["episodes"]}</span>',
             f'Alert 15–120 min before the first reading &gt;250 · {num(ey["alarms_per_patient_day"], 2)} alarms per patient-day'),
    ])

    def det_row(name, m):
        return (f'<tr><td>{name}</td><td class="tabular-nums">{m["positives"]}</td>'
                f'<td class="tabular-nums">{pct(m["sensitivity"])}</td><td class="tabular-nums">{pct(m["specificity"])}</td>'
                f'<td class="tabular-nums">{pct(m["precision"])}</td>'
                f'<td class="tabular-nums">{num(m["roc_auc"], 3) if "roc_auc" in m else "–"}</td>'
                f'<td class="tabular-nums">{num(m["pr_auc"], 3) if "pr_auc" in m else "–"}</td></tr>')

    det_rows = "".join([
        det_row("Hypo &lt;70 · GlycoTwin model", hypo),
        det_row("Hypo &lt;70 · simple rule (CGM &lt;90 and falling)", rb["hypo_lt70"]),
        det_row("Hypo &lt;70 · model, windows with CGM 70–250 now", ne["hypo_lt70"]),
        det_row("Hyper &gt;250 · GlycoTwin model", hyper),
        det_row("Hyper &gt;250 · simple rule (CGM &gt;200 and rising)", rb["hyper_gt250"]),
        det_row("Hyper &gt;250 · model, windows with CGM 70–250 now", ne["hyper_gt250"]),
    ])

    ab_names = {
        "fused_all_features": "Fused EHR + wearables (all features)",
        "wearables_only": "Wearables only (no EHR)",
        "ehr_only_plus_hour": "EHR only + hour of day",
        "without_cgm": "Without CGM features",
        "without_hr_hrv_sleep": "Without heart rate, HRV and sleep",
        "without_history_and_genetics": "Without past hypos, family history, TCF7L2",
        "without_hour_of_day": "Without hour of day",
    }
    full = ab["fused_all_features"]
    ab_rows = ""
    for key, row in ab.items():
        dm = row["regressor_mae"] - full["regressor_mae"]
        ab_rows += (f'<tr><td>{ab_names.get(key, key)}</td><td class="tabular-nums">{row["features"]}</td>'
                    f'<td class="tabular-nums">{num(row["regressor_mae"], 2)}</td>'
                    f'<td class="tabular-nums">{"" if key == "fused_all_features" else f"{dm:+.2f}"}</td>'
                    f'<td class="tabular-nums">{num(row["hypo_pr_auc"], 3)}</td>'
                    f'<td class="tabular-nums">{num(row["hyper_pr_auc"], 3)}</td></tr>')

    wo = ab["wearables_only"]
    ehr_only = ab["ehr_only_plus_hour"]
    phys = ab["without_hr_hrv_sleep"]
    hour = ab["without_hour_of_day"]
    nocgm = ab["without_cgm"]
    fusion_sentence = (
        f'Adding the EHR to the wearable stream: forecast MAE {num(wo["regressor_mae"], 2)} → {num(full["regressor_mae"], 2)} mg/dL '
        f'({change_word(full["regressor_mae"], wo["regressor_mae"], True, 0.02)} with EHR), '
        f'hypo PR-AUC {num(wo["hypo_pr_auc"], 3)} → {num(full["hypo_pr_auc"], 3)} '
        f'({change_word(full["hypo_pr_auc"], wo["hypo_pr_auc"], False, 0.005)} with EHR), '
        f'hyper PR-AUC {num(wo["hyper_pr_auc"], 3)} → {num(full["hyper_pr_auc"], 3)} '
        f'({change_word(full["hyper_pr_auc"], wo["hyper_pr_auc"], False, 0.005)} with EHR). '
        f'The EHR alone (plus hour of day) gives MAE {num(ehr_only["regressor_mae"], 2)} mg/dL; without the CGM features MAE is '
        f'{num(nocgm["regressor_mae"], 2)} mg/dL'
        + (', so the CGM stream carries most of the short-term signal.'
           if nocgm["regressor_mae"] > full["regressor_mae"] + 1 and ehr_only["regressor_mae"] > full["regressor_mae"] + 1 else '.'))
    small_phys = (abs(phys["regressor_mae"] - full["regressor_mae"]) <= 0.1
                  and abs(phys["hypo_pr_auc"] - full["hypo_pr_auc"]) <= 0.01
                  and abs(phys["hyper_pr_auc"] - full["hyper_pr_auc"]) <= 0.01)
    phys_sentence = (
        f'Removing heart rate, HRV and sleep: MAE {num(phys["regressor_mae"], 2)} vs {num(full["regressor_mae"], 2)} mg/dL, '
        f'hypo PR-AUC {num(phys["hypo_pr_auc"], 3)} vs {num(full["hypo_pr_auc"], 3)}'
        + ('. In this synthetic data these signals add little independent information.' if small_phys else '.'))
    hour_sentence = (
        f'Without hour of day MAE is {num(hour["regressor_mae"], 2)} mg/dL (fused {num(full["regressor_mae"], 2)})'
        + (', so part of the skill comes from the daily routine built into the simulator.'
           if hour["regressor_mae"] > full["regressor_mae"] + 0.5 else '.'))

    swap_sentence = (
        f'EHR-swap test: scoring the held-out patients with another patient\'s EHR changes MAE {num(reg["mae"], 2)} → '
        f'{num(sw["regressor_mae"], 2)} mg/dL, hypo PR-AUC {num(hypo["pr_auc"], 3)} → {num(sw["hypo_pr_auc"], 3)} and '
        f'hyper PR-AUC {num(hyper["pr_auc"], 3)} → {num(sw["hyper_pr_auc"], 3)}.'
        + (' Worse scores with the wrong EHR show the model uses the static record.'
           if sw["regressor_mae"] > reg["mae"] and sw["hypo_pr_auc"] < hypo["pr_auc"] else ''))

    cvc = cv["classifier"]
    cv_rows = (
        f'<tr><td>2 h forecast</td><td class="tabular-nums">MAE {num(cv["regressor"]["mae"], 2)} mg/dL '
        f'(persistence {num(cv["baseline_persistence"]["mae"], 2)}) · R² {num(cv["regressor"]["r2"], 3)} · '
        f'Clarke A {num(cv["regressor"]["clarke_pct"]["A"])}%</td></tr>'
        f'<tr><td>Hypo &lt;70</td><td class="tabular-nums">sensitivity {pct(cvc["hypo_lt70"]["sensitivity"])} · '
        f'precision {pct(cvc["hypo_lt70"]["precision"])} · PR-AUC {num(cvc["hypo_lt70"]["pr_auc"], 3)}</td></tr>'
        f'<tr><td>Hyper &gt;250</td><td class="tabular-nums">sensitivity {pct(cvc["hyper_gt250"]["sensitivity"])} · '
        f'precision {pct(cvc["hyper_gt250"]["precision"])} · PR-AUC {num(cvc["hyper_gt250"]["pr_auc"], 3)}</td></tr>')

    lc = d["label_counts"]
    per_patient, rem = divmod(int(d["windows"]), int(d["patients"]))
    recorded = int(d["days_per_patient"]) * 24 * 60 // int(d["sampling_minutes"])
    if rem == 0 and recorded - per_patient == HORIZON_STEPS:
        windows_sentence = (
            f'{d["patients"]} synthetic patients × {thousands(per_patient)} labelled windows = {thousands(d["windows"])} windows. '
            f'Each patient has {d["days_per_patient"]} days of {d["sampling_minutes"]}-min data ({thousands(recorded)} readings); '
            f'the final 2 h ({HORIZON_STEPS} readings) has no 2-h label.')
    else:
        windows_sentence = (f'{d["patients"]} synthetic patients, {d["days_per_patient"]} days at {d["sampling_minutes"]}-min sampling: '
                            f'{thousands(d["windows"])} labelled windows.')
    feature_list = ", ".join(f"<code>{c}</code>" for c in WEARABLE_FEATURES)
    ehr_list = ", ".join(f"<code>{c}</code>" for c in EHR_FEATURES)
    params = ", ".join(f"{k}={v}" for k, v in HGB_PARAMS.items())
    min_egfr = float(ehr["egfr"].min())
    egfr_note = (f'<li>No persona has eGFR below 60 (lowest {num(min_egfr)} mL/min/1.73m²), so the simulator\'s eGFR &lt;60 effects '
                 f'(slower insulin and sulfonylurea clearance, higher hypo propensity) are never exercised in this data.</li>' if min_egfr >= 60 else '')
    # metrics.json notes are appended only if they add something the fixed bullets below do not already say.
    covered = ("synthetic", "calibrat")
    notes = "".join(f"<li>{n}</li>" for n in metrics.get("notes", [])
                    if not any(w in n.lower() for w in covered))

    return f"""
<div class="mc-intro gt-card">
  <p><b>Source.</b> Every number on this tab is read at build time from <code>models/metrics.json</code>, written by
  <code>models/evaluate.py</code> from the shipped bundle <code>models/glycotwin_model.joblib</code>. Held-out results use
  {h["test_patients"]} patients ({thousands(h["windows"])} windows) that the model never saw in training.</p>
  <p><b>Risk scores are not calibrated probabilities.</b> The classifier is trained with <code>class_weight="balanced"</code>,
  which inflates scores for rare events. The page shows them as 0–1 risk scores next to the alert thresholds.</p>
</div>
<div class="grid-3 mc-tiles">{tiles}</div>
<div class="gt-card note-card">
  <b>False alarms.</b> Most alarms are not followed by an event: {pct(eh["false_alarm_fraction"], 0)} of hypo alarms and
  {pct(ey["false_alarm_fraction"], 0)} of hyper alarms (alarm = start of a run of alert windows). That is about
  {num(eh["alarms_per_patient_day"], 2)} hypo and {num(ey["alarms_per_patient_day"], 2)} hyper alarms per patient-day.
  Median warning lead time: {lead(eh)} (hypo), {lead(ey)} (hyper).
</div>
<div class="grid-2">
  <div class="gt-card"><div class="card-title">2 h forecast error on held-out patients (lower is better)</div><div id="mcMaeChart" class="chart" style="height:260px;"></div></div>
  <div class="gt-card"><div class="card-title">Clarke error grid zones, % of held-out windows</div><div id="mcClarkeChart" class="chart" style="height:260px;"></div></div>
</div>
<div class="gt-card">
  <div class="card-title">Event detection on held-out patients (alert thresholds: hypo ≥ {thr["hypo"]}, hyper ≥ {thr["hyper"]})</div>
  <div class="table-wrap"><table class="data-table">
    <thead><tr><th>Event · method</th><th>Positive windows</th><th>Sensitivity</th><th>Specificity</th><th>Precision</th><th>ROC-AUC</th><th>PR-AUC</th></tr></thead>
    <tbody>{det_rows}</tbody></table></div>
  <p class="small muted">A window is positive when any CGM reading in the next 2 h is &lt;70 (hypo) or &gt;250 mg/dL (hyper).
  The "windows with CGM 70–250 now" rows keep only windows whose current reading is in 70–250 mg/dL, so the event has not started yet.</p>
</div>
<div class="grid-2">
  <div class="gt-card">
    <div class="card-title">5-fold patient-grouped cross-validation (all {d["patients"]} patients)</div>
    <div class="table-wrap"><table class="data-table"><tbody>{cv_rows}</tbody></table></div>
    <p class="small muted">{cv["note"][:1].upper() + cv["note"][1:]}.</p>
  </div>
  <div class="gt-card">
    <div class="card-title">Does the model use the EHR?</div>
    <p class="small">{swap_sentence}</p>
  </div>
</div>
<div class="gt-card">
  <div class="card-title">Feature ablation (5-fold patient-grouped CV)</div>
  <div class="table-wrap"><table class="data-table">
    <thead><tr><th>Feature set</th><th>Features</th><th>Forecast MAE</th><th>Δ MAE vs fused</th><th>Hypo PR-AUC</th><th>Hyper PR-AUC</th></tr></thead>
    <tbody>{ab_rows}</tbody></table></div>
  <ul class="small bullets"><li>{fusion_sentence}</li><li>{phys_sentence}</li><li>{hour_sentence}</li></ul>
</div>
<div class="grid-2">
  <div class="gt-card">
    <div class="card-title">Data</div>
    <ul class="small bullets">
      <li>{windows_sentence}</li>
      <li>Labels: {thousands(lc["0"])} no event · {thousands(lc["1"])} hypo &lt;70 · {thousands(lc["2"])} hyper &gt;250 (next 2 h).</li>
      <li>CGM readings &lt;70 mg/dL: {thousands(d["cgm_readings_lt70"])} (of which {d["cgm_readings_lt54"]} &lt;54, ADA Level 2), in {d["patients_with_any_lt70"]} of {d["patients"]} patients.</li>
      <li>Insulin-treated personas: {n_ins} of {len(ehr)}; {n_ins_test} of them in the held-out split.</li>
      <li>{d["features"]["total"]} model inputs = {d["features"]["wearable"]} wearable + {d["features"]["ehr"]} EHR.</li>
    </ul>
  </div>
  <div class="gt-card">
    <div class="card-title">Model</div>
    <ul class="small bullets">
      <li>scikit-learn <code>HistGradientBoostingClassifier</code> (3 classes: none / hypo / hyper in next 2 h) and
      <code>HistGradientBoostingRegressor</code> (glucose exactly 2 h ahead); {params}; classifier uses class_weight="balanced".</li>
      <li>Alert rule: hypo if hypo score ≥ {thr["hypo"]}, else hyper if hyper score ≥ {thr["hyper"]}. Thresholds chosen by F2 score
      on out-of-fold predictions of the training patients only.</li>
      <li>Latency: {num(lat["single_window_ms_median"], 1)} ms median per single window (p95 {num(lat["single_window_ms_p95"], 1)} ms),
      {num(lat["batched_us_per_window"], 1)} µs per window batched; CPU only, measured by <code>models/evaluate.py</code> on one machine
      (the figures vary slightly from run to run). This page does not run the model:
      it replays outputs computed when the page was built.</li>
    </ul>
  </div>
</div>
<div class="gt-card">
  <div class="card-title">Model inputs</div>
  <p class="small"><b>Wearable ({len(WEARABLE_FEATURES)}):</b> {feature_list}</p>
  <p class="small"><b>EHR ({len(EHR_FEATURES)}):</b> {ehr_list}</p>
</div>
<div class="gt-card limits">
  <div class="card-title">Limitations</div>
  <ul class="small bullets">
    <li>Synthetic data only (<code>data/generate_datasets.py</code>). Simulator parameters are illustrative; population ranges are
    loosely informed by published Indian literature and are not calibrated to any dataset. Results do not transfer to real patients.</li>
    <li>Risk scores are uncalibrated model outputs, not the probability of an event.</li>
    <li>Precision is modest: {pct(hypo["precision"])} for hypo and {pct(hyper["precision"])} for hyper alerts on held-out windows;
    most alarms are not followed by an event.</li>
    <li>No insulin-treated patient is in the held-out set ({n_ins_test} of {n_ins}), so performance for insulin users is untested.</li>
    <li>The forecast is a single value exactly 2 h ahead, not a trajectory; the alert covers any reading in the next 2 h, so the two can differ.</li>
    {egfr_note}
    <li>No clinical validation. Research prototype, not a medical device; not for diagnosis or dosing decisions.</li>
    {notes}
  </ul>
</div>
"""


# ----------------------------------------------------------------------------------------------
# Page template (placeholders: %%DATA_JSON%%, %%MODEL_CARD%%)
# ----------------------------------------------------------------------------------------------

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GlycoTwin Prototype</title>
<meta name="description" content="GlycoTwin: research prototype of a Type 2 diabetes digital twin that fuses a synthetic EHR with synthetic wearable data and replays a scikit-learn model's 2-hour hypo/hyperglycaemia risk alerts. Synthetic data only; not a medical device.">
<script src="plotly.min.js"></script>
<style>
:root {
  --navy: #0F172A; --navy-2: #1E293B; --navy-3: #334155;
  --teal: #0D9488; --teal-2: #0F766E; --teal-bg: #CCFBF1;
  --ink: #0F172A; --ink-2: #334155; --muted: #64748B; --line: #E2E8F0;
  --bg: #F8FAFC; --card: #FFFFFF;
  --red: #DC2626; --red-dark: #991B1B; --red-bg: #FEF2F2;
  --amber: #B45309; --amber-bg: #FFFBEB;
  --orange: #C2410C; --orange-bg: #FFF7ED;
  --green: #047857; --green-bg: #ECFDF5;
  --blue: #1D4ED8; --blue-bg: #EFF6FF;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body { background: var(--bg); color: var(--ink); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; font-size: 14px; line-height: 1.5; }
body { overflow-x: hidden; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; background: #F1F5F9; padding: 0 4px; border-radius: 4px; overflow-wrap: anywhere; }
.tabular-nums { font-variant-numeric: tabular-nums; }
.small { font-size: 12px; } .muted { color: var(--muted); }
[hidden] { display: none !important; }

.disclaimer { background: #111827; color: #F9FAFB; font-size: 12px; padding: 6px 16px; text-align: center; position: sticky; top: 0; z-index: 60; }
.disclaimer span { color: #CBD5E1; margin-left: 6px; }

.app-layout { display: flex; min-height: 100vh; }
.sidebar { width: 280px; min-width: 280px; background: var(--navy); color: #CBD5E1; padding: 18px 16px; position: fixed; top: 30px; bottom: 0; left: 0; overflow-y: auto; z-index: 50; }
.sidebar-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 16px; }
.brand { display: flex; align-items: center; gap: 10px; }
.brand-mark { width: 36px; height: 36px; border-radius: 10px; background: var(--teal); display: flex; align-items: center; justify-content: center; color: #fff; font-weight: 800; font-size: 15px; }
.brand-name { font-size: 18px; font-weight: 800; color: #F8FAFC; letter-spacing: -0.02em; line-height: 1.1; }
.brand-sub { font-size: 11px; color: #94A3B8; }
.menu-btn { display: none; background: var(--navy-2); color: #F8FAFC; border: 1px solid var(--navy-3); border-radius: 8px; padding: 6px 12px; font-size: 13px; font-weight: 600; cursor: pointer; }
.side-label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #94A3B8; margin: 6px 0 8px; }
.nav-btn { display: block; width: 100%; background: var(--navy-2); border: 1px solid var(--navy-3); border-radius: 8px; padding: 9px 12px; margin-bottom: 6px; color: #CBD5E1; font-size: 13px; font-weight: 500; text-align: left; cursor: pointer; }
.nav-btn:hover { border-color: var(--teal); }
.nav-btn.active { background: var(--teal); border-color: var(--teal); color: #fff; font-weight: 700; }
.patient-select { width: 100%; background: var(--navy-2); border: 1px solid var(--navy-3); border-radius: 8px; padding: 8px 10px; color: #F8FAFC; font-size: 12px; margin-bottom: 10px; }
.p360 { background: var(--navy-2); border: 1px solid var(--navy-3); border-radius: 12px; padding: 12px; font-size: 12px; }
.p360-row { display: flex; justify-content: space-between; gap: 8px; padding: 4px 0; border-bottom: 1px solid var(--navy-3); }
.p360-row:last-child { border-bottom: none; }
.p360-row.stack { flex-direction: column; gap: 1px; }
.p360-k { color: #94A3B8; } .p360-v { color: #F8FAFC; font-weight: 600; text-align: right; }
.p360-row.stack .p360-v { text-align: left; font-size: 11px; }
.side-foot { margin-top: 14px; font-size: 11px; color: #94A3B8; line-height: 1.45; }

.main-canvas { margin-left: 280px; flex: 1; min-width: 0; padding: 20px 26px 40px; }
.section-head h2 { font-size: 19px; font-weight: 800; letter-spacing: -0.01em; }
.section-head p { font-size: 13px; color: var(--muted); margin-top: 2px; }
.section-head { margin-bottom: 14px; }

.gt-card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; box-shadow: 0 1px 2px rgba(15,23,42,0.05); padding: 14px 16px; margin-bottom: 14px; min-width: 0; }
.card-title { font-size: 13px; font-weight: 700; color: var(--ink); margin-bottom: 8px; }
.grid-2, .grid-3, .grid-4, .grid-hero, .grid-sim { display: grid; gap: 14px; }
.grid-2 { grid-template-columns: 1fr 1fr; } .grid-3 { grid-template-columns: repeat(3, 1fr); } .grid-4 { grid-template-columns: repeat(4, 1fr); }
.grid-hero { grid-template-columns: 1.3fr 2fr; } .grid-sim { grid-template-columns: 1fr 2fr; }
.grid-2 > *, .grid-3 > *, .grid-4 > *, .grid-hero > *, .grid-sim > * { min-width: 0; }
.grid-2, .grid-3, .grid-4 { margin-bottom: 14px; }
.grid-2 > .gt-card, .grid-3 > .gt-card, .grid-4 > .gt-card { margin-bottom: 0; }
.grid-live { display: grid; grid-template-columns: 1fr 1.25fr; grid-template-rows: auto 1fr; grid-template-areas: "hero alert" "kpi alert"; gap: 14px; margin-bottom: 14px; align-items: start; }
.grid-live > * { min-width: 0; }
.area-hero { grid-area: hero; margin-bottom: 0 !important; } .area-kpi { grid-area: kpi; } .area-alert { grid-area: alert; }
.area-alert .alert-card { margin: 0; }
.stat .stat-label { font-size: 11px; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: 0.03em; }
.stat .stat-value { font-size: 24px; font-weight: 800; color: var(--ink); margin-top: 2px; line-height: 1.2; }
.stat .stat-sub { font-size: 12px; color: var(--muted); margin-top: 2px; }
.unit { font-size: 12px; font-weight: 500; color: var(--muted); }

.replay { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 12px 16px; margin-bottom: 14px; }
.replay-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 10px; flex-wrap: wrap; }
.pt-name { font-size: 18px; font-weight: 800; }
.chip { display: inline-block; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 10px; vertical-align: middle; }
.chip-held { background: var(--blue-bg); color: var(--blue); border: 1px solid #BFDBFE; }
.chip-train { background: #F1F5F9; color: var(--ink-2); border: 1px solid #CBD5E1; }
.pill { font-size: 12px; font-weight: 700; padding: 4px 10px; border-radius: 12px; white-space: nowrap; }
.ts-line { font-size: 13px; color: var(--muted); margin-top: 4px; }
.ts-line b { color: var(--ink); }
.replay input[type=range] { width: 100%; margin: 10px 0 6px; accent-color: var(--teal); }
.btn-row { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.btn { background: #fff; border: 1px solid #CBD5E1; color: var(--ink); border-radius: 8px; padding: 5px 10px; font-size: 12px; font-weight: 600; cursor: pointer; }
.btn:hover { border-color: var(--teal); color: var(--teal-2); }
.btn-teal { background: var(--teal); color: #fff; border-color: var(--teal); }
.btn-teal:hover { background: var(--teal-2); color: #fff; }
.start-note { font-size: 11px; color: var(--muted); margin-left: auto; max-width: 520px; text-align: right; }

.hero-top { display: flex; justify-content: space-between; align-items: center; gap: 8px; flex-wrap: wrap; }
.hero-label { font-size: 11px; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
.hero-val { display: flex; align-items: baseline; gap: 8px; margin-top: 8px; flex-wrap: wrap; }
.hero-num { font-size: 44px; font-weight: 800; line-height: 1; }
.hero-roc { margin-left: auto; font-size: 14px; font-weight: 700; color: var(--ink-2); }
.range-track { position: relative; height: 10px; border-radius: 5px; margin: 14px 0 4px;
  background: linear-gradient(to right, #991B1B 0%, #991B1B 3.89%, #DC2626 3.89%, #DC2626 8.33%, #10B981 8.33%, #10B981 38.89%, #F59E0B 38.89%, #F59E0B 58.33%, #DC2626 58.33%, #DC2626 100%); }
.range-needle { position: absolute; top: -5px; width: 6px; height: 20px; background: var(--navy); border: 2px solid #fff; border-radius: 3px; transform: translateX(-50%); box-shadow: 0 1px 3px rgba(0,0,0,0.4); }
.range-labels { position: relative; height: 30px; font-size: 10px; color: var(--muted); }
.range-labels span { position: absolute; transform: translateX(-50%); white-space: nowrap; }
.range-labels .r2 { top: 14px; }
.kpi-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.kpi-grid .gt-card { margin-bottom: 0; }
.kpi-label { font-size: 11px; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: 0.03em; }
.kpi-val { font-size: 22px; font-weight: 800; margin-top: 2px; line-height: 1.25; }
.kpi-sub { font-size: 12px; color: var(--muted); margin-top: 2px; }

.alert-card { border-radius: 12px; padding: 14px 18px; margin: 14px 0; border-left: 6px solid; }
.alert-card .a-kicker { font-size: 11px; font-weight: 800; letter-spacing: 0.05em; text-transform: uppercase; }
.alert-card .a-title { font-size: 19px; font-weight: 800; margin-top: 2px; }
.alert-card .a-body { font-size: 13px; margin-top: 8px; color: var(--ink-2); }
.alert-grid { display: grid; grid-template-columns: 1fr; gap: 10px; margin-top: 10px; }
.alert-grid > * { min-width: 0; }
.scorebox { background: rgba(255,255,255,0.75); border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px; font-size: 12px; }
.scorebox .row { display: flex; justify-content: space-between; gap: 8px; padding: 2px 0; }
.scorebar { height: 6px; background: #E2E8F0; border-radius: 3px; position: relative; margin: 2px 0 6px; }
.scorebar i { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 3px; }
.scorebar b { position: absolute; top: -3px; width: 2px; height: 12px; background: var(--ink); }
.actualbox { background: #F8FAFC; border: 1px dashed #94A3B8; border-radius: 10px; padding: 10px 12px; font-size: 12px; color: var(--ink-2); }
.st-hypo { background: var(--red-bg); border-color: var(--red); } .st-hypo .a-kicker, .st-hypo .a-title { color: var(--red-dark); }
.st-hyper { background: var(--orange-bg); border-color: var(--orange); } .st-hyper .a-kicker, .st-hyper .a-title { color: var(--orange); }
.st-watch { background: var(--amber-bg); border-color: #F59E0B; } .st-watch .a-kicker, .st-watch .a-title { color: var(--amber); }
.st-ok { background: var(--green-bg); border-color: #10B981; } .st-ok .a-kicker, .st-ok .a-title { color: var(--green); }
.rescue { margin-top: 8px; background: #fff; border: 1px solid #FCA5A5; border-radius: 8px; padding: 8px 10px; color: var(--red-dark); font-size: 13px; }

.chart-head { display: flex; justify-content: space-between; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 4px; }
.chart-head .stats { display: flex; gap: 14px; flex-wrap: wrap; font-size: 12px; color: var(--muted); }
.chart-head .stats b { color: var(--ink); }
.chart { width: 100%; }
.chart-fallback { padding: 20px; color: var(--muted); font-size: 12px; text-align: center; border: 1px dashed var(--line); border-radius: 8px; }
label.toggle { font-size: 12px; color: var(--ink-2); display: inline-flex; gap: 6px; align-items: center; cursor: pointer; }

.badge-sandbox { display: inline-block; background: #FEF3C7; color: #92400E; border: 1px solid #FCD34D; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 10px; }
.ctrl { margin-bottom: 14px; }
.ctrl label { font-size: 12px; font-weight: 700; }
.ctrl .ends { display: flex; justify-content: space-between; font-size: 11px; color: var(--muted); }
.ctrl .ends b { color: var(--teal-2); }
.ctrl input[type=range] { width: 100%; accent-color: var(--teal); }
.ctrl .hint { font-size: 11px; color: var(--muted); }
.ctx { font-size: 12px; background: #F8FAFC; border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; margin-bottom: 12px; }
details.rules summary { cursor: pointer; font-size: 12px; font-weight: 700; color: var(--ink-2); }
details.rules ul { margin-top: 6px; }
.bullets { padding-left: 18px; } .bullets li { margin-bottom: 4px; }

.drop { border: 2px dashed #CBD5E1; border-radius: 12px; padding: 20px; text-align: center; background: #fff; cursor: pointer; display: block; }
.drop:hover, .drop.drag { border-color: var(--teal); background: #F0FDFA; }
.drop .t1 { font-size: 14px; font-weight: 700; }
.drop .t2 { font-size: 12px; color: var(--muted); margin-top: 2px; }
.check { display: flex; justify-content: space-between; align-items: flex-start; gap: 10px; padding: 10px 12px; border: 1px solid var(--line); border-radius: 8px; margin-bottom: 8px; background: #F8FAFC; font-size: 13px; }
.check .d { font-size: 12px; color: var(--muted); margin-top: 2px; }
.b-pass, .b-warn, .b-fail { font-size: 11px; font-weight: 800; padding: 2px 8px; border-radius: 10px; white-space: nowrap; }
.b-pass { background: #DCFCE7; color: #166534; } .b-warn { background: #FEF3C7; color: #92400E; } .b-fail { background: #FEE2E2; color: #991B1B; }
.verdict { border-radius: 10px; padding: 10px 14px; font-size: 13px; font-weight: 700; margin-bottom: 12px; }

.table-wrap { width: 100%; overflow-x: auto; -webkit-overflow-scrolling: touch; }
.data-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.data-table th { background: #F8FAFC; text-align: left; font-size: 11px; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: 0.03em; padding: 8px; border-bottom: 2px solid var(--line); white-space: nowrap; }
.data-table td { padding: 7px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
.data-table tr.clickable { cursor: pointer; } .data-table tr.clickable:hover td { background: #F0FDFA; }
.tier { font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 10px; white-space: nowrap; }
.tier-High { background: #FEE2E2; color: #991B1B; } .tier-Moderate { background: #FEF3C7; color: #92400E; } .tier-Low { background: #DCFCE7; color: #166534; }
.now-tag { font-size: 10px; font-weight: 800; padding: 2px 7px; border-radius: 10px; white-space: nowrap; }
.now-low { background: #FEE2E2; color: #991B1B; border: 1px solid #FCA5A5; } .now-high { background: #FFEDD5; color: #9A3412; border: 1px solid #FDBA74; }
.al-1 { color: var(--red-dark); font-weight: 800; } .al-2 { color: var(--orange); font-weight: 800; } .al-0 { color: var(--muted); }

.note-card { font-size: 13px; background: var(--amber-bg); border-color: #FCD34D; }
.mc-intro p { font-size: 13px; margin-bottom: 6px; } .mc-intro p:last-child { margin-bottom: 0; }
.limits { border-left: 4px solid var(--amber); }

.page-foot { font-size: 12px; color: var(--muted); border-top: 1px solid var(--line); padding-top: 12px; margin-top: 10px; }
.page-foot p { margin-bottom: 4px; }
.toast { position: fixed; bottom: 18px; right: 18px; max-width: calc(100vw - 36px); background: var(--navy); color: #F8FAFC; padding: 10px 16px; border-radius: 10px; font-size: 13px; box-shadow: 0 8px 24px rgba(0,0,0,0.3); transform: translateY(90px); opacity: 0; transition: all 0.25s ease; z-index: 99; }
.toast.show { transform: translateY(0); opacity: 1; }

@media (max-width: 1200px) {
  .grid-4 { grid-template-columns: repeat(2, 1fr); }
  .grid-hero, .grid-sim { grid-template-columns: 1fr; }
  .grid-live { grid-template-columns: 1fr; grid-template-rows: auto; grid-template-areas: "hero" "alert" "kpi"; }
  .start-note { margin-left: 0; text-align: left; max-width: none; width: 100%; }
}
@media (max-width: 900px) {
  .disclaimer { position: static; }
  .app-layout { display: block; }
  .sidebar { position: static; width: 100%; min-width: 0; padding: 10px 12px; }
  .sidebar-head { margin-bottom: 0; }
  .menu-btn { display: inline-block; }
  .sidebar-body { display: none; margin-top: 12px; }
  .sidebar.open .sidebar-body { display: block; }
  .main-canvas { margin-left: 0; padding: 12px 12px 30px; }
  .grid-2, .grid-3, .grid-4, .grid-hero, .grid-sim, .alert-grid { grid-template-columns: 1fr; }
  .kpi-grid { grid-template-columns: 1fr 1fr; gap: 10px; }
  .kpi-val { font-size: 18px; }
  .hero-num { font-size: 38px; }
  .pt-name { font-size: 16px; }
  .alert-card .a-title { font-size: 16px; }
  .alert-card { padding: 12px 14px; }
  .gt-card { padding: 12px; }
  .section-head h2 { font-size: 17px; }
}
@media (max-width: 420px) {
  .kpi-grid .gt-card { padding: 10px; }
  .kpi-val { font-size: 16px; }
  .kpi-sub { font-size: 11px; }
}
</style>
</head>
<body>
<div class="disclaimer" role="note"><strong>Synthetic data only. Research prototype, not a medical device.</strong><span>Not for diagnosis or dosing decisions.</span></div>
<div class="app-layout">
  <aside class="sidebar" id="sidebar">
    <div class="sidebar-head">
      <div class="brand">
        <div class="brand-mark">GT</div>
        <div><div class="brand-name">GlycoTwin</div><div class="brand-sub">T2D digital twin · research prototype</div></div>
      </div>
      <button class="menu-btn" id="menuBtn" aria-expanded="false" aria-controls="sidebarBody" onclick="toggleMenu()">Menu</button>
    </div>
    <div class="sidebar-body" id="sidebarBody">
      <div class="side-label">Views</div>
      <button class="nav-btn active" id="nav0" onclick="switchSection(0)">Live Monitor (replay)</button>
      <button class="nav-btn" id="nav1" onclick="switchSection(1)">What-If Sandbox</button>
      <button class="nav-btn" id="nav2" onclick="switchSection(2)">Data Ingest</button>
      <button class="nav-btn" id="nav3" onclick="switchSection(3)">Cohort</button>
      <button class="nav-btn" id="nav4" onclick="switchSection(4)">Model Card</button>
      <div class="side-label" style="margin-top:14px;">Patient (EHR + wearables)</div>
      <select id="patientSelect" class="patient-select" onchange="switchPatient(parseInt(this.value, 10))" aria-label="Select demo patient"></select>
      <div id="patientCard" class="p360"></div>
      <div class="side-foot">Model: scikit-learn HistGradientBoosting classifier + regressor. Outputs were computed in Python when this page was built and are replayed here.</div>
    </div>
  </aside>

  <main class="main-canvas">
    <div class="replay" id="replayBar">
      <div class="replay-top">
        <div>
          <div class="pt-name" id="ptName"></div>
          <div class="ts-line">Synthetic replay time: <b class="tabular-nums" id="tsNow"></b> <span id="dayNow"></span></div>
        </div>
        <div class="pill" id="alertPill"></div>
      </div>
      <input id="timeSlider" type="range" min="0" max="1" value="0" step="1" oninput="setStep(parseInt(this.value, 10))" aria-label="Replay time (15-minute windows)">
      <div class="btn-row">
        <button class="btn" onclick="moveStep(-4)">−1 h</button>
        <button class="btn" onclick="moveStep(-1)">−15 min</button>
        <button class="btn" onclick="moveStep(1)">+15 min</button>
        <button class="btn" onclick="moveStep(4)">+1 h</button>
        <button class="btn" onclick="jumpAlert(-1)">Previous alert</button>
        <button class="btn" onclick="jumpAlert(1)">Next alert</button>
        <button class="btn" onclick="jumpEvent()" title="Uses the replay's actual readings, which the model does not see">30 min before next out-of-range reading</button>
        <div class="start-note" id="startNote"></div>
      </div>
    </div>

    <!-- LIVE MONITOR -->
    <section id="sec0">
      <div class="grid-live">
        <div class="gt-card area-hero" id="glucoseHero"></div>
        <div class="area-alert" id="alertCard"></div>
        <div class="kpi-grid area-kpi" id="kpiGrid"></div>
      </div>
      <div class="gt-card">
        <div class="chart-head">
          <div class="card-title" style="margin:0;">CGM: last 8 h and the next 2 h</div>
          <div class="stats" id="tirStats"></div>
        </div>
        <label class="toggle"><input type="checkbox" id="showActual" checked onchange="toggleActual(this.checked)"> Show what actually happened next (hidden from the model)</label>
        <div id="cgmChart" class="chart" style="height:320px;"></div>
      </div>
      <div class="gt-card">
        <div class="card-title">Model risk scores up to the selected window (uncalibrated, 0–1)</div>
        <div id="riskChart" class="chart" style="height:210px;"></div>
      </div>
      <div class="gt-card">
        <div class="chart-head">
          <div class="card-title" style="margin:0;">7-day replay overview: tap a point to jump there</div>
          <div class="stats" id="overviewStats"></div>
        </div>
        <div id="overviewChart" class="chart" style="height:200px;"></div>
      </div>
      <div class="btn-row" style="margin-bottom:10px;">
        <button class="btn btn-teal" onclick="showToast('Demo only: no message is sent. A real system would notify the patient app.')">Send nudge to patient (Demo)</button>
        <button class="btn btn-teal" onclick="showToast('Demo only: nothing is written. A real system would log the advisory to the EHR.')">Log advisory to EHR (Demo)</button>
      </div>
    </section>

    <!-- WHAT-IF SANDBOX -->
    <section id="sec1" hidden>
      <div class="section-head">
        <h2>What-If Sandbox <span class="badge-sandbox">Rule-based · illustrative · not the ML model</span></h2>
        <p>Explore how a meal, a walk or (for insulin users) a dose might shift glucose over the next 2 h. The curves come from the simple hand-set rules listed below, not from the trained model, and are not personalised beyond the EHR flags shown. Not for dosing decisions.</p>
      </div>
      <div class="grid-sim">
        <div class="gt-card">
          <div class="ctx" id="simContext"></div>
          <div class="ctrl">
            <label for="simCarbs">Extra carbohydrate now (g)</label>
            <div class="ends"><span>0 g</span><b id="lblCarbs">0 g</b><span>120 g</span></div>
            <input id="simCarbs" type="range" min="0" max="120" step="5" value="0" oninput="runSandbox()">
            <div class="hint">For scale: about 15 g per roti or per 1/3 cup of cooked rice.</div>
          </div>
          <div class="ctrl">
            <label for="simWalk">Brisk walk starting now (min)</label>
            <div class="ends"><span>0 min</span><b id="lblWalk">0 min</b><span>90 min</span></div>
            <input id="simWalk" type="range" min="0" max="90" step="5" value="0" oninput="runSandbox()">
            <div class="hint" id="walkHint"></div>
          </div>
          <div class="ctrl" id="insulinCtrl">
            <label for="simInsulin">Rapid-acting insulin in the scenario (U)</label>
            <div class="ends"><span>0 U</span><b id="lblInsulin">0 U</b><span>20 U</span></div>
            <input id="simInsulin" type="range" min="0" max="20" step="1" value="0" oninput="runSandbox()">
            <div class="hint">Scenario input only. This tool gives no dose advice.</div>
          </div>
          <div class="ctx" id="insulinNote" hidden>This patient's EHR lists no insulin regimen, so insulin is not offered in the sandbox.</div>
          <button class="btn" onclick="resetSandbox()">Reset scenario</button>
          <details class="rules" style="margin-top:12px;">
            <summary>Rules used by the sandbox</summary>
            <ul class="small bullets" id="rulesList"></ul>
          </details>
        </div>
        <div>
          <div class="grid-3" id="simCards"></div>
          <div class="gt-card"><div id="simChart" class="chart" style="height:320px;"></div></div>
          <div class="gt-card" id="simNote"></div>
        </div>
      </div>
    </section>

    <!-- DATA INGEST -->
    <section id="sec2" hidden>
      <div class="section-head">
        <h2>Data Ingest: wearable CSV validation</h2>
        <p>Upload a CSV with a timestamp column and a CGM glucose column (mg/dL); heart_rate, hrv, steps_15m, sleep_stage, meal_carbs_g and insulin_units are optional. The file is read in your browser and never leaves it. Five checks run on the parsed rows. This page validates only: scoring new data needs the Python pipeline (<code>models/train_fusion_model.py</code>).</p>
      </div>
      <div class="grid-2">
        <label class="drop" id="dropZone" for="csvInput">
          <div class="t1">Choose a CSV file (or drop it here)</div>
          <div class="t2">Comma, semicolon or tab separated · up to 5 MB · e.g. columns <code>timestamp,glucose_cgm</code></div>
          <input id="csvInput" type="file" accept=".csv,.txt,text/csv" style="display:none;" onchange="handleFile(this.files[0]); this.value='';">
        </label>
        <div class="gt-card" style="margin-bottom:0;">
          <div class="card-title">No file to hand?</div>
          <div class="btn-row">
            <button class="btn btn-teal" onclick="loadSample()">Validate bundled sample (48 rows of the selected patient)</button>
            <button class="btn" onclick="downloadSample()">Download that sample as CSV</button>
          </div>
          <p class="small muted" style="margin-top:8px;">The sample is the selected patient's first 12 h of synthetic wearable data in the GlycoTwin schema, without labels or model outputs. Edit it and upload it to see the checks react.</p>
        </div>
      </div>
      <div id="ingestResult" style="margin-top:14px;"></div>
    </section>

    <!-- COHORT -->
    <section id="sec3" hidden>
      <div class="section-head">
        <h2>Cohort: synthetic population and 2-hour risk board</h2>
        <p>KPIs are computed in the page from the embedded synthetic EHR of all patients. The risk board scores each patient's latest 15-min window with the shipped model (computed at build time).</p>
      </div>
      <div class="grid-4" id="cohortKpis"></div>
      <div class="grid-3">
        <div class="gt-card"><div class="card-title">Baseline HbA1c (ADA target &lt;7.0%)</div><div id="hba1cChart" class="chart" style="height:250px;"></div></div>
        <div class="gt-card"><div class="card-title">Hypo-risk tier (EHR: sulfonylurea/insulin, age)</div><div id="tierChart" class="chart" style="height:250px;"></div></div>
        <div class="gt-card"><div class="card-title">Primary medication</div><div id="medChart" class="chart" style="height:250px;"></div></div>
      </div>
      <div class="gt-card">
        <div class="chart-head">
          <div class="card-title" style="margin:0;">2-hour risk board: latest window per patient; already out of range first, then highest risk score</div>
          <button class="btn" id="boardToggle" onclick="toggleBoard()">Show all</button>
        </div>
        <p class="small muted" id="boardNote" style="margin-bottom:8px;"></p>
        <div class="table-wrap"><table class="data-table" id="boardTable"></table></div>
      </div>
    </section>

    <!-- MODEL CARD -->
    <section id="sec4" hidden>
      <div class="section-head">
        <h2>Model card</h2>
        <p>What the model is, how it was tested, and where it falls short.</p>
      </div>
      %%MODEL_CARD%%
    </section>

    <footer class="page-foot">
      <p><b>Synthetic data only. Research prototype, not a medical device; not for diagnosis or dosing decisions.</b></p>
      <p>Uses only synthetic data, so no personal health data is processed (designed with DPDP Act 2023 principles in mind). Patient names are generated.</p>
      <p>Glucose bands follow ADA definitions: Level 1 hypoglycaemia &lt;70 mg/dL, Level 2 &lt;54 mg/dL, target range 70–180 mg/dL, marked hyperglycaemia &gt;250 mg/dL.</p>
    </footer>
  </main>
</div>
<div id="toast" class="toast" role="status" aria-live="polite"></div>

<script id="gt-data" type="application/json">%%DATA_JSON%%</script>
<script>
"use strict";
const D = JSON.parse(document.getElementById("gt-data").textContent);
const C = D.clinical, THR = D.thresholds, MAE = D.holdout_mae, H = D.horizon_steps, R = D.sandbox;
const HIST = 32; // 8 h of history on the main chart
let pi = 0, step = 0, section = 0, showActual = true, boardAll = false, overviewBound = false;
const $ = id => document.getElementById(id);
const FONT = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif';

function esc(s) { return String(s).replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c])); }
function fx(x, nd) { return (x === null || x === undefined || !isFinite(x)) ? "–" : Number(x).toFixed(nd === undefined ? 1 : nd); }
function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }
function tsMs(s) {
  const m = /^(\d{4})-(\d{1,2})-(\d{1,2})[ T](\d{1,2}):(\d{2})(?::(\d{2}))?/.exec(String(s).trim());
  if (!m) return NaN;
  return Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +(m[6] || 0));
}
function msTs(ms) {
  const d = new Date(ms), p = n => (n < 10 ? "0" : "") + n;
  return d.getUTCFullYear() + "-" + p(d.getUTCMonth() + 1) + "-" + p(d.getUTCDate()) + " " + p(d.getUTCHours()) + ":" + p(d.getUTCMinutes());
}
function addMin(s, m) { return msTs(tsMs(s) + m * 60000); }
function hhmm(s) { return String(s).slice(11, 16); }
// The EHR's clinical_risk_tier is a hypoglycaemia-risk tier set from drug class and age, not a general risk tier.
const TIER_HELP = "Hypo-risk tier from the EHR: Low without a sulfonylurea or insulin; Moderate with one; High with one and age over 58.";
function tierLabel(t) { return String(t).replace(/ Risk$/, "") + " hypo risk"; }

function gState(g) {
  if (g < C.level2) return {label: "Level 2 hypoglycaemia (<54)", color: "#991B1B", bg: "#FEE2E2"};
  if (g < C.hypo) return {label: "Level 1 hypoglycaemia (<70)", color: "#DC2626", bg: "#FEE2E2"};
  if (g <= C.target_high) return {label: "In target range (70–180)", color: "#047857", bg: "#D1FAE5"};
  if (g <= C.hyper) return {label: "Above target (180–250)", color: "#B45309", bg: "#FEF3C7"};
  return {label: "Marked hyperglycaemia (>250)", color: "#DC2626", bg: "#FEE2E2"};
}
function plotOk(id) {
  if (typeof Plotly === "undefined") { $(id).innerHTML = '<div class="chart-fallback">Chart library (plotly.min.js) did not load.</div>'; return false; }
  return true;
}
function axis(extra) { return Object.assign({showgrid: true, gridcolor: "#F1F5F9", zeroline: false, linecolor: "#CBD5E1", tickfont: {size: 11, color: "#64748B"}, automargin: true}, extra || {}); }
function layout(extra) {
  return Object.assign({paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: {family: FONT, size: 11, color: "#334155"},
    margin: {l: 48, r: 12, t: 10, b: 40}, showlegend: false, hoverlabel: {font: {family: FONT, size: 12}}}, extra || {});
}
const CFG = {responsive: true, displayModeBar: false};
function bands(yMax) {
  const b = (y0, y1, c) => ({type: "rect", xref: "paper", x0: 0, x1: 1, y0: y0, y1: y1, fillcolor: c, line: {width: 0}, layer: "below"});
  return [b(0, C.level2, "rgba(153,27,27,0.14)"), b(C.level2, C.hypo, "rgba(220,38,38,0.10)"), b(C.hypo, C.target_high, "rgba(16,185,129,0.10)"),
          b(C.target_high, C.hyper, "rgba(245,158,11,0.10)"), b(C.hyper, Math.max(yMax, 500), "rgba(220,38,38,0.07)")];
}

/* ---------------- navigation ---------------- */
function toggleMenu(force) {
  const sb = $("sidebar"), open = force === undefined ? !sb.classList.contains("open") : force;
  sb.classList.toggle("open", open);
  $("menuBtn").setAttribute("aria-expanded", open ? "true" : "false");
  $("menuBtn").textContent = open ? "Close" : "Menu";
}
function switchSection(i) {
  section = i;
  for (let k = 0; k < 5; k++) { $("sec" + k).hidden = (k !== i); $("nav" + k).classList.toggle("active", k === i); }
  $("replayBar").hidden = !(i === 0 || i === 1);
  if (window.matchMedia("(max-width: 900px)").matches) toggleMenu(false);
  renderSection();
  window.scrollTo(0, 0);
}
function renderSection() {
  if (section === 0) renderLiveCharts();
  if (section === 1) runSandbox();
  if (section === 3) renderCohortCharts();
  if (section === 4) renderModelCardCharts();
}

/* ---------------- patient ---------------- */
function P() { return D.patients[pi]; }
function switchPatient(i) {
  pi = i;
  $("patientSelect").value = String(i);
  const p = P();
  $("timeSlider").max = p.cgm.length - 1;
  resetSandbox(true);
  renderPatientCard();
  $("startNote").textContent = p.default_reason + " Drag the slider or use the buttons to replay any 15-min window of the 7 days.";
  overviewBound = false;
  setStep(p.default_step);
}
function renderPatientCard() {
  const p = P(), e = p.ehr;
  const tierKey = e.clinical_risk_tier.split(" ")[0];
  const dx = e.past_diagnoses === "None recorded" ? "None recorded" : e.past_diagnoses.split(";").map(s => esc(s.trim())).join(", ");
  const row = (k, v, stack) => `<div class="p360-row${stack ? " stack" : ""}"><span class="p360-k">${k}</span><span class="p360-v">${v}</span></div>`;
  $("patientCard").innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;gap:6px;margin-bottom:6px;">
      <span style="font-weight:700;color:#F8FAFC;font-size:14px;">${esc(e.name)}</span>
      <span class="tier tier-${tierKey}" title="${TIER_HELP}">${esc(tierLabel(e.clinical_risk_tier))}</span></div>
    <div style="margin-bottom:8px;"><span class="chip ${p.tag === "held_out" ? "chip-held" : "chip-train"}">${esc(p.tag_label)}</span></div>
    ${row("Patient ID", esc(e.patient_id))}
    ${row("Age · sex", e.age + " y · " + esc(e.gender))}
    ${row("BMI", fx(e.bmi) + " kg/m²")}
    ${row("Diabetes duration", fx(e.diabetes_duration_years) + " y")}
    ${row("Baseline HbA1c", fx(e.baseline_hba1c, 2) + "%")}
    ${row("Fasting glucose", e.fasting_glucose_mg_dl + " mg/dL")}
    ${row("Blood pressure", e.systolic_bp + "/" + e.diastolic_bp + " mmHg")}
    ${row("eGFR", fx(e.egfr) + " mL/min/1.73m²")}
    ${row("LDL · TG", e.ldl_cholesterol_mg_dl + " · " + e.triglycerides_mg_dl + " mg/dL")}
    ${row("Diet profile", esc(e.diet_profile), true)}
    ${row("Primary medication", esc(e.primary_medication), true)}
    ${row("Insulin regimen", esc(e.insulin_regimen), true)}
    ${row("Past diagnoses", dx, true)}
    ${row("Hypo episodes, past 12 m", e.prior_hypo_episodes_12m)}
    ${row("Family history of T2D", esc(e.family_history_t2d))}
    ${row("TCF7L2 rs7903146 risk alleles", e.tcf7l2_rs7903146_risk_alleles + " of 2")}`;
}

/* ---------------- replay controls ---------------- */
function setStep(s) {
  const p = P();
  step = clamp(isFinite(s) ? s : 0, 0, p.cgm.length - 1);
  $("timeSlider").value = String(step);
  renderLive();
  if (section === 1) runSandbox();
}
function moveStep(d) { setStep(step + d); }
function jumpAlert(dir) {
  const a = P().alert;
  for (let j = step + dir; j >= 0 && j < a.length; j += dir) {
    if (a[j] !== 0 && (j === 0 || a[j - 1] === 0)) { setStep(j); return; }
  }
  showToast(dir > 0 ? "No later alert in this replay." : "No earlier alert in this replay.");
}
function jumpEvent() {
  const g = P().cgm, out = v => v < C.hypo || v > C.hyper;
  for (let j = step + 3; j < g.length; j++) {
    if (out(g[j]) && !out(g[j - 1])) { setStep(j - 2); return; }
  }
  showToast("No later out-of-range reading in this replay.");
}
function toggleActual(v) { showActual = v; renderLive(); }

/* ---------------- live monitor ---------------- */
function phaseOf(p, s) {
  const st = D.sleep_levels[p.sleep[s]];
  if (st !== "Awake") return "Asleep (" + st + " sleep stage)";
  for (let k = 0; k < H && s - k >= 0; k++) {
    if (p.carbs[s - k] > 0) return "Post-meal: " + p.carbs[s - k] + " g carbs logged " + (k * 15) + " min ago";
  }
  if (p.steps[s] >= 500) return "Active: " + p.steps[s] + " steps in the last 15 min";
  return "Awake; no meal logged in the last 2 h";
}
function sumBack(arr, s, n) { let t = 0; for (let k = 0; k < n && s - k >= 0; k++) t += arr[s - k]; return t; }
function mean(a) { return a.reduce((x, y) => x + y, 0) / a.length; }
// Loop-based min/max: spreading a large array into Math.min/max overflows the call stack (~125k values).
function minOf(a) { let m = Infinity; for (let i = 0; i < a.length; i++) if (a[i] < m) m = a[i]; return m; }
function maxOf(a) { let m = -Infinity; for (let i = 0; i < a.length; i++) if (a[i] > m) m = a[i]; return m; }

function renderLive() {
  const p = P(), s = step, g = p.cgm[s], prev = s > 0 ? p.cgm[s - 1] : g, roc = g - prev;
  const st = gState(g), ts = p.ts[s];
  const day = Math.floor((tsMs(ts) - tsMs(p.ts[0].slice(0, 10) + " 00:00")) / 86400000) + 1;
  $("ptName").innerHTML = `${esc(p.name)} <span class="muted" style="font-size:12px;font-weight:500;">${esc(p.id)}</span> <span class="chip ${p.tag === "held_out" ? "chip-held" : "chip-train"}">${esc(p.tag_label)}</span>`;
  $("tsNow").textContent = ts;
  $("dayNow").textContent = "· day " + day + " of 7 · window " + (s + 1) + " of " + p.cgm.length;

  const pos = v => clamp((v - 40) / 360 * 100, 0, 100);
  $("glucoseHero").innerHTML = `
    <div class="hero-top"><span class="hero-label">Current CGM reading</span>
      <span class="chip" style="background:${st.bg};color:${st.color};">${st.label}</span></div>
    <div class="hero-val"><span class="hero-num tabular-nums" style="color:${st.color};">${fx(g)}</span><span class="muted">mg/dL</span>
      <span class="hero-roc tabular-nums">${roc > 2 ? "rising" : (roc < -2 ? "falling" : "steady")} ${roc >= 0 ? "+" : ""}${fx(roc)} mg/dL per 15 min</span></div>
    <div class="range-track" aria-hidden="true"><div class="range-needle" style="left:${fx(pos(g), 2)}%;"></div></div>
    <div class="range-labels" aria-hidden="true">
      <span style="left:${fx(pos(C.hypo), 2)}%;">70</span><span style="left:${fx(pos(C.target_high), 2)}%;">180</span>
      <span style="left:${fx(pos(C.hyper), 2)}%;">250</span><span style="left:${fx((pos(C.target_high) + pos(C.hypo)) / 2, 2)}%;color:#047857;font-weight:700;">target</span>
      <span class="r2" style="left:${fx(pos(C.level2), 2)}%;">54</span><span class="r2" style="left:${fx((pos(C.hyper) + 100) / 2, 2)}%;">marked high</span></div>`;

  const ins2h = sumBack(p.ins, s, H), carbs2h = sumBack(p.carbs, s, H);
  const hr1 = mean(p.hr.slice(Math.max(0, s - 3), s + 1)), hrv1 = mean(p.hrv.slice(Math.max(0, s - 3), s + 1));
  $("kpiGrid").innerHTML = `
    <div class="gt-card"><div class="kpi-label">Heart rate · HRV</div>
      <div class="kpi-val tabular-nums">${p.hr[s]} <span class="unit">bpm</span> · ${p.hrv[s]} <span class="unit">ms</span></div>
      <div class="kpi-sub tabular-nums">1 h mean: ${fx(hr1, 0)} bpm · ${fx(hrv1, 0)} ms</div></div>
    <div class="gt-card"><div class="kpi-label">Activity (steps)</div>
      <div class="kpi-val tabular-nums">${p.steps[s]} <span class="unit">in 15 min</span></div>
      <div class="kpi-sub tabular-nums">Last 1 h: ${sumBack(p.steps, s, 4)} · last 2 h: ${sumBack(p.steps, s, 8)}</div></div>
    <div class="gt-card"><div class="kpi-label">Sleep · phase</div>
      <div class="kpi-val">${esc(D.sleep_levels[p.sleep[s]])}</div>
      <div class="kpi-sub">${esc(phaseOf(p, s))}</div></div>
    <div class="gt-card"><div class="kpi-label">Logged in the last 2 h</div>
      <div class="kpi-val tabular-nums">${carbs2h} <span class="unit">g carbs</span></div>
      <div class="kpi-sub">${p.on_insulin ? "Insulin: " + ins2h + " U logged (" + esc(p.ehr.insulin_regimen) + ")" : "Not on insulin (EHR) · " + esc(p.ehr.primary_medication)}</div></div>`;

  renderAlertCard();

  // 8 h statistics
  const w = p.cgm.slice(Math.max(0, s - HIST), s + 1), m = mean(w);
  const sd = Math.sqrt(w.reduce((a, v) => a + (v - m) * (v - m), 0) / w.length);
  const tir = 100 * w.filter(v => v >= C.hypo && v <= C.target_high).length / w.length;
  const tbr = 100 * w.filter(v => v < C.hypo).length / w.length;
  $("tirStats").innerHTML = `<span>Last ${fx((w.length - 1) / 4, 2).replace(/\.?0+$/, "")} h (${w.length} readings):</span>
    <span>time in range 70–180 <b class="tabular-nums">${fx(tir, 0)}%</b></span><span>below 70 <b class="tabular-nums">${fx(tbr, 0)}%</b></span>
    <span>CV <b class="tabular-nums">${fx(m > 0 ? 100 * sd / m : 0, 0)}%</b> (target ≤36%)</span>`;
  renderLiveCharts();
}

function renderAlertCard() {
  const p = P(), s = step, g = p.cgm[s], a = p.alert[s], ph = p.p_hypo[s], pr = p.p_hyper[s], f = p.f2h[s];
  const tF = hhmm(addMin(p.ts[s], 120));
  let cls, kicker, title, body;
  const medNote = [p.on_su ? "sulfonylurea" : "", p.on_insulin ? "insulin" : ""].filter(Boolean).join(" and ");
  const fcTxt = "forecast " + fx(f) + " mg/dL at " + tF;
  if (g < C.hypo) {
    // Current low: rescue text applies (box below). No "predicted low" snack advice here.
    cls = "st-hypo";
    kicker = "CGM below 70 now · " + (a === 1 ? "model hypo alert, next 2 h" : (a === 2 ? "model hyper alert, next 2 h" : "no model alert"));
    title = (g < C.level2 ? "Level 2 hypoglycaemia now: CGM " + fx(g) + " mg/dL (below 54)" : "Level 1 hypoglycaemia now: CGM " + fx(g) + " mg/dL (below 70)");
    if (a === 1) body = "This is a current low, not only a predicted one. The model also expects readings below 70 mg/dL to continue within the next 2 h (" + fcTxt + ").";
    else if (a === 2) body = "The model's hyper risk score is above its threshold for the next 2 h (" + fcTxt + "), but the current reading is low: treat the current low first.";
    else body = "This is a current low. The model's hypo risk score is below its alert threshold, so it does not flag further lows in the next 2 h (" + fcTxt + "). Treat the current low and keep watching the CGM.";
  } else if (g > C.hyper) {
    cls = "st-hyper";
    kicker = "CGM above 250 now · " + (a === 2 ? "model hyper alert, next 2 h" : (a === 1 ? "model hypo alert, next 2 h" : "no model alert"));
    title = "Marked hyperglycaemia now: CGM " + fx(g) + " mg/dL (above 250)";
    const plan = " Review the recent meal, missed doses and hydration, and follow the care team's plan for high readings." +
                 (p.on_insulin ? " Any insulin correction follows the plan agreed with the care team; this tool gives no dose advice." : "");
    if (a === 2) body = "This is a current high reading, not only a predicted one. The model also expects readings above 250 mg/dL to continue within the next 2 h (" + fcTxt + ")." + plan;
    else if (a === 1) body = "The reading is high now, but the model flags a possible reading below 70 mg/dL within 2 h (" + fcTxt + "): review recent glucose-lowering medication" + (medNote ? " (" + medNote + ")" : "") + " with the care team and keep watching the CGM.";
    else body = "This is a current high reading. The model's hyper risk score is below its alert threshold for the next 2 h (" + fcTxt + ")." + plan;
  } else if (a === 1) {
    cls = "st-hypo"; kicker = "Model alert · next 2 h";
    title = "Hypoglycaemia risk alert: the model flags a possible reading below 70 mg/dL within 2 h";
    body = "Predicted low, not a current low: consider a snack and review medication timing with the care team" +
           (medNote ? " (this patient takes " + medNote + ")" : "") + ". Keep watching the CGM.";
  } else if (a === 2) {
    cls = "st-hyper"; kicker = "Model alert · next 2 h";
    title = "Marked hyperglycaemia risk alert: the model flags a possible reading above 250 mg/dL within 2 h";
    body = "Review the recent meal and activity, drink water, and follow the care team's plan for high readings." +
           (p.on_insulin ? " Any insulin correction follows the plan agreed with the care team; this tool gives no dose advice." : "");
  } else if (f < C.hypo) {
    cls = "st-watch"; kicker = "No alert · forecast below range";
    title = "Point forecast " + fx(f) + " mg/dL at " + tF + ", but the hypo risk score is below the alert threshold";
    body = "The two model outputs disagree; keep watching the CGM.";
  } else if (f > C.hyper) {
    cls = "st-watch"; kicker = "No alert · forecast above 250";
    title = "Point forecast " + fx(f) + " mg/dL at " + tF + ", but the hyper risk score is below the alert threshold";
    body = "The two model outputs disagree; keep watching the CGM.";
  } else if (f > C.target_high) {
    cls = "st-watch"; kicker = "No alert · above target";
    title = "Forecast above target: " + fx(f) + " mg/dL at " + tF + " (180–250 mg/dL)";
    body = "No model alert. Above the 70–180 target range but below the 250 mg/dL alert level.";
  } else {
    cls = "st-ok"; kicker = "No alert · in target range";
    title = "Forecast in target range: " + fx(f) + " mg/dL at " + tF;
    body = "No model alert for the next 2 h.";
  }
  let rescue = "";
  if (g < C.hypo) {
    rescue = `<div class="rescue"><b>CGM is below 70 mg/dL now${g < C.level2 ? " (Level 2, below 54)" : ""}.</b> Take 15 g of fast-acting carbohydrate (for example glucose tablets or juice) and recheck in 15 minutes; repeat if still below 70.${g < C.level2 ? " Level 2 readings need prompt treatment and should be reported to the care team." : ""}</div>`;
  }
  const bar = (v, t, col) => `<div class="scorebar"><i style="width:${fx(v * 100, 1)}%;background:${col};"></i><b style="left:${fx(t * 100, 1)}%;"></b></div>`;
  const act = p.actual2h[s], ev = p.actual_event[s];
  const fut = p.cgm.slice(s + 1, s + 1 + H);
  const evTxt = ev === 1 ? "yes: a reading below 70 mg/dL" : (ev === 2 ? "yes: a reading above 250 mg/dL" : "no out-of-range reading");
  const futTxt = fut.length ? `Range of the next ${fut.length} readings: <b class="tabular-nums">${fx(Math.min(...fut))}–${fx(Math.max(...fut))}</b> mg/dL.<br>` : "";
  const actualHtml = showActual ? `<div class="actualbox"><b>What actually happened (hidden from the model)</b><br>
      Reading at +2 h: <b class="tabular-nums">${fx(act)}</b> mg/dL.<br>${futTxt}Event in the next 2 h: <b>${evTxt}</b>.</div>`
    : `<div class="actualbox">Actual outcome hidden. Tick "Show what actually happened" to compare.</div>`;
  $("alertCard").innerHTML = `<div class="alert-card ${cls}">
      <div class="a-kicker">${kicker}</div><div class="a-title">${esc(title)}</div>
      <div class="a-body">${esc(body)}</div>${rescue}
      <div class="alert-grid">
        <div class="scorebox">
          <div class="row"><span>Hypo risk score (uncalibrated)</span><b class="tabular-nums">${fx(ph, 3)}</b></div>${bar(ph, THR.hypo, "#DC2626")}
          <div class="row"><span>Hyper risk score (uncalibrated)</span><b class="tabular-nums">${fx(pr, 3)}</b></div>${bar(pr, THR.hyper, "#C2410C")}
          <div class="row muted"><span>Alert thresholds (black ticks): hypo ≥ ${THR.hypo}, hyper ≥ ${THR.hyper}</span></div>
          <div class="row"><span>Regression forecast at +2 h (${tF})</span><b class="tabular-nums">${fx(f)} mg/dL ± ${fx(MAE)}</b></div>
          <div class="muted" style="margin-top:4px;">± is the held-out mean absolute error. The alert covers any reading in the next 2 h; the forecast is the single value exactly 2 h ahead, so they can differ. Scores are model outputs, not probabilities.</div>
        </div>
        ${actualHtml}
      </div></div>`;
  const pill = $("alertPill");
  const now = g < C.hypo ? "CGM below 70 now · " : (g > C.hyper ? "CGM above 250 now · " : "");
  if (a === 1) { pill.style.background = "#FEE2E2"; pill.style.color = "#991B1B"; pill.textContent = now + "Hypo alert (next 2 h)"; }
  else if (a === 2) { pill.style.background = "#FFEDD5"; pill.style.color = "#9A3412"; pill.textContent = now + "Hyper alert (next 2 h)"; }
  else if (now) { pill.style.background = g < C.hypo ? "#FEE2E2" : "#FFEDD5"; pill.style.color = g < C.hypo ? "#991B1B" : "#9A3412"; pill.textContent = now + "no model alert"; }
  else { pill.style.background = "#DCFCE7"; pill.style.color = "#166534"; pill.textContent = "No model alert"; }
}

function renderLiveCharts() {
  if (section !== 0) return;
  const p = P(), s = step, a0 = Math.max(0, s - HIST);
  const xs = p.ts.slice(a0, s + 1), ys = p.cgm.slice(a0, s + 1);
  const xEnd = addMin(p.ts[s], 130), xStart = addMin(p.ts[s], -HIST * 15 - 10), tF = addMin(p.ts[s], 120);
  const futX = [p.ts[s]].concat(p.ts.slice(s + 1, s + 1 + H)), futY = [p.cgm[s]].concat(p.cgm.slice(s + 1, s + 1 + H));
  const f = p.f2h[s], fc = gState(f).color;
  const vals = ys.concat(showActual ? futY : []).concat([f + MAE, showActual ? p.actual2h[s] : 0]);
  const yMax = Math.max(200, Math.ceil((Math.max(...vals) + 25) / 50) * 50);
  if (plotOk("cgmChart")) {
    const traces = [
      {x: xs, y: ys, mode: "lines+markers", name: "CGM (seen by the model)", line: {color: "#0D9488", width: 2}, marker: {size: 6, color: "#0D9488"},
       hovertemplate: "%{x}<br>CGM %{y:.1f} mg/dL<extra></extra>"},
      {x: [tF], y: [f], mode: "markers", name: "Model forecast at +2 h (± held-out MAE)", marker: {size: 13, symbol: "diamond", color: fc, line: {color: "#fff", width: 2}},
       error_y: {type: "constant", value: MAE, visible: true, color: fc, thickness: 2, width: 6}, hovertemplate: "Forecast for %{x}<br>%{y:.1f} mg/dL<extra></extra>"}];
    if (showActual) {
      traces.push({x: futX, y: futY, mode: "lines+markers", name: "Actual next 2 h (hidden from the model)", line: {color: "#64748B", width: 2, dash: "dot"},
        marker: {size: 5, color: "#64748B"}, hovertemplate: "%{x}<br>Actual %{y:.1f} mg/dL<extra></extra>"});
      traces.push({x: [tF], y: [p.actual2h[s]], mode: "markers", name: "Actual reading at +2 h", marker: {size: 11, symbol: "circle-open", color: "#334155", line: {width: 2}},
        hovertemplate: "Actual at %{x}<br>%{y:.1f} mg/dL<extra></extra>"});
    }
    const shapes = bands(yMax).concat([
      {type: "rect", x0: p.ts[s], x1: xEnd, yref: "paper", y0: 0, y1: 1, fillcolor: "rgba(148,163,184,0.10)", line: {width: 0}, layer: "below"},
      {type: "line", x0: p.ts[s], x1: p.ts[s], yref: "paper", y0: 0, y1: 1, line: {color: "#334155", width: 1.5, dash: "dot"}}]);
    Plotly.react("cgmChart", traces, layout({
      showlegend: true, legend: {orientation: "h", x: 0, y: -0.18, font: {size: 11}},
      margin: {l: 48, r: 12, t: 22, b: 40}, shapes: shapes,
      annotations: [{x: p.ts[s], y: 1, yref: "paper", text: "selected window ", showarrow: false, xanchor: "right", yanchor: "bottom", font: {size: 10, color: "#334155"}},
                    {x: p.ts[s], y: 1, yref: "paper", text: " next 2 h", showarrow: false, xanchor: "left", yanchor: "bottom", font: {size: 10, color: "#64748B"}}],
      xaxis: axis({type: "date", range: [xStart, xEnd], tickformat: "%H:%M\n%b %d"}),
      yaxis: axis({range: [40, yMax], title: {text: "mg/dL", font: {size: 11}}})}), CFG);
  }
  if (plotOk("riskChart")) {
    const rx = p.ts.slice(a0, s + 1);
    Plotly.react("riskChart", [
      {x: rx, y: p.p_hypo.slice(a0, s + 1), mode: "lines+markers", name: "Hypo risk score", line: {color: "#DC2626", width: 2}, marker: {size: 5},
       hovertemplate: "%{x}<br>Hypo score %{y:.3f}<extra></extra>"},
      {x: rx, y: p.p_hyper.slice(a0, s + 1), mode: "lines+markers", name: "Hyper risk score", line: {color: "#C2410C", width: 2, dash: "dash"}, marker: {size: 5, symbol: "square"},
       hovertemplate: "%{x}<br>Hyper score %{y:.3f}<extra></extra>"}],
      layout({showlegend: true, legend: {orientation: "h", x: 0, y: -0.32, font: {size: 11}}, margin: {l: 48, r: 12, t: 8, b: 40},
        shapes: [{type: "line", xref: "paper", x0: 0, x1: 1, y0: THR.hypo, y1: THR.hypo, line: {color: "#991B1B", width: 1, dash: "dot"}},
                 {type: "line", x0: p.ts[s], x1: p.ts[s], yref: "paper", y0: 0, y1: 1, line: {color: "#334155", width: 1.5, dash: "dot"}}],
        annotations: [{xref: "paper", x: 0, y: Math.max(THR.hypo, THR.hyper), text: "alert thresholds ≈ " + THR.hypo + " / " + THR.hyper, showarrow: false, xanchor: "left", yanchor: "bottom", font: {size: 10, color: "#991B1B"}}],
        xaxis: axis({type: "date", range: [xStart, xEnd], tickformat: "%H:%M\n%b %d"}), yaxis: axis({range: [0, 1.02], tickformat: ".1f"})}), CFG);
  }
  renderOverview();
}

function renderOverview() {
  const p = P(), s = step;
  const idx = p.cgm.map((_, i) => i);
  const hy = idx.filter(i => p.alert[i] === 1), hr = idx.filter(i => p.alert[i] === 2);
  const lo = p.cgm.filter(v => v < C.hypo).length, hi = p.cgm.filter(v => v > C.hyper).length;
  $("overviewStats").innerHTML = `<span>CGM readings &lt;70: <b>${lo}</b> · &gt;250: <b>${hi}</b></span><span>Alert windows: hypo <b>${hy.length}</b> · hyper <b>${hr.length}</b></span>`;
  if (!plotOk("overviewChart")) return;
  const yMax = Math.max(200, Math.ceil((Math.max(...p.cgm) + 25) / 50) * 50);
  Plotly.react("overviewChart", [
    {x: p.ts, y: p.cgm, customdata: idx, mode: "lines", name: "CGM", line: {color: "#0D9488", width: 1.5}, hovertemplate: "%{x}<br>CGM %{y:.1f} mg/dL<extra></extra>"},
    {x: hy.map(i => p.ts[i]), y: hy.map(i => p.cgm[i]), customdata: hy, mode: "markers", name: "Hypo alert window", marker: {size: 8, color: "#DC2626", symbol: "triangle-down"},
     hovertemplate: "%{x}<br>Hypo alert<extra></extra>"},
    {x: hr.map(i => p.ts[i]), y: hr.map(i => p.cgm[i]), customdata: hr, mode: "markers", name: "Hyper alert window", marker: {size: 8, color: "#C2410C", symbol: "triangle-up"},
     hovertemplate: "%{x}<br>Hyper alert<extra></extra>"}],
    layout({showlegend: true, legend: {orientation: "h", x: 0, y: -0.35, font: {size: 11}}, margin: {l: 48, r: 12, t: 8, b: 40},
      shapes: bands(yMax).concat([{type: "line", x0: p.ts[s], x1: p.ts[s], yref: "paper", y0: 0, y1: 1, line: {color: "#0F172A", width: 2}}]),
      xaxis: axis({type: "date", tickformat: "%b %d"}), yaxis: axis({range: [40, yMax]})}), CFG);
  if (!overviewBound) {
    const el = $("overviewChart");
    if (el.on) { el.removeAllListeners && el.removeAllListeners("plotly_click"); el.on("plotly_click", ev => { const pt = ev.points && ev.points[0]; if (pt && pt.customdata !== undefined) setStep(pt.customdata); }); overviewBound = true; }
  }
}

/* ---------------- what-if sandbox (rule-based) ---------------- */
function resetSandbox(silent) {
  $("simCarbs").value = 0; $("simWalk").value = 0; $("simInsulin").value = 0;
  const p = P();
  $("insulinCtrl").hidden = !p.on_insulin;
  $("insulinNote").hidden = p.on_insulin;
  if (!silent) runSandbox();
}
function carbShape(m) { const k = m / R.carb_peak_min; return k * Math.exp(1 - k); }
function insShape(m) { const k = m / R.insulin_time_constant_min; return 1 - (1 + k) * Math.exp(-k); }
function runSandbox() {
  if (section !== 1) return;
  const p = P(), s = step;
  const carbs = parseInt($("simCarbs").value, 10), walk = parseInt($("simWalk").value, 10);
  const ins = p.on_insulin ? parseInt($("simInsulin").value, 10) : 0;
  $("lblCarbs").textContent = carbs + " g"; $("lblWalk").textContent = walk + " min"; $("lblInsulin").textContent = ins + " U";
  const blocks = Math.ceil(walk / 15);
  $("walkHint").textContent = walk > 0 ? "Counts in " + blocks + " of the 15-min windows (a " + walk + "-min walk lowers glucose for every minute walked)." : "No walk in the scenario.";
  const exMult = (p.on_su || p.on_insulin) ? R.walk_multiplier_su_or_insulin : 1;
  const g0 = p.cgm[s], f = p.f2h[s];
  const t = [], base = [], sc = [];
  for (let k = 0; k <= H; k++) {
    const m = k * 15; t.push(m);
    const b = g0 + (f - g0) * m / 120;
    base.push(b);
    const walked = Math.min(m, walk);  // minutes walked by time m (so a 10-min walk counts)
    const v = b + carbs * R.carb_rise_mg_dl_per_g * carbShape(m) - walked * R.walk_drop_mg_dl_per_min * exMult - ins * R.insulin_drop_mg_dl_per_unit * insShape(m);
    sc.push(clamp(v, R.floor_mg_dl, R.ceiling_mg_dl));
  }
  const nadir = Math.min(...sc), peak = Math.max(...sc), inR = sc.filter(v => v >= C.hypo && v <= C.target_high).length, low = sc.filter(v => v < C.hypo).length;
  const changed = carbs > 0 || walk > 0 || ins > 0;
  const a = p.alert[s];
  $("simContext").innerHTML = `<b>Selected window:</b> ${esc(p.ts[s])} · CGM ${fx(g0)} mg/dL<br>
    <b>ML model at this window:</b> forecast ${fx(f)} mg/dL at +2 h · hypo score ${fx(p.p_hypo[s], 3)} · hyper score ${fx(p.p_hyper[s], 3)} · ${a === 1 ? "hypo alert" : (a === 2 ? "hyper alert" : "no alert")}<br>
    <span class="muted">The baseline below is a straight line from the current reading to that forecast; the sandbox does not re-run the model.</span>`;
  const card = (label, val, sub, col) => `<div class="gt-card stat"><div class="stat-label">${label}</div><div class="stat-value tabular-nums" style="color:${col};">${val}</div><div class="stat-sub">${sub}</div></div>`;
  $("simCards").innerHTML =
    card("Scenario lowest", fx(nadir, 0) + ' <span class="unit">mg/dL</span>', "Baseline lowest " + fx(Math.min(...base), 0) + " mg/dL", gState(nadir).color) +
    card("Scenario highest", fx(peak, 0) + ' <span class="unit">mg/dL</span>', "Baseline highest " + fx(Math.max(...base), 0) + " mg/dL", gState(peak).color) +
    card("Points in 70–180", inR + ' <span class="unit">of ' + sc.length + "</span>", low > 0 ? low + " point(s) below 70 mg/dL" : "None below 70 mg/dL", inR === sc.length ? "#047857" : "#B45309");
  if (plotOk("simChart")) {
    const yMax = Math.max(300, Math.ceil((Math.max(...sc, ...base) + 20) / 50) * 50);
    Plotly.react("simChart", [
      {x: t, y: base, mode: "lines", name: "Baseline: line to the ML forecast (not a trajectory)", line: {color: "#64748B", width: 2, dash: "dash"}, hovertemplate: "+%{x} min<br>%{y:.0f} mg/dL<extra></extra>"},
      {x: t, y: sc, mode: "lines+markers", name: "Rule-based scenario (illustrative)", line: {color: "#0D9488", width: 2.5}, marker: {size: 7}, hovertemplate: "+%{x} min<br>%{y:.0f} mg/dL<extra></extra>"}],
      layout({showlegend: true, legend: {orientation: "h", x: 0, y: -0.22, font: {size: 11}}, margin: {l: 48, r: 12, t: 10, b: 40}, shapes: bands(yMax),
        xaxis: axis({title: {text: "minutes after the selected window", font: {size: 11}}, dtick: 15, range: [-3, 123]}), yaxis: axis({range: [40, yMax], title: {text: "mg/dL", font: {size: 11}}})}), CFG);
  }
  let note;
  if (!changed) note = "No change entered yet: the scenario equals the baseline. Move a slider to explore.";
  else if (nadir < C.hypo) note = "The scenario dips below 70 mg/dL (lowest " + fx(nadir, 0) + " mg/dL). Review this kind of scenario with the care team before trying it.";
  else if (peak > C.hyper && carbs > 0) note = "With " + carbs + " g extra carbohydrate the scenario peaks at " + fx(peak, 0) + " mg/dL, above 250 mg/dL.";
  else if (peak > C.hyper) note = "The scenario stays above 250 mg/dL at its highest (" + fx(peak, 0) + " mg/dL).";
  else note = "The scenario stays between " + fx(nadir, 0) + " and " + fx(peak, 0) + " mg/dL over the next 2 h.";
  $("simNote").innerHTML = `<div class="card-title">Sandbox note</div><div class="small">${esc(note)}</div>
    <div class="small muted" style="margin-top:6px;">Illustrative arithmetic only. It is not the trained model, not personalised beyond the medication flags in the EHR, and not for dosing decisions.</div>`;
}

/* ---------------- data ingest ---------------- */
const ALIASES = {
  timestamp: ["timestamp", "time", "datetime", "date_time", "time_stamp", "device_timestamp", "date"],
  glucose: ["glucose_cgm", "glucose", "cgm", "glucose_mg_dl", "sgv", "glucose_value", "glucose_value_mg_dl", "historic_glucose_mg_dl", "bg"],
  heart_rate: ["heart_rate", "hr", "heart_rate_bpm", "bpm"],
  hrv: ["hrv", "hrv_ms", "rmssd"],
  steps_15m: ["steps_15m", "steps"],
  sleep_stage: ["sleep_stage", "sleep"],
  meal_carbs_g: ["meal_carbs_g", "carbs", "carbs_g", "carbohydrates_g"],
  insulin_units: ["insulin_units", "insulin", "insulin_u"],
};
const BOUNDS = {glucose: [40, 400, "mg/dL"], heart_rate: [30, 220, "bpm"], hrv: [5, 300, "ms"], steps_15m: [0, 3000, "steps"], meal_carbs_g: [0, 300, "g"], insulin_units: [0, 50, "U"]};
const MAX_RATE = 4; // mg/dL per minute between consecutive readings
function normName(s) { return String(s).toLowerCase().trim().replace(/\(.*?\)/g, m => "_" + m.slice(1, -1)).replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, ""); }
function parseDelimited(text) {
  text = text.replace(/^﻿/, "");
  const first = (text.split(/\r?\n/).find(l => l.trim().length) || "");
  const cnt = d => first.split(d).length - 1;
  const delim = [",", ";", "\t"].reduce((b, d) => (cnt(d) > cnt(b) ? d : b), ",");
  const rows = []; let row = [], field = "", q = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (q) { if (ch === '"') { if (text[i + 1] === '"') { field += '"'; i++; } else q = false; } else field += ch; }
    else if (ch === '"') q = true;
    else if (ch === delim) { row.push(field); field = ""; }
    else if (ch === "\n" || ch === "\r") { if (ch === "\r" && text[i + 1] === "\n") i++; row.push(field); rows.push(row); row = []; field = ""; }
    else field += ch;
  }
  if (field.length || row.length) { row.push(field); rows.push(row); }
  return {rows: rows.filter(r => r.some(c => c.trim() !== "")), delim: delim};
}
function numOrNaN(s) { const t = String(s === undefined ? "" : s).trim(); return /^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/.test(t) ? parseFloat(t) : NaN; }
function parseTsFlexible(s) {
  const t = String(s === undefined ? "" : s).trim();
  let ms = tsMs(t);
  if (isFinite(ms)) return ms;
  const m = /^(\d{1,2})[-\/.](\d{1,2})[-\/.](\d{4})[ T](\d{1,2}):(\d{2})(?::(\d{2}))?/.exec(t); // DD-MM-YYYY HH:MM
  if (m) return Date.UTC(+m[3], +m[2] - 1, +m[1], +m[4], +m[5], +(m[6] || 0));
  return NaN;
}
function badge(v) { return v === "PASS" ? '<span class="b-pass">PASS</span>' : (v === "WARN" ? '<span class="b-warn">WARN</span>' : '<span class="b-fail">FAIL</span>'); }
function handleFile(file) {
  if (!file) return;
  if (file.size > 5 * 1024 * 1024) { renderIngestError(file.name, "File is larger than 5 MB."); return; }
  const rd = new FileReader();
  rd.onload = () => ingestText(String(rd.result), file.name);
  rd.onerror = () => renderIngestError(file.name, "The browser could not read this file.");
  rd.readAsText(file);
}
function renderIngestError(name, msg) {
  $("ingestResult").innerHTML = `<div class="verdict" style="background:#FEE2E2;color:#991B1B;">${esc(name)}: ${esc(msg)}</div>`;
}
function sampleCsv() {
  const p = P(), n = Math.min(48, p.cgm.length);
  const lines = ["patient_id,timestamp,glucose_cgm,heart_rate,hrv,steps_15m,sleep_stage,meal_carbs_g,insulin_units"];
  for (let i = 0; i < n; i++) lines.push([p.id, p.ts[i] + ":00", fx(p.cgm[i]), p.hr[i], p.hrv[i], p.steps[i], D.sleep_levels[p.sleep[i]], p.carbs[i], p.ins[i]].join(","));
  return lines.join("\n") + "\n";
}
function loadSample() { ingestText(sampleCsv(), "bundled sample (" + P().id + ", first 48 windows)"); }
function downloadSample() {
  const blob = new Blob([sampleCsv()], {type: "text/csv"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "glycotwin_sample_" + P().id + ".csv";
  document.body.appendChild(a); a.click(); setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 500);
}
function ingestText(text, name) {
  const parsed = parseDelimited(text), rows = parsed.rows;
  const checks = [];
  // Step 1: schema
  let hIdx = -1, map = {};
  for (let r = 0; r < Math.min(10, rows.length); r++) {
    const names = rows[r].map(normName), m = {};
    for (const key in ALIASES) { const j = names.findIndex(n => ALIASES[key].includes(n)); if (j >= 0) m[key] = j; }
    if (m.timestamp !== undefined && m.glucose !== undefined) { hIdx = r; map = m; break; }
  }
  if (hIdx < 0) {
    checks.push({n: "1. Schema", v: "FAIL", d: "No header row with both a timestamp column and a glucose column was found in the first 10 lines. Expected names such as timestamp and glucose_cgm."});
    ["2. Continuity", "3. Physiological bounds", "4. Rate of change", "5. Missingness"].forEach(n => checks.push({n: n, v: "FAIL", d: "Not run: the schema check failed."}));
    renderIngest(name, parsed, checks, null);
    return;
  }
  const header = rows[hIdx], data = rows.slice(hIdx + 1);
  const ragged = data.filter(r => r.length !== header.length).length;
  const optional = Object.keys(map).filter(k => k !== "timestamp" && k !== "glucose").map(k => header[map[k]]);
  const recs = data.map((r, i) => ({i: i, tsRaw: r[map.timestamp], ms: parseTsFlexible(r[map.timestamp]), g: numOrNaN(r[map.glucose]), r: r}));
  const gVals = recs.map(x => x.g).filter(v => isFinite(v));
  const medG = gVals.length ? gVals.slice().sort((a, b) => a - b)[Math.floor(gVals.length / 2)] : NaN;
  let v1 = data.length === 0 ? "FAIL" : "PASS", d1 = `Timestamp column "${esc(header[map.timestamp])}", glucose column "${esc(header[map.glucose])}"` +
    (optional.length ? "; optional columns: " + optional.map(esc).join(", ") : "; no optional columns") + ". " + data.length + " data rows.";
  if (ragged > 0) { v1 = v1 === "FAIL" ? v1 : "WARN"; d1 += " " + ragged + " row(s) have a different number of fields from the header."; }
  if (isFinite(medG) && medG < 30) { v1 = v1 === "FAIL" ? v1 : "WARN"; d1 += " Median glucose " + fx(medG) + " looks like mmol/L; this page expects mg/dL."; }
  if (data.length === 0) d1 += " The file has a header but no data rows.";
  checks.push({n: "1. Schema", v: v1, d: d1});
  // Step 2: continuity
  const bad = recs.filter(x => !isFinite(x.ms)).length, ok = recs.filter(x => isFinite(x.ms));
  let unsorted = 0; for (let k = 1; k < ok.length; k++) if (ok[k].ms < ok[k - 1].ms) unsorted++;
  const sorted = ok.slice().sort((a, b) => a.ms - b.ms || a.i - b.i);
  let dups = 0, gaps = 0; const iv = [];
  for (let k = 1; k < sorted.length; k++) { const dm = (sorted[k].ms - sorted[k - 1].ms) / 60000; if (dm === 0) dups++; else { iv.push(dm); if (dm > 30) gaps++; } }
  const medIv = iv.length ? iv.slice().sort((a, b) => a - b)[Math.floor(iv.length / 2)] : NaN;
  let v2 = "PASS";
  if (ok.length < 2 || bad > 0.2 * recs.length) v2 = "FAIL"; else if (bad || unsorted || dups || gaps) v2 = "WARN";
  checks.push({n: "2. Continuity", v: v2, d: `${ok.length} of ${recs.length} timestamps parsed${bad ? " (" + bad + " unreadable)" : ""}; median interval ${fx(medIv, 0)} min; ${gaps} gap(s) over 30 min; ${dups} duplicate timestamp(s); ${unsorted ? unsorted + " out-of-order row(s) (sorted before the next checks)" : "rows in time order"}.`});
  // Step 3: bounds
  let oob = 0; const oobParts = [];
  for (const key in BOUNDS) {
    if (map[key] === undefined) continue;
    const [lo, hi, u] = BOUNDS[key];
    const c = recs.filter(x => { const v = key === "glucose" ? x.g : numOrNaN(x.r[map[key]]); return isFinite(v) && (v < lo || v > hi); }).length;
    if (c) oobParts.push(c + " " + (key === "glucose" ? "glucose" : esc(header[map[key]])) + " value(s) outside " + lo + "–" + hi + " " + u);
    oob += c;
  }
  const v3 = oob === 0 ? "PASS" : (oob <= 0.05 * Math.max(1, recs.length) ? "WARN" : "FAIL");
  checks.push({n: "3. Physiological bounds", v: v3, d: oob ? oobParts.join("; ") + "." : "All numeric values within bounds (glucose 40–400 mg/dL, HR 30–220 bpm, HRV 5–300 ms, steps 0–3000 per row, carbs 0–300 g, insulin 0–50 U)."});
  // Step 4: rate of change
  const gs = sorted.filter(x => isFinite(x.g));
  let pairs = 0, fast = 0, maxRate = 0;
  for (let k = 1; k < gs.length; k++) {
    const dm = (gs[k].ms - gs[k - 1].ms) / 60000;
    if (dm <= 0 || dm > 30) continue;
    pairs++; const rate = Math.abs(gs[k].g - gs[k - 1].g) / dm; maxRate = Math.max(maxRate, rate); if (rate > MAX_RATE) fast++;
  }
  const v4 = pairs === 0 ? "WARN" : (fast === 0 ? "PASS" : (fast <= 0.05 * pairs ? "WARN" : "FAIL"));
  checks.push({n: "4. Rate of change", v: v4, d: pairs ? `${fast} of ${pairs} consecutive-reading pairs change faster than ${MAX_RATE} mg/dL per minute (likely sensor artefact); fastest ${fx(maxRate, 2)} mg/dL/min.` : "No consecutive readings 30 min apart or less to compare."});
  // Step 5: missingness
  const missG = recs.filter(x => !isFinite(x.g)).length, pm = recs.length ? 100 * missG / recs.length : 100;
  const optMiss = Object.keys(map).filter(k => k !== "timestamp" && k !== "glucose").map(k => {
    const miss = data.filter(r => { const c = String(r[map[k]] === undefined ? "" : r[map[k]]).trim(); return c === "" || /^(na|n\/a|nan|null)$/i.test(c); }).length;
    return miss ? esc(header[map[k]]) + " " + fx(100 * miss / Math.max(1, data.length), 0) + "%" : null; }).filter(Boolean);
  let coverage = "";
  if (sorted.length >= 2 && isFinite(medIv) && medIv > 0) {
    const expected = Math.round((sorted[sorted.length - 1].ms - sorted[0].ms) / 60000 / medIv) + 1;
    coverage = ` Coverage: ${gVals.length} numeric readings for ${expected} expected at a ${fx(medIv, 0)}-min cadence.`;
  }
  const v5 = pm < 5 ? "PASS" : (pm <= 20 ? "WARN" : "FAIL");
  checks.push({n: "5. Missingness", v: v5, d: `${missG} of ${recs.length} glucose values missing or non-numeric (${fx(pm, 1)}%).${optMiss.length ? " Missing in optional columns: " + optMiss.join(", ") + "." : ""}${coverage}`});
  renderIngest(name, parsed, checks, {header: header, data: data, sorted: sorted, gVals: gVals});
}
function renderIngest(name, parsed, checks, res) {
  const fails = checks.filter(c => c.v === "FAIL").length, warns = checks.filter(c => c.v === "WARN").length;
  const verdict = fails ? `<div class="verdict" style="background:#FEE2E2;color:#991B1B;">${esc(name)}: ${fails} check(s) failed. Fix the file before using it.</div>`
    : (warns ? `<div class="verdict" style="background:#FEF3C7;color:#92400E;">${esc(name)}: usable with ${warns} warning(s).</div>`
             : `<div class="verdict" style="background:#DCFCE7;color:#166534;">${esc(name)}: all 5 checks passed.</div>`);
  let kpis = "", preview = "";
  const GB = BOUNDS.glucose;
  const gIn = res ? res.gVals.filter(v => v >= GB[0] && v <= GB[1]) : [];
  if (res && res.gVals.length) {
    const g = gIn, nOut = res.gVals.length - gIn.length;
    const pctOf = f => g.length ? fx(100 * g.filter(f).length / g.length, 1) + "%" : "–";
    const span = res.sorted.length ? esc(msTs(res.sorted[0].ms)) + " → " + esc(msTs(res.sorted[res.sorted.length - 1].ms)) : "–";
    const of = "of in-bounds readings (" + GB[0] + "–" + GB[1] + " mg/dL)";
    kpis = `<div class="grid-4">
      <div class="gt-card stat"><div class="stat-label">In-bounds glucose readings</div><div class="stat-value tabular-nums">${g.length} <span class="unit">of ${res.gVals.length} numeric</span></div><div class="stat-sub">${nOut ? nOut + " outside " + GB[0] + "–" + GB[1] + " mg/dL excluded from these tiles · " : ""}${span}</div></div>
      <div class="gt-card stat"><div class="stat-label">Mean glucose</div><div class="stat-value tabular-nums">${g.length ? fx(mean(g)) : "–"} <span class="unit">mg/dL</span></div><div class="stat-sub">${g.length ? "min " + fx(minOf(g)) + " · max " + fx(maxOf(g)) + " · " : ""}${of}</div></div>
      <div class="gt-card stat"><div class="stat-label">Time in range 70–180</div><div class="stat-value tabular-nums">${pctOf(v => v >= C.hypo && v <= C.target_high)}</div><div class="stat-sub">${of}</div></div>
      <div class="gt-card stat"><div class="stat-label">Below 70 · above 250</div><div class="stat-value tabular-nums">${pctOf(v => v < C.hypo)} · ${pctOf(v => v > C.hyper)}</div><div class="stat-sub">${of}</div></div></div>`;
  }
  if (!res && parsed.rows.length) {
    preview = `<div class="gt-card"><div class="card-title">First lines of the file as read</div><div class="table-wrap"><table class="data-table"><tbody>${parsed.rows.slice(0, 6).map(r => "<tr>" + r.slice(0, 12).map(c => "<td>" + esc(c) + "</td>").join("") + "</tr>").join("")}</tbody></table></div></div>`;
  }
  if (res) {
    const cols = res.header.slice(0, 12);
    preview = `<div class="gt-card"><div class="card-title">First rows as parsed (${res.data.length} data rows)</div><div class="table-wrap"><table class="data-table">
      <thead><tr>${cols.map(c => "<th>" + esc(c) + "</th>").join("")}</tr></thead>
      <tbody>${res.data.slice(0, 10).map(r => "<tr>" + cols.map((_, j) => "<td>" + esc(r[j] === undefined ? "" : r[j]) + "</td>").join("") + "</tr>").join("")}</tbody></table></div></div>`;
  }
  $("ingestResult").innerHTML = verdict + kpis +
    `<div class="gt-card"><div class="card-title">Validation checks</div>${checks.map(c => `<div class="check"><div><b>${c.n}</b><div class="d">${c.d}</div></div>${badge(c.v)}</div>`).join("")}</div>` +
    (res && res.gVals.length ? '<div class="gt-card"><div class="card-title">Uploaded glucose</div><div id="ingestChart" class="chart" style="height:260px;"></div></div>' : "") + preview;
  if (res && res.gVals.length && plotOk("ingestChart")) {
    const pts = res.sorted.filter(x => isFinite(x.g)), gx = [], gy = [];
    pts.forEach((x, k) => { if (k > 0 && (x.ms - pts[k - 1].ms) / 60000 > 30) { gx.push(msTs(x.ms - 60000)); gy.push(null); } gx.push(msTs(x.ms)); gy.push(x.g); });
    const yMax = Math.max(200, Math.ceil((maxOf(res.gVals) + 25) / 50) * 50);
    // Markers only for small files: thousands of SVG markers make large uploads slow to draw.
    Plotly.react("ingestChart", [{x: gx, y: gy, mode: pts.length > 2000 ? "lines" : "lines+markers", connectgaps: false, line: {color: "#0D9488", width: pts.length > 2000 ? 1 : 2}, marker: {size: 5},
      hovertemplate: "%{x}<br>%{y:.1f} mg/dL<extra></extra>"}],
      layout({shapes: bands(yMax), xaxis: axis({type: "date"}), yaxis: axis({range: [Math.min(40, minOf(res.gVals) - 10), yMax], title: {text: "mg/dL", font: {size: 11}}})}), CFG);
  }
}
(function bindDrop() {
  const z = $("dropZone");
  ["dragenter", "dragover"].forEach(e => z.addEventListener(e, ev => { ev.preventDefault(); z.classList.add("drag"); }));
  ["dragleave", "drop"].forEach(e => z.addEventListener(e, ev => { ev.preventDefault(); z.classList.remove("drag"); }));
  z.addEventListener("drop", ev => { const f = ev.dataTransfer && ev.dataTransfer.files[0]; if (f) handleFile(f); });
})();

/* ---------------- cohort ---------------- */
function renderCohortKpis() {
  const E = D.ehr, n = E.length;
  const h = E.map(e => e.baseline_hba1c), m = mean(h), sd = Math.sqrt(h.reduce((a, v) => a + (v - m) * (v - m), 0) / (n - 1));
  const count = f => E.filter(f).length, share = k => fx(100 * k / n, 0) + "%";
  const tiers = ["Low Risk", "Moderate Risk", "High Risk"].map(t => count(e => e.clinical_risk_tier === t));
  const ins = count(e => e.insulin_regimen !== "None"), su = count(e => e.primary_medication.includes("Sulfonylurea"));
  const ins_ids = E.filter(e => e.insulin_regimen !== "None").map(e => e.patient_id);
  const held = new Set(D.board.filter(b => b.split === "held_out").map(b => b.id));
  const tile = (l, v, s) => `<div class="gt-card stat"><div class="stat-label">${l}</div><div class="stat-value tabular-nums">${v}</div><div class="stat-sub">${s}</div></div>`;
  $("cohortKpis").innerHTML =
    tile("Synthetic patients", n, "Each with an EHR record and 7 days of 15-min wearable data") +
    tile("Mean baseline HbA1c", fx(m, 2) + "%", "SD " + fx(sd, 2) + " · " + count(e => e.baseline_hba1c >= 7) + " (" + share(count(e => e.baseline_hba1c >= 7)) + ") at or above 7.0%") +
    tile("Hypo-risk tier (sulfonylurea/insulin, age)", tiers[2] + ' <span class="unit">high</span>', tiers[1] + " moderate · " + tiers[0] + " low") +
    tile("Held-out patients", held.size, (n - held.size) + " training patients; held-out ones are never seen in training") +
    tile("Sulfonylurea users", su, share(su) + " of the cohort; a hypoglycaemia risk factor") +
    tile("Insulin-treated", ins, ins_ids.filter(i => held.has(i)).length + " of them in the held-out split") +
    tile("eGFR below 60", count(e => e.egfr < 60), "Lowest eGFR in the cohort: " + fx(Math.min(...E.map(e => e.egfr))) + " mL/min/1.73m²") +
    tile("Hypo episode in past 12 m", count(e => e.prior_hypo_episodes_12m > 0), "Family history of T2D: " + count(e => e.family_history_t2d === "Yes") + " · TCF7L2 risk allele carriers: " + count(e => e.tcf7l2_rs7903146_risk_alleles > 0));
}
function renderCohortCharts() {
  const E = D.ehr;
  if (plotOk("hba1cChart")) {
    Plotly.react("hba1cChart", [{x: E.map(e => e.baseline_hba1c), type: "histogram", xbins: {size: 0.25}, marker: {color: "#0D9488", line: {color: "#fff", width: 1}},
      hovertemplate: "HbA1c %{x}%<br>%{y} patients<extra></extra>"}],
      layout({margin: {l: 40, r: 10, t: 18, b: 40}, shapes: [{type: "line", x0: 7, x1: 7, yref: "paper", y0: 0, y1: 1, line: {color: "#991B1B", width: 2, dash: "dash"}}],
        annotations: [{x: 7, y: 1, yref: "paper", text: "7.0%", showarrow: false, xanchor: "left", yanchor: "bottom", font: {size: 10, color: "#991B1B"}}],
        xaxis: axis({title: {text: "HbA1c (%)", font: {size: 11}}}), yaxis: axis({title: {text: "patients", font: {size: 11}}})}), CFG);
  }
  if (plotOk("tierChart")) {
    const t = ["Low Risk", "Moderate Risk", "High Risk"], v = t.map(x => E.filter(e => e.clinical_risk_tier === x).length);
    Plotly.react("tierChart", [{x: t.map(tierLabel), y: v, type: "bar", text: v.map(String), textposition: "outside", marker: {color: ["#10B981", "#F59E0B", "#DC2626"]},
      hovertemplate: "%{x}: %{y} patients<extra></extra>"}],
      layout({margin: {l: 40, r: 10, t: 18, b: 40}, xaxis: axis({showgrid: false}), yaxis: axis({range: [0, Math.max(...v) * 1.18], title: {text: "patients", font: {size: 11}}})}), CFG);
  }
  if (plotOk("medChart")) {
    const counts = {}; E.forEach(e => { counts[e.primary_medication] = (counts[e.primary_medication] || 0) + 1; });
    const meds = Object.keys(counts).sort((a, b) => counts[a] - counts[b] || (a < b ? -1 : 1));
    const short = s => s.replace("Lifestyle & Diet / ", "").replace("Metformin", "Met").replace(" (Sitagliptin)", "").replace(" (Dapagliflozin)", "");
    Plotly.react("medChart", [{y: meds.map(short), x: meds.map(k => counts[k]), customdata: meds, type: "bar", orientation: "h", marker: {color: "#0D9488"},
      text: meds.map(k => String(counts[k])), textposition: "outside", cliponaxis: false, hovertemplate: "%{customdata}: %{x} patients<extra></extra>"}],
      layout({margin: {l: 10, r: 24, t: 10, b: 36}, xaxis: axis({range: [0, Math.max(...meds.map(k => counts[k])) * 1.2], title: {text: "patients", font: {size: 11}}}), yaxis: axis({showgrid: false, tickfont: {size: 10, color: "#334155"}})}), CFG);
  }
}
function toggleBoard() { boardAll = !boardAll; renderBoard(); }
function renderBoard() {
  const B = D.board, rows = boardAll ? B : B.slice(0, 20);
  $("boardToggle").textContent = boardAll ? "Show top 20" : "Show all " + B.length;
  const nLow = B.filter(b => b.cgm < C.hypo).length, nHigh = B.filter(b => b.cgm > C.hyper).length;
  $("boardNote").innerHTML = `Sorted with patients whose CGM is already out of range first (${nLow} below 70, ${nHigh} above 250 mg/dL now), then by the higher of the two risk scores. Scores are uncalibrated (alert at hypo ≥ ${THR.hypo} or hyper ≥ ${THR.hyper}). The tier is the EHR's hypoglycaemia-risk tier (sulfonylurea or insulin, and age), not an overall risk. Latest window: ${esc(B[0].ts)} (synthetic). Training patients are in-sample for the model, so their scores are optimistic. Rows marked "Open" are demo patients you can replay.`;
  const nowTag = g => g < C.level2 ? '<span class="now-tag now-low">Level 2 low now (&lt;54)</span>' : (g < C.hypo ? '<span class="now-tag now-low">Below 70 now</span>'
    : (g > C.hyper ? '<span class="now-tag now-high">Already &gt;250 now</span>' : (g > C.target_high ? '<span class="muted">180–250</span>' : '<span class="muted">70–180</span>')));
  const al = a => a === 1 ? '<span class="al-1">Hypo alert</span>' : (a === 2 ? '<span class="al-2">Hyper alert</span>' : '<span class="al-0">none</span>');
  $("boardTable").innerHTML = `<thead><tr><th>#</th><th>Patient</th><th>Hypo-risk tier</th><th>Split</th><th>CGM now</th><th>State now</th><th>+2 h forecast</th><th>Hypo score</th><th>Hyper score</th><th>Alert</th><th>Medication</th><th></th></tr></thead><tbody>` +
    rows.map((b, i) => `<tr${b.demo >= 0 ? ' class="clickable" onclick="openDemo(' + b.demo + ')"' : ""}><td class="tabular-nums">${i + 1}</td>
      <td><b>${esc(b.id)}</b><br><span class="muted">${esc(b.name)}</span></td><td><span class="tier tier-${b.tier.split(" ")[0]}" title="${TIER_HELP}">${esc(tierLabel(b.tier))}</span></td>
      <td>${b.split === "held_out" ? "held-out" : "training"}</td><td class="tabular-nums" style="color:${gState(b.cgm).color};font-weight:700;">${fx(b.cgm)}</td>
      <td>${nowTag(b.cgm)}</td><td class="tabular-nums">${fx(b.f2h)}</td><td class="tabular-nums">${fx(b.ph, 3)}</td><td class="tabular-nums">${fx(b.pr, 3)}</td><td>${al(b.alert)}</td>
      <td class="small">${esc(b.med)}${b.insulin !== "None" ? "<br>+ " + esc(b.insulin) : ""}</td><td>${b.demo >= 0 ? '<button class="btn">Open</button>' : ""}</td></tr>`).join("") + "</tbody>";
}
function openDemo(i) { switchPatient(i); switchSection(0); }

/* ---------------- model card charts ---------------- */
function renderModelCardCharts() {
  if (plotOk("mcMaeChart")) {
    const k = Object.keys(D.mc.mae), v = k.map(x => D.mc.mae[x]);
    Plotly.react("mcMaeChart", [{y: k, x: v, type: "bar", orientation: "h", marker: {color: ["#0D9488", "#94A3B8", "#CBD5E1"]}, text: v.map(x => fx(x) + " mg/dL"), textposition: "outside", cliponaxis: false,
      hovertemplate: "%{y}: %{x:.2f} mg/dL<extra></extra>"}],
      layout({margin: {l: 10, r: 20, t: 10, b: 40}, xaxis: axis({range: [0, Math.max(...v) * 1.45], title: {text: "mean absolute error at +2 h (mg/dL)", font: {size: 11}}}), yaxis: axis({showgrid: false, autorange: "reversed"})}), CFG);
  }
  if (plotOk("mcClarkeChart")) {
    const z = ["A", "B", "C", "D", "E"], series = Object.keys(D.mc.clarke), cols = ["#0D9488", "#94A3B8"];
    Plotly.react("mcClarkeChart", series.map((s, i) => ({x: z.map(q => "Zone " + q), y: z.map(q => D.mc.clarke[s][q]), name: s, type: "bar", marker: {color: cols[i % cols.length]},
      text: z.map(q => fx(D.mc.clarke[s][q])), textposition: "outside", hovertemplate: s + " · %{x}: %{y:.2f}%<extra></extra>"})),
      layout({showlegend: true, barmode: "group", bargap: 0.25, bargroupgap: 0.06, legend: {orientation: "h", x: 0, y: -0.2}, margin: {l: 40, r: 10, t: 10, b: 40},
        xaxis: axis({showgrid: false}), yaxis: axis({range: [0, 110], title: {text: "% of windows", font: {size: 11}}})}), CFG);
  }
}

function showToast(msg) { const t = $("toast"); t.textContent = msg; t.classList.add("show"); clearTimeout(showToast._t); showToast._t = setTimeout(() => t.classList.remove("show"), 3200); }

/* ---------------- init ---------------- */
(function init() {
  const sel = $("patientSelect");
  D.patients.forEach((p, i) => { const o = document.createElement("option"); o.value = String(i); o.textContent = p.id + " · " + p.name + " · " + (p.tag === "held_out" ? "held-out" : "training") + (p.on_insulin ? " · insulin" : ""); sel.appendChild(o); });
  $("rulesList").innerHTML = [
    "Baseline: straight line from the current CGM reading to the ML model's +2 h forecast.",
    "Carbohydrate: +" + R.carb_rise_mg_dl_per_g + " mg/dL per gram at the peak, " + R.carb_peak_min + " min after eating, tailing off afterwards.",
    "Walking: −" + R.walk_drop_mg_dl_per_min + " mg/dL per minute walked, ×" + R.walk_multiplier_su_or_insulin + " if the EHR lists a sulfonylurea or insulin.",
    "Insulin (insulin users only): −" + R.insulin_drop_mg_dl_per_unit + " mg/dL per unit when fully absorbed; absorption time constant " + R.insulin_time_constant_min + " min.",
    "Results are clipped to " + R.floor_mg_dl + "–" + R.ceiling_mg_dl + " mg/dL. The numbers are hand-set for illustration and not fitted to any data."
  ].map(s => "<li>" + esc(s) + "</li>").join("");
  renderCohortKpis();
  renderBoard();
  switchPatient(0);
  switchSection(0);
})();
</script>
</body>
</html>
"""


def build_html():
    ehr, wear = load_data(os.path.join(ROOT, "data"))
    if not os.path.exists(BUNDLE_PATH):
        raise SystemExit("models/glycotwin_model.joblib is missing; run models/train_fusion_model.py first.")
    bundle = joblib.load(BUNDLE_PATH)
    with open(METRICS_PATH, encoding="utf-8") as f:
        metrics = json.load(f)
    check_inputs(bundle, metrics)

    print("Computing wearable features and scoring every window with the shipped bundle...")
    fused, _ = prepare_fused_dataset(ehr, compute_wearable_features(wear))
    fused = fused.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    X = encode_features(fused, bundle["feature_cols"])
    P = bundle["classifier"].predict_proba(X)
    A = apply_alert_thresholds(P, bundle["alert_thresholds"])
    R = bundle["regressor"].predict(X)

    # Sanity check: this pipeline must reproduce the held-out MAE recorded in metrics.json.
    te = fused["patient_id"].isin(bundle["test_patients"]).to_numpy()
    mae = float(np.mean(np.abs(fused.loc[te, "future_glucose_2h"].to_numpy() - R[te])))
    if abs(mae - metrics["holdout"]["regressor"]["mae"]) > 0.01:
        raise SystemExit(f"Builder held-out MAE {mae:.3f} differs from metrics.json; inference is out of sync.")

    data, insulin_ids, insulin_example = build_payload(ehr, fused, P, A, R, bundle, metrics)
    json_data = json.dumps(data, ensure_ascii=True, separators=(",", ":")).replace("<", "\\u003c")
    html = (TEMPLATE
            .replace("%%MODEL_CARD%%", model_card_html(metrics, bundle, ehr, insulin_ids))
            .replace("%%DATA_JSON%%", json_data))

    with open(OUT_HTML, "w", encoding="utf-8", newline="\n") as f:
        f.write(html)

    import plotly
    src = os.path.join(os.path.dirname(plotly.__file__), "package_data", "plotly.min.js")
    if os.path.exists(src):
        shutil.copyfile(src, OUT_PLOTLY)
    elif not os.path.exists(OUT_PLOTLY):
        print("Warning: plotly.min.js not found in the plotly package; charts will not render.")

    print(f"Wrote {os.path.relpath(OUT_HTML, ROOT)} ({len(html) / 1024:.0f} KB) and {os.path.relpath(OUT_PLOTLY, ROOT)}.")
    print(f"  Demo patients: {', '.join(p['id'] + ' [' + p['tag'] + ']' for p in data['patients'])}")
    print(f"  Insulin example: {insulin_example}; builder held-out MAE {mae:.2f} mg/dL matches metrics.json.")
    print("  Publish: cp dashboard/index.html public/index.html && cp dashboard/plotly.min.js public/plotly.min.js")


if __name__ == "__main__":
    build_html()
