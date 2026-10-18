"""
GlycoTwin: Reproducible Evaluation
Run from the repository root:  python models/evaluate.py

Scores the shipped model bundle (models/glycotwin_model.joblib) on its held-out
patients, then runs 5-fold patient-grouped cross-validation, naive baselines,
a Clarke Error Grid, episode-level early-warning analysis, feature ablations and
a latency benchmark. Everything is written to models/metrics.json.
Every performance number in the README, slides and video should come from that file.
"""

import json
import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_fusion_model import (EHR_FEATURES, FEATURE_COLS, HYPER_MG_DL, HYPO_MG_DL,  # noqa: E402
                                WEARABLE_FEATURES, apply_alert_thresholds, compute_wearable_features,
                                encode_features, load_data, make_models, prepare_fused_dataset)

HORIZON = 8  # 15-min steps in 2 hours


def clarke_zones(ref, pred):
    """Clarke Error Grid zones (Clarke et al., Diabetes Care 1987), mg/dL. Returns % per zone."""
    z = []
    for r, p in zip(ref, pred):
        if (r <= 70 and p <= 70) or (0.8 * r <= p <= 1.2 * r):
            z.append("A")
        elif (r >= 180 and p <= 70) or (r <= 70 and p >= 180):
            z.append("E")
        elif ((70 <= r <= 290) and p >= r + 110) or ((130 <= r <= 180) and p <= 1.4 * r - 182):
            z.append("C")
        elif (r >= 240 and 70 <= p <= 180) or (r <= 175 / 3 and 70 <= p <= 180) or ((175 / 3 <= r <= 70) and p >= 1.2 * r):
            z.append("D")
        else:
            z.append("B")
    s = pd.Series(z).value_counts(normalize=True)
    return {k: round(float(s.get(k, 0.0)) * 100, 2) for k in "ABCDE"}


def reg_metrics(y, p):
    return {"mae": round(float(mean_absolute_error(y, p)), 2),
            "rmse": round(float(mean_squared_error(y, p) ** 0.5), 2),
            "r2": round(float(r2_score(y, p)), 3),
            "clarke_pct": clarke_zones(y, p)}


def binary_metrics(y_true, alert, score=None):
    tp = int(((alert == 1) & (y_true == 1)).sum()); fp = int(((alert == 1) & (y_true == 0)).sum())
    fn = int(((alert == 0) & (y_true == 1)).sum()); tn = int(((alert == 0) & (y_true == 0)).sum())
    out = {"positives": int(y_true.sum()), "tp": tp, "fp": fp, "fn": fn,
           "sensitivity": round(tp / max(tp + fn, 1), 3),
           "specificity": round(tn / max(tn + fp, 1), 4),
           "precision": round(tp / max(tp + fp, 1), 3)}
    if score is not None and 0 < y_true.sum() < len(y_true):
        out["roc_auc"] = round(float(roc_auc_score(y_true, score)), 3)
        out["pr_auc"] = round(float(average_precision_score(y_true, score)), 3)
        out["prevalence"] = round(float(y_true.mean()), 4)
    return out


def cls_metrics(y, proba, alerts):
    cm = confusion_matrix(y, alerts, labels=[0, 1, 2])
    out = {"confusion_matrix_rows_true_cols_alert": cm.tolist(),
           "macro_f1": round(float(f1_score(y, alerts, average="macro")), 3)}
    for k, name in ((1, "hypo_lt70"), (2, "hyper_gt250")):
        out[name] = binary_metrics((y == k).astype(int), (alerts == k).astype(int), proba[:, k])
    return out


def episodes(glucose, cond):
    """Start indices of contiguous runs where cond(glucose) is true."""
    flag = cond(glucose)
    starts = np.where(flag & ~np.r_[False, flag[:-1]])[0]
    return starts, flag


def episode_metrics(frame, alerts, k, cond):
    """For each event episode in held-out patients: was an alert for that class raised
    15-120 minutes BEFORE the first out-of-range reading, while the patient was not yet in the event?"""
    caught, leads, n_ep, false_alarm_edges, alarm_edges = 0, [], 0, 0, 0
    patient_days = 0.0
    for pid, g in frame.groupby("patient_id", sort=False):
        idx = g.index.to_numpy()
        glu = g["glucose_cgm"].to_numpy()
        a = (alerts[idx] == k)
        lab = g["adverse_event_2h"].to_numpy() == k
        patient_days += len(g) * 15 / 1440
        starts, in_event = episodes(glu, cond)
        for s in starts:
            if s < HORIZON:
                continue  # not enough history to judge an early warning
            n_ep += 1
            window = range(s - HORIZON, s)
            hits = [j for j in window if a[j] and not in_event[j]]
            if hits:
                caught += 1
                leads.append((s - hits[0]) * 15)
        edges = np.where(a & ~np.r_[False, a[:-1]])[0]
        alarm_edges += len(edges)
        false_alarm_edges += int(sum(1 for e in edges if not lab[e]))
    return {"episodes": n_ep, "warned_15_to_120_min_ahead": caught,
            "episode_sensitivity": round(caught / max(n_ep, 1), 3),
            "median_lead_time_min": float(np.median(leads)) if leads else None,
            "alarms_per_patient_day": round(alarm_edges / max(patient_days, 1e-9), 2),
            "false_alarm_fraction": round(false_alarm_edges / max(alarm_edges, 1), 3)}


def rule_baseline(frame):
    """Simple clinical rule without ML: hypo if CGM < 90 and falling; hyper if CGM > 200 and rising."""
    g = frame["glucose_cgm"].to_numpy(); v = frame["glucose_velocity_15m"].to_numpy()
    out = np.zeros(len(frame), dtype=int)
    out[(g > 200) & (v > 0)] = 2
    out[(g < 90) & (v < 0)] = 1
    return out


def main():
    bundle = joblib.load("models/glycotwin_model.joblib")
    clf, reg, thresholds = bundle["classifier"], bundle["regressor"], bundle["alert_thresholds"]
    cols = bundle["feature_cols"]

    ehr, wear = load_data()
    fused, _ = prepare_fused_dataset(ehr, compute_wearable_features(wear))
    fused = fused.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    X = encode_features(fused, cols)
    y = fused["adverse_event_2h"].to_numpy()
    yr = fused["future_glucose_2h"].to_numpy()
    groups = fused["patient_id"].to_numpy()
    te = np.isin(groups, bundle["test_patients"])
    tr = np.isin(groups, bundle["train_patients"])

    # ---------- Held-out evaluation of the SHIPPED bundle ----------
    P = clf.predict_proba(X[te]); A = apply_alert_thresholds(P, thresholds); R = reg.predict(X[te])
    hold = fused[te].reset_index(drop=True)
    assert abs(mean_absolute_error(yr[te], R) - bundle["metrics"]["holdout_mae"]) < 0.01, \
        "Shipped bundle does not match the shipped data. Re-run models/train_fusion_model.py."
    onset = ((hold["glucose_cgm"] >= HYPO_MG_DL) & (hold["glucose_cgm"] <= HYPER_MG_DL)).to_numpy()
    g_now = hold["glucose_cgm"].to_numpy()
    lin = np.clip(g_now + 8 * hold["glucose_velocity_15m"].to_numpy(), 40, 400)
    rule = rule_baseline(hold)

    holdout = {
        "split": "patient-level 75/25 (random_state=42); thresholds chosen on training patients only",
        "test_patients": len(bundle["test_patients"]), "windows": int(te.sum()),
        "alert_thresholds": thresholds,
        "classifier": cls_metrics(y[te], P, A),
        "classifier_not_already_in_event": cls_metrics(y[te][onset], P[onset], A[onset]),
        "episodes_hypo": episode_metrics(hold, A, 1, lambda g: g < HYPO_MG_DL),
        "episodes_hyper": episode_metrics(hold, A, 2, lambda g: g > HYPER_MG_DL),
        "rule_baseline_events": cls_metrics(y[te], np.eye(3)[rule], rule),
        "regressor": reg_metrics(yr[te], R),
        "baseline_persistence": reg_metrics(yr[te], g_now),
        "baseline_linear_extrapolation": reg_metrics(yr[te], lin),
    }
    print("Held-out evaluation done.")

    # EHR-swap test: give each held-out patient another patient's EHR at inference.
    rng = np.random.RandomState(0)
    tp = np.array(bundle["test_patients"]); perm = dict(zip(tp, rng.permutation(tp)))
    swapped_ehr = ehr.set_index("patient_id").loc[hold["patient_id"].map(perm)].reset_index(drop=True)
    hold_sw = hold.copy()
    for c in EHR_FEATURES:
        hold_sw[c] = swapped_ehr[c].to_numpy()
    Xs = encode_features(hold_sw, cols)
    holdout["ehr_swap_test"] = {
        "description": "held-out patients scored with a shuffled patient's EHR; worse scores mean the model uses the EHR",
        "regressor_mae": round(float(mean_absolute_error(yr[te], reg.predict(Xs))), 2),
        "hypo_pr_auc": round(float(average_precision_score(y[te] == 1, clf.predict_proba(Xs)[:, 1])), 3),
        "hyper_pr_auc": round(float(average_precision_score(y[te] == 2, clf.predict_proba(Xs)[:, 2])), 3),
    }

    # ---------- 5-fold patient-grouped cross-validation (all 200 patients) ----------
    oofP = np.zeros((len(X), 3)); oofR = np.zeros(len(X))
    for a, b in GroupKFold(n_splits=5).split(X, y, groups):
        c_, r_ = make_models(cols)
        c_.fit(X.iloc[a], y[a]); r_.fit(X.iloc[a], yr[a])
        oofP[b] = c_.predict_proba(X.iloc[b]); oofR[b] = r_.predict(X.iloc[b])
    oofA = apply_alert_thresholds(oofP, thresholds)
    cv = {"folds": 5, "grouping": "patient",
          "note": "uses the shipped alert thresholds (chosen on the 150 training patients)",
          "classifier": cls_metrics(y, oofP, oofA),
          "regressor": reg_metrics(yr, oofR),
          "baseline_persistence": reg_metrics(yr, fused["glucose_cgm"].to_numpy())}
    print("Cross-validation done.")

    # ---------- Feature ablations (5-fold patient-grouped CV over all patients) ----------
    hist_gen = ["prior_hypo_episodes_12m", "family_history_t2d", "tcf7l2_rs7903146_risk_alleles"]
    physio = ["heart_rate", "hr_mean_1h", "hrv", "hrv_mean_1h", "sleep_stage"]
    cgm = ["glucose_cgm", "glucose_velocity_15m", "glucose_velocity_30m", "glucose_mean_1h", "glucose_std_1h"]
    configs = {
        "fused_all_features": cols,
        "wearables_only": [c for c in cols if c in WEARABLE_FEATURES],
        "ehr_only_plus_hour": [c for c in cols if c in EHR_FEATURES] + ["hour_of_day"],
        "without_cgm": [c for c in cols if c not in cgm],
        "without_hr_hrv_sleep": [c for c in cols if c not in physio],
        "without_history_and_genetics": [c for c in cols if c not in hist_gen],
        "without_hour_of_day": [c for c in cols if c != "hour_of_day"],
    }
    ablation = {}
    for name, fc in configs.items():
        Xa = encode_features(fused, fc)
        if fc == cols:
            Pa, Ra = oofP, oofR
        else:
            Pa = np.zeros((len(Xa), 3)); Ra = np.zeros(len(Xa))
            for a, b in GroupKFold(n_splits=5).split(Xa, y, groups):
                c_, r_ = make_models(fc)
                c_.fit(Xa.iloc[a], y[a]); r_.fit(Xa.iloc[a], yr[a])
                Pa[b] = c_.predict_proba(Xa.iloc[b]); Ra[b] = r_.predict(Xa.iloc[b])
        ablation[name] = {"features": len(fc),
                          "regressor_mae": round(float(mean_absolute_error(yr, Ra)), 2),
                          "hypo_pr_auc": round(float(average_precision_score(y == 1, Pa[:, 1])), 3),
                          "hyper_pr_auc": round(float(average_precision_score(y == 2, Pa[:, 2])), 3)}
        print(f"Ablation {name}: {ablation[name]}")

    # ---------- Latency ----------
    row = X[te].iloc[[0]]
    for _ in range(20):
        clf.predict_proba(row); reg.predict(row)
    t = []
    for _ in range(200):
        t0 = time.perf_counter(); clf.predict_proba(row); reg.predict(row); t.append((time.perf_counter() - t0) * 1000)
    batch = X[te].iloc[:5000]
    t0 = time.perf_counter(); clf.predict_proba(batch); reg.predict(batch)
    per_row_us = (time.perf_counter() - t0) / len(batch) * 1e6

    label_counts = {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    metrics = {
        "data": {
            "patients": int(fused["patient_id"].nunique()), "windows": int(len(fused)),
            "days_per_patient": 7, "sampling_minutes": 15,
            "features": {"total": len(cols), "wearable": len([c for c in cols if c in WEARABLE_FEATURES]),
                         "ehr": len([c for c in cols if c in EHR_FEATURES])},
            "label_counts": label_counts,
            "labels": {"0": "no event in next 2 h (70-250 mg/dL)",
                       "1": "any CGM reading < 70 mg/dL in next 2 h",
                       "2": "any CGM reading > 250 mg/dL in next 2 h"},
            "cgm_readings_lt54": int((fused["glucose_cgm"] < 54).sum()),
            "cgm_readings_lt70": int((fused["glucose_cgm"] < 70).sum()),
            "patients_with_any_lt70": int(fused.loc[fused["glucose_cgm"] < 70, "patient_id"].nunique()),
        },
        "holdout": holdout,
        "grouped_cv_5fold": cv,
        "ablation_grouped_cv_5fold": ablation,
        "latency": {"single_window_ms_median": round(float(np.median(t)), 2),
                    "single_window_ms_p95": round(float(np.percentile(t, 95)), 2),
                    "batched_us_per_window": round(float(per_row_us), 1),
                    "hardware_note": "CPU only; measured on the machine that ran this script"},
        "notes": [
            "All data is synthetic (data/generate_datasets.py); results do not transfer to real patients.",
            "Classifier probabilities are not calibrated; the dashboards label them as risk scores.",
        ],
    }
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    h = holdout
    print("\n================ GlycoTwin metrics (held-out patients) ================")
    print(f"Patients {metrics['data']['patients']} | windows {metrics['data']['windows']} | labels {label_counts}")
    print(f"2 h forecast MAE {h['regressor']['mae']} mg/dL (persistence {h['baseline_persistence']['mae']}), "
          f"R2 {h['regressor']['r2']}, Clarke A {h['regressor']['clarke_pct']['A']}% / "
          f"A+B {round(h['regressor']['clarke_pct']['A'] + h['regressor']['clarke_pct']['B'], 2)}%")
    for k in ("hypo_lt70", "hyper_gt250"):
        m = h["classifier"][k]
        print(f"{k}: sensitivity {m['sensitivity']} precision {m['precision']} ROC-AUC {m.get('roc_auc')} PR-AUC {m.get('pr_auc')}")
    print(f"Hypo episodes warned 15-120 min ahead: {h['episodes_hypo']['warned_15_to_120_min_ahead']}/{h['episodes_hypo']['episodes']} "
          f"(median lead {h['episodes_hypo']['median_lead_time_min']} min)")
    print(f"Hyper episodes warned 15-120 min ahead: {h['episodes_hyper']['warned_15_to_120_min_ahead']}/{h['episodes_hyper']['episodes']} "
          f"(median lead {h['episodes_hyper']['median_lead_time_min']} min)")
    print(f"Latency: {metrics['latency']['single_window_ms_median']} ms per single window (median)")
    print("Wrote models/metrics.json")


if __name__ == "__main__":
    main()
