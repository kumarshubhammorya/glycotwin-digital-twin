"""
GlycoTwin: prediction checks for the shipped model bundle.

Run from anywhere:  python models/test_predictions.py

Read-only: loads data/*.csv, models/glycotwin_model.joblib and models/metrics.json,
and writes nothing (no model, no metrics, no bytecode cache). Every check uses the
same helpers as training and the dashboards (compute_wearable_features ->
prepare_fused_dataset -> encode_features -> apply_alert_thresholds).

Checks
  (a) The metrics stored in the bundle, and the held-out numbers in
      models/metrics.json, match a fresh re-score of the 50 held-out patients.
  (b) IND-PAT-196 (held out, sulfonylurea): a hypoglycaemia alert fires in the
      2 h before its first CGM reading < 70 mg/dL, while CGM is still >= 70.
  (c) IND-PAT-077 (held out): a hyperglycaemia alert fires in the 2 h before
      its first CGM reading > 250 mg/dL, while CGM is still <= 250.
  (d) IND-PAT-046 (held out, stable): no alert on any stable window
      (no event in the next 2 h, CGM 80-160 mg/dL, change <= 3 mg/dL per 15 min).
  (e) encode_features maps every category that appears in the CSVs to a
      non-negative code, and the bundle stores the same category levels.

Prints PASS/FAIL per check and exits with status 1 if any check fails.
Synthetic data only. Research prototype, not a medical device.
"""

import json
import os
import sys

sys.dont_write_bytecode = True  # keep the run read-only (no __pycache__)

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (average_precision_score, f1_score, mean_absolute_error,  # noqa: E402
                             r2_score, roc_auc_score)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "models"))
from train_fusion_model import (CATEGORY_LEVELS, HYPER_MG_DL, HYPO_MG_DL,  # noqa: E402
                                apply_alert_thresholds, compute_wearable_features,
                                encode_features, load_data, prepare_fused_dataset)

BUNDLE_PATH = os.path.join(ROOT, "models", "glycotwin_model.joblib")
METRICS_PATH = os.path.join(ROOT, "models", "metrics.json")
HORIZON = 8  # 8 x 15 min = 2 h

results = []


def check(name, ok, detail):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    for line in detail:
        print(f"       {line}")


def close(a, b, tol):
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


def first_onset(frame, cond):
    """Row position of the first reading that meets cond, or None."""
    hits = np.flatnonzero(cond(frame["glucose_cgm"].to_numpy()))
    return int(hits[0]) if len(hits) else None


def early_warning(frame, pid, k, cond, label):
    """Is a class-k alert raised 15-120 min before the patient's first out-of-range reading?"""
    g = frame[frame["patient_id"] == pid].reset_index(drop=True)
    if g.empty:
        return False, [f"{pid} not found in the held-out data"]
    s = first_onset(g, cond)
    if s is None:
        return False, [f"{pid} has no {label} reading in the data"]
    lo = max(0, s - HORIZON)
    pre = g.iloc[lo:s]
    hits = pre[(pre["alert"] == k) & ~cond(pre["glucose_cgm"].to_numpy())]
    detail = [f"first {label} reading: {g.loc[s, 'timestamp']} at {g.loc[s, 'glucose_cgm']:.1f} mg/dL"]
    if s < HORIZON:
        detail.append(f"only {s} windows of history before it (less than 2 h)")
    if len(hits):
        t0 = hits.iloc[0]
        lead = (s - hits.index[0]) * 15
        detail.append(f"first alert: {t0['timestamp']} (CGM {t0['glucose_cgm']:.1f} mg/dL, "
                      f"risk score {t0['score']:.2f}), {lead} min before the first {label} reading")
        detail.append(f"{len(hits)} of {len(pre)} windows in the preceding 2 h carried this alert")
    else:
        detail.append(f"no {label} alert in the {len(pre)} windows before it")
    return len(hits) > 0, detail


def main():
    for p in (BUNDLE_PATH, METRICS_PATH):
        if not os.path.exists(p):
            print(f"[FAIL] missing {os.path.relpath(p, ROOT)}. Run models/train_fusion_model.py "
                  f"and models/evaluate.py first.")
            sys.exit(1)

    bundle = joblib.load(BUNDLE_PATH)
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    clf, reg = bundle["classifier"], bundle["regressor"]
    cols, thresholds = bundle["feature_cols"], bundle["alert_thresholds"]

    ehr, wear = load_data(os.path.join(ROOT, "data"))
    test_ids = set(bundle["test_patients"])
    print(f"Bundle: {len(cols)} features, alert thresholds {thresholds}, "
          f"{len(bundle['train_patients'])} training / {len(test_ids)} held-out patients\n")

    # Score the held-out patients only (features are computed per patient, so this
    # gives exactly the rows evaluate.py scores).
    held = wear[wear["patient_id"].isin(test_ids)]
    fused, _ = prepare_fused_dataset(ehr, compute_wearable_features(held))
    fused = fused.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    X = encode_features(fused, cols)
    proba = clf.predict_proba(X)
    alerts = apply_alert_thresholds(proba, thresholds)
    pred = reg.predict(X)
    y = fused["adverse_event_2h"].to_numpy()
    yr = fused["future_glucose_2h"].to_numpy()
    fused["alert"] = alerts

    # ---------- (a) stored metrics == fresh re-score ----------
    mae, r2 = mean_absolute_error(yr, pred), r2_score(yr, pred)
    f1 = f1_score(y, alerts, average="macro")
    stored = bundle["metrics"]
    h = metrics["holdout"]
    rows = [
        ("bundle holdout_mae", mae, stored.get("holdout_mae"), 0.006),
        ("bundle holdout_r2", r2, stored.get("holdout_r2"), 0.0006),
        ("bundle holdout_macro_f1_alerts", f1, stored.get("holdout_macro_f1_alerts"), 0.0006),
        ("metrics.json regressor mae", mae, h["regressor"]["mae"], 0.006),
        ("metrics.json regressor r2", r2, h["regressor"]["r2"], 0.0006),
        ("metrics.json held-out windows", len(fused), h["windows"], 0),
    ]
    for k, key in ((1, "hypo_lt70"), (2, "hyper_gt250")):
        yb, ab = (y == k), (alerts == k)
        tp, fp, fn = int((yb & ab).sum()), int((~yb & ab).sum()), int((yb & ~ab).sum())
        m = h["classifier"][key]
        rows += [
            (f"metrics.json {key} sensitivity", tp / max(tp + fn, 1), m["sensitivity"], 0.0006),
            (f"metrics.json {key} precision", tp / max(tp + fp, 1), m["precision"], 0.0006),
            (f"metrics.json {key} ROC-AUC", roc_auc_score(yb, proba[:, k]), m["roc_auc"], 0.0006),
            (f"metrics.json {key} PR-AUC", average_precision_score(yb, proba[:, k]), m["pr_auc"], 0.0006),
        ]
    ok = all(close(a, b, tol) for _, a, b, tol in rows) and thresholds == h["alert_thresholds"]
    detail = [f"{n}: re-scored {a if isinstance(a, int) else round(float(a), 4)} vs stored {b}"
              + ("" if close(a, b, tol) else "  <-- mismatch") for n, a, b, tol in rows]
    detail.append(f"alert thresholds: bundle {thresholds} vs metrics.json {h['alert_thresholds']}")
    check("(a) stored metrics match a re-score of the held-out set", ok, detail)

    # ---------- (b) hypo early warning for IND-PAT-196 ----------
    fused["score"] = proba[:, 1]
    ok, detail = early_warning(fused, "IND-PAT-196", 1, lambda g: g < HYPO_MG_DL, "<70 mg/dL")
    check("(b) IND-PAT-196: hypo alert within 2 h before the first <70 mg/dL reading", ok, detail)

    # ---------- (c) hyper early warning for IND-PAT-077 ----------
    fused["score"] = proba[:, 2]
    ok, detail = early_warning(fused, "IND-PAT-077", 2, lambda g: g > HYPER_MG_DL, ">250 mg/dL")
    check("(c) IND-PAT-077: hyper alert within 2 h before the first >250 mg/dL reading", ok, detail)

    # ---------- (d) no alert on stable windows of IND-PAT-046 ----------
    g = fused[fused["patient_id"] == "IND-PAT-046"]
    stable = g[(g["adverse_event_2h"] == 0) & g["glucose_cgm"].between(80, 160)
               & (g["glucose_velocity_15m"].abs() <= 3)]
    n_alert = int((stable["alert"] != 0).sum())
    ok = len(stable) > 0 and n_alert == 0
    detail = [f"{len(stable)} stable windows, {n_alert} with an alert",
              f"all IND-PAT-046 windows: {int((g['alert'] != 0).sum())} alerts in {len(g)}"]
    if len(stable):
        pidx = stable.index
        detail.append(f"highest risk scores on stable windows: hypo {proba[pidx, 1].max():.3f}, "
                      f"hyper {proba[pidx, 2].max():.3f} (thresholds {thresholds['hypo']} / {thresholds['hyper']})")
    check("(d) IND-PAT-046: no alert on stable windows", ok, detail)

    # ---------- (e) every CSV category gets a non-negative code ----------
    all_fused, _ = prepare_fused_dataset(ehr, wear)
    detail, ok = [], True
    for c, levels in CATEGORY_LEVELS.items():
        uniq = all_fused[[c]].drop_duplicates().reset_index(drop=True)
        codes = encode_features(uniq, [c])[c].to_numpy(dtype=float)
        bad = [v for v, code in zip(uniq[c].astype(str), codes) if np.isnan(code) or code < 0]
        ok &= not bad
        detail.append(f"{c}: {len(uniq)} distinct values in the CSVs, {len(levels)} known levels"
                      + (f", UNMAPPED {bad}" if bad else ", all mapped"))
    text_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(fused[c])]
    uncovered = [c for c in text_cols if c not in CATEGORY_LEVELS]
    ok &= not uncovered
    detail.append(f"text-valued model features: {len(text_cols)}, "
                  + (f"WITHOUT category levels: {uncovered}" if uncovered else "all have fixed category levels"))
    same = bundle.get("category_levels") == CATEGORY_LEVELS
    ok &= same
    detail.append("bundle category_levels " + ("match" if same else "DIFFER FROM") + " train_fusion_model.CATEGORY_LEVELS")
    check("(e) encode_features maps every CSV category to a non-negative code", ok, detail)

    n_fail = results.count(False)
    print(f"\n{len(results) - n_fail} of {len(results)} checks passed.")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
