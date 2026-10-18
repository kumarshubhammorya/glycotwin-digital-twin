"""
GlycoTwin: Multimodal Model Training
Fuses static EHR features with dynamic wearable time-series features to predict:
1. Adverse glycaemic event in the next 2 hours (classification:
   0 = no event, 1 = hypoglycaemia <70 mg/dL, 2 = hyperglycaemia >250 mg/dL).
2. Glucose value exactly 2 hours ahead (regression, mg/dL).

Run from the repository root:  python models/train_fusion_model.py
The model is trained on the training patients only, so the held-out patients
shown in the dashboards are genuinely unseen. Full evaluation (cross-validation,
baselines, Clarke grid, ablations) lives in models/evaluate.py.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import classification_report, mean_absolute_error, r2_score, f1_score, fbeta_score

HYPO_MG_DL = 70.0
HYPER_MG_DL = 250.0
CLASS_NAMES = {0: "No event (70-250 mg/dL)", 1: "Hypoglycaemia (<70 mg/dL)", 2: "Hyperglycaemia (>250 mg/dL)"}

# Fixed category levels so training and every inference path encode identically.
CATEGORY_LEVELS = {
    "sleep_stage": ["Awake", "Light", "Deep", "REM"],
    "gender": ["Female", "Male"],
    "diet_profile": [
        "North Indian (High Carb/Roti)", "South Indian (Rice Predominant)",
        "Vegetarian Balanced", "Non-Vegetarian/High Protein",
    ],
    "primary_medication": [
        "Lifestyle & Diet / Metformin 500mg", "Metformin 1000mg", "Metformin + DPP4i (Sitagliptin)",
        "Metformin + Sulfonylurea (Gliclazide)", "Metformin + Sulfonylurea (Glimepiride)",
        "Metformin + SGLT2i (Dapagliflozin)", "Metformin + Dual Oral Agents",
    ],
    "insulin_regimen": ["None", "Basal Insulin (Glargine 10-20U)", "Basal-Bolus (Glargine + Aspart)"],
    "clinical_risk_tier": ["Low Risk", "Moderate Risk", "High Risk"],
    "family_history_t2d": ["No", "Yes"],
}

WEARABLE_FEATURES = [
    "glucose_cgm", "glucose_velocity_15m", "glucose_velocity_30m",
    "glucose_mean_1h", "glucose_std_1h", "heart_rate", "hr_mean_1h",
    "hrv", "hrv_mean_1h", "steps_15m", "steps_sum_1h", "steps_sum_2h",
    "meal_carbs_g", "carbs_sum_2h", "insulin_units", "insulin_sum_2h",
    "hour_of_day", "sleep_stage",
]
EHR_FEATURES = [
    "age", "gender", "bmi", "diabetes_duration_years", "baseline_hba1c",
    "fasting_glucose_mg_dl", "systolic_bp", "diastolic_bp", "egfr",
    "diet_profile", "primary_medication", "insulin_regimen", "clinical_risk_tier",
    "prior_hypo_episodes_12m", "family_history_t2d", "tcf7l2_rs7903146_risk_alleles",
]
FEATURE_COLS = WEARABLE_FEATURES + EHR_FEATURES
CAT_FEATURES = [c for c in FEATURE_COLS if c in CATEGORY_LEVELS]

HGB_PARAMS = dict(max_iter=150, learning_rate=0.08, max_depth=6, random_state=42)


def load_data(data_dir="data"):
    # keep_default_na=False keeps the literal "None" insulin regimen as a string, not NaN.
    ehr = pd.read_csv(os.path.join(data_dir, "synthetic_ehr.csv"), keep_default_na=False)
    wear = pd.read_csv(os.path.join(data_dir, "synthetic_wearables.csv"))
    return ehr, wear


def compute_wearable_features(df_wearables):
    """Backward-looking temporal features per patient (no future information)."""
    df = df_wearables.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by=["patient_id", "timestamp"]).reset_index(drop=True)

    # 1 hour = 4 steps of 15 min; 2 hours = 8 steps
    grouped = df.groupby("patient_id")
    df["glucose_velocity_15m"] = grouped["glucose_cgm"].diff(1).fillna(0)
    df["glucose_velocity_30m"] = grouped["glucose_cgm"].diff(2).fillna(0)
    df["glucose_mean_1h"] = grouped["glucose_cgm"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    df["glucose_std_1h"] = grouped["glucose_cgm"].transform(lambda s: s.rolling(4, min_periods=1).std()).fillna(0)
    df["hr_mean_1h"] = grouped["heart_rate"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    df["hrv_mean_1h"] = grouped["hrv"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    df["steps_sum_1h"] = grouped["steps_15m"].transform(lambda s: s.rolling(4, min_periods=1).sum())
    df["steps_sum_2h"] = grouped["steps_15m"].transform(lambda s: s.rolling(8, min_periods=1).sum())
    df["carbs_sum_2h"] = grouped["meal_carbs_g"].transform(lambda s: s.rolling(8, min_periods=1).sum())
    df["insulin_sum_2h"] = grouped["insulin_units"].transform(lambda s: s.rolling(8, min_periods=1).sum())
    df["hour_of_day"] = df["timestamp"].dt.hour
    return df


def prepare_fused_dataset(df_ehr, df_wearables_features):
    """Merges static EHR attributes onto every wearable row (patient-level fusion)."""
    fused = pd.merge(df_wearables_features, df_ehr, on="patient_id", how="inner")
    return fused, list(FEATURE_COLS)


def encode_features(df, feature_cols=FEATURE_COLS):
    """Returns the model matrix with categoricals as fixed integer codes.
    Unknown categories become -1, which HistGradientBoosting treats as missing."""
    X = df[feature_cols].copy()
    for c in feature_cols:
        if c in CATEGORY_LEVELS:
            X[c] = pd.Categorical(X[c].astype(str), categories=CATEGORY_LEVELS[c]).codes.astype("int32")
            X[c] = X[c].where(X[c] >= 0, np.nan)
    return X


def categorical_indices(feature_cols=FEATURE_COLS):
    return [feature_cols.index(c) for c in feature_cols if c in CATEGORY_LEVELS]


def make_models(feature_cols=FEATURE_COLS):
    cat_idx = categorical_indices(feature_cols)
    clf = HistGradientBoostingClassifier(categorical_features=cat_idx, class_weight="balanced", **HGB_PARAMS)
    reg = HistGradientBoostingRegressor(categorical_features=cat_idx, **HGB_PARAMS)
    return clf, reg


def split_patients(fused, test_size=0.25, seed=42):
    pids = np.array(sorted(fused["patient_id"].unique()), dtype=str)
    train_p, test_p = train_test_split(pids, test_size=test_size, random_state=seed)
    return sorted(train_p), sorted(test_p)


def apply_alert_thresholds(proba, thresholds):
    """Alert rule: hypo if P(hypo) >= t_hypo, else hyper if P(hyper) >= t_hyper, else none."""
    out = np.zeros(len(proba), dtype=int)
    out[proba[:, 2] >= thresholds["hyper"]] = 2
    out[proba[:, 1] >= thresholds["hypo"]] = 1
    return out


def choose_thresholds(X, y, groups, n_splits=5, beta=2.0):
    """Picks per-class probability thresholds on out-of-fold predictions from the
    TRAINING patients only (F2 score: recall weighted above precision)."""
    oof = np.zeros((len(X), 3))
    for a, b in GroupKFold(n_splits=n_splits).split(X, y, groups):
        clf, _ = make_models(list(X.columns))
        clf.fit(X.iloc[a], y[a])
        oof[b] = clf.predict_proba(X.iloc[b])
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    thresholds = {}
    for k, name in ((1, "hypo"), (2, "hyper")):
        yb = (y == k).astype(int)
        scores = [fbeta_score(yb, (oof[:, k] >= t).astype(int), beta=beta, zero_division=0) for t in grid]
        thresholds[name] = float(grid[int(np.argmax(scores))])
    return thresholds


def train_models():
    print("Loading datasets...")
    df_ehr, df_wearables = load_data()

    print("Extracting dynamic wearable features...")
    df_features = compute_wearable_features(df_wearables)

    print("Fusing EHR + wearable streams...")
    fused, feature_cols = prepare_fused_dataset(df_ehr, df_features)
    X = encode_features(fused, feature_cols)
    y_cls = fused["adverse_event_2h"].to_numpy()
    y_reg = fused["future_glucose_2h"].to_numpy()

    # Patient-level split: no patient appears in both train and test.
    train_patients, test_patients = split_patients(fused)
    tr = fused["patient_id"].isin(train_patients).to_numpy()
    te = ~tr
    print(f"Split: {len(train_patients)} train patients ({tr.sum()} windows), "
          f"{len(test_patients)} held-out patients ({te.sum()} windows).")
    print(f"Features: {len(feature_cols)} ({len(WEARABLE_FEATURES)} wearable + {len(EHR_FEATURES)} EHR).")

    print("\nChoosing alert thresholds by 5-fold patient-grouped CV on training patients...")
    thresholds = choose_thresholds(X[tr].reset_index(drop=True), y_cls[tr], fused.loc[tr, "patient_id"].to_numpy())
    print(f"Thresholds: {thresholds}")

    print("\nTraining HistGradientBoosting classifier (2-hour event)...")
    clf, reg = make_models(feature_cols)
    clf.fit(X[tr], y_cls[tr])
    proba = clf.predict_proba(X[te])
    y_alert = apply_alert_thresholds(proba, thresholds)
    print("\n--- HELD-OUT CLASSIFICATION (alert thresholds applied) ---")
    print(classification_report(y_cls[te], y_alert, labels=[0, 1, 2],
                                target_names=[CLASS_NAMES[k] for k in range(3)], digits=3, zero_division=0))

    print("Training HistGradientBoosting regressor (glucose at +2 h)...")
    reg.fit(X[tr], y_reg[tr])
    y_pred = reg.predict(X[te])
    mae = mean_absolute_error(y_reg[te], y_pred)
    r2 = r2_score(y_reg[te], y_pred)
    print(f"Held-out MAE: {mae:.2f} mg/dL | R2: {r2:.3f}")

    package = {
        "classifier": clf,
        "regressor": reg,
        "feature_cols": feature_cols,
        "cat_features": CAT_FEATURES,
        "category_levels": CATEGORY_LEVELS,
        "class_names": CLASS_NAMES,
        "alert_thresholds": thresholds,
        "train_patients": list(train_patients),
        "test_patients": list(test_patients),
        "metrics": {
            "holdout_mae": round(float(mae), 2),
            "holdout_r2": round(float(r2), 3),
            "holdout_macro_f1_alerts": round(float(f1_score(y_cls[te], y_alert, average="macro")), 3),
        },
    }
    os.makedirs("models", exist_ok=True)
    joblib.dump(package, "models/glycotwin_model.joblib", compress=3)
    print("\nSaved models/glycotwin_model.joblib. Run `python models/evaluate.py` for the full metrics report.")


if __name__ == "__main__":
    train_models()
