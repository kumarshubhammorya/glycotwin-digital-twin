"""
GlycoTwin: Multimodal AI Model Training & Validation Engine
Fuses Static EHR features with Dynamic Wearable time-series features to predict:
1. Adverse Glycemic Event (Classification: In-Range vs. Severe Hypoglycemia vs. Severe Hyperglycemia) 2 hours in advance.
2. Exact Blood Glucose Trajectory (Regression: mg/dL) 2 hours in advance.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import classification_report, mean_absolute_error, r2_score, f1_score

def compute_wearable_features(df_wearables):
    """
    Extracts physiological temporal features from the continuous wearable time-series.
    """
    df = df_wearables.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by=["patient_id", "timestamp"]).reset_index(drop=True)
    
    # Rolling features per patient (1 hour = 4 steps of 15m; 2 hours = 8 steps)
    grouped = df.groupby("patient_id")
    
    # 1. Glucose rate of change (Velocity dG/dt)
    df["glucose_velocity_15m"] = grouped["glucose_cgm"].diff(1).fillna(0)
    df["glucose_velocity_30m"] = grouped["glucose_cgm"].diff(2).fillna(0)
    
    # 2. Rolling averages and variability
    df["glucose_mean_1h"] = grouped["glucose_cgm"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    df["glucose_std_1h"] = grouped["glucose_cgm"].transform(lambda s: s.rolling(4, min_periods=1).std()).fillna(0)
    
    # 3. Heart Rate and HRV patterns
    df["hr_mean_1h"] = grouped["heart_rate"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    df["hrv_mean_1h"] = grouped["hrv"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    
    # 4. Activity and metabolic load
    df["steps_sum_1h"] = grouped["steps_15m"].transform(lambda s: s.rolling(4, min_periods=1).sum())
    df["steps_sum_2h"] = grouped["steps_15m"].transform(lambda s: s.rolling(8, min_periods=1).sum())
    df["carbs_sum_2h"] = grouped["meal_carbs_g"].transform(lambda s: s.rolling(8, min_periods=1).sum())
    df["insulin_sum_2h"] = grouped["insulin_units"].transform(lambda s: s.rolling(8, min_periods=1).sum())
    
    # Hour of day (circadian feature)
    df["hour_of_day"] = df["timestamp"].dt.hour
    
    return df

def prepare_fused_dataset(df_ehr, df_wearables_features):
    """
    Merges static EHR clinical attributes with dynamic wearable telemetry.
    """
    # Merge on patient_id
    fused = pd.merge(df_wearables_features, df_ehr, on="patient_id", how="inner")
    
    # Categorical encodings
    categorical_cols = [
        "gender", "diet_profile", "primary_medication", 
        "insulin_regimen", "clinical_risk_tier", "sleep_stage"
    ]
    
    for col in categorical_cols:
        fused[col] = fused[col].astype("category")
        
    feature_cols = [
        # Real-time wearable metrics
        "glucose_cgm", "glucose_velocity_15m", "glucose_velocity_30m",
        "glucose_mean_1h", "glucose_std_1h", "heart_rate", "hr_mean_1h",
        "hrv", "hrv_mean_1h", "steps_15m", "steps_sum_1h", "steps_sum_2h",
        "meal_carbs_g", "carbs_sum_2h", "insulin_units", "insulin_sum_2h",
        "hour_of_day", "sleep_stage",
        # Static clinical EHR metrics
        "age", "gender", "bmi", "diabetes_duration_years", "baseline_hba1c",
        "fasting_glucose_mg_dl", "systolic_bp", "diastolic_bp", "egfr",
        "diet_profile", "primary_medication", "insulin_regimen", "clinical_risk_tier"
    ]
    
    return fused, feature_cols

def train_models():
    os.makedirs("models", exist_ok=True)
    
    print("Loading datasets...")
    df_ehr = pd.read_csv("data/synthetic_ehr.csv")
    df_wearables = pd.read_csv("data/synthetic_wearables.csv")
    
    print("Extracting dynamic wearable features...")
    df_features = compute_wearable_features(df_wearables)
    
    print("Performing multimodal data fusion (EHR + Wearables)...")
    fused_df, feature_cols = prepare_fused_dataset(df_ehr, df_features)
    
    # Patient-level train-test split to ensure no data leakage across the same patient
    unique_patients = np.array(list(fused_df["patient_id"].unique()), dtype=str)
    train_patients, test_patients = train_test_split(unique_patients, test_size=0.25, random_state=42)
    
    train_mask = fused_df["patient_id"].isin(train_patients)
    test_mask = fused_df["patient_id"].isin(test_patients)
    
    X_train = fused_df.loc[train_mask, feature_cols].copy()
    y_train_cls = fused_df.loc[train_mask, "adverse_event_2h"].to_numpy()
    y_train_reg = fused_df.loc[train_mask, "future_glucose_2h"].to_numpy()
    
    X_test = fused_df.loc[test_mask, feature_cols].copy()
    y_test_cls = fused_df.loc[test_mask, "adverse_event_2h"].to_numpy()
    y_test_reg = fused_df.loc[test_mask, "future_glucose_2h"].to_numpy()
    
    print(f"Dataset split: {len(X_train)} train samples, {len(X_test)} test samples.")
    print(f"Training on {len(feature_cols)} multimodal clinical & wearable features.")
    
    # Identify categorical feature indices for HistGradientBoosting
    cat_features = [col for col in feature_cols if X_train[col].dtype.name == "category"]
    print(f"Categorical features detected: {cat_features}")
    
    # Convert category dtypes to integer codes for sklearn HistGradientBoosting
    cat_indices = []
    for c in cat_features:
        X_train[c] = X_train[c].cat.codes.astype("int32")
        X_test[c] = X_test[c].cat.codes.astype("int32")
        cat_indices.append(feature_cols.index(c))

    # 1. Train Classification Model (Predicting In-Range vs Hypo vs Hyper 2h ahead)
    print("\nTraining HistGradientBoosting Classifier (2-hour Adverse Event Predictor)...")
    clf = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_depth=6,
        categorical_features=cat_indices,
        class_weight="balanced", # Essential for rare critical events like severe hypoglycemia
        random_state=42
    )
    clf.fit(X_train, y_train_cls)
    
    y_pred_cls = clf.predict(X_test)
    print("\n--- CLASSIFICATION PERFORMANCE REPORT (2-Hour Horizon) ---")
    target_names = ["0: In-Range (70-180)", "1: Hypoglycemia (<70)", "2: Hyperglycemia (>250)"]
    print(classification_report(y_test_cls, y_pred_cls, target_names=target_names, digits=3))
    
    # 2. Train Continuous Regression Model (Predicting exact glucose value 2h ahead)
    print("Training HistGradientBoosting Regressor (2-hour Continuous Glucose Forecaster)...")
    reg = HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.08,
        max_depth=6,
        categorical_features=cat_indices,
        random_state=42
    )
    reg.fit(X_train, y_train_reg)
    
    y_pred_reg = reg.predict(X_test)
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    r2 = r2_score(y_test_reg, y_pred_reg)
    print(f"\n--- REGRESSION PERFORMANCE ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f} mg/dL")
    print(f"R-squared (R2 Score): {r2:.3f}")
    
    # 3. Save model package
    model_package = {
        "classifier": clf,
        "regressor": reg,
        "feature_cols": feature_cols,
        "cat_features": cat_features,
        "metrics": {
            "mae": float(mae),
            "r2": float(r2),
            "macro_f1": float(f1_score(y_test_cls, y_pred_cls, average="macro"))
        }
    }
    
    joblib_path = "models/glycotwin_model.joblib"
    joblib.dump(model_package, joblib_path)
    print(f"\nSuccessfully exported trained AI model bundle to {joblib_path}")
    print("Stage 2 Complete!")

if __name__ == "__main__":
    train_models()
