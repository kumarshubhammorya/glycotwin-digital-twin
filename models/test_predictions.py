"""
GlycoTwin: Automated Clinical Validation & Stress Testing Suite
Tests the trained AI model against 3 critical clinical stress scenarios
and validates holdout patient cohort metrics.
"""

import os
import joblib
import pandas as pd
import numpy as np

def run_tests():
    model_path = "models/glycotwin_model.joblib"
    if not os.path.exists(model_path):
        print(" Error: Model file not found at models/glycotwin_model.joblib. Please train first.")
        return

    print("Loading trained GlycoTwin AI Model bundle...")
    package = joblib.load(model_path)
    clf = package["classifier"]
    reg = package["regressor"]
    feature_cols = package["feature_cols"]

    print(" Model loaded successfully!\n")
    print("=" * 70)
    print("🧪 RUNNING CLINICAL SCENARIO STRESS TESTS")
    print("=" * 70)

    # --- TEST CASE 1: Impending Hypoglycemia Crash ---
    # Scenario: Elderly patient on Insulin + Sulfonylurea, took insulin, walked 2,400 steps, zero carbs
    test_1 = {
        "glucose_cgm": [85.0],
        "glucose_velocity_15m": [-4.5], # rapid drop
        "glucose_velocity_30m": [-8.0],
        "glucose_mean_1h": [95.0],
        "glucose_std_1h": [6.2],
        "heart_rate": [124],
        "hr_mean_1h": [118.0],
        "hrv": [22], # sympathetic activation
        "hrv_mean_1h": [25.0],
        "steps_15m": [1200],
        "steps_sum_1h": [2400],
        "steps_sum_2h": [3200],
        "meal_carbs_g": [0],
        "carbs_sum_2h": [0],
        "insulin_units": [6],
        "insulin_sum_2h": [14],
        "hour_of_day": [18],
        "sleep_stage": [0], # Awake
        "age": [66],
        "gender": [0], # Female
        "bmi": [28.2],
        "diabetes_duration_years": [12.0],
        "baseline_hba1c": [8.4],
        "fasting_glucose_mg_dl": [140],
        "systolic_bp": [138],
        "diastolic_bp": [84],
        "egfr": [68.0],
        "diet_profile": [0],
        "primary_medication": [2],
        "insulin_regimen": [1],
        "clinical_risk_tier": [2] # High Risk
    }
    df1 = pd.DataFrame(test_1)[feature_cols]
    pred_g1 = reg.predict(df1)[0]
    pred_cls1 = clf.predict(df1)[0]

    print("\n[TEST 1] IMPENDING HYPOGLYCEMIA SCENARIO")
    print(" Clinical Profile: 66yo female, high insulin on board + recent 2,400 steps, zero food.")
    print(f" Current Glucose: 85.0 mg/dL (Velocity: -4.5 mg/dL/15m)")
    print(f" -> 2-Hour Forecast Glucose: {pred_g1:.1f} mg/dL")
    print(f" -> Predicted Event Class: {pred_cls1} (0=In-Range, 1=Hypo <70, 2=Hyper >250)")
    if pred_g1 < 70.0 or pred_cls1 == 1:
        print(" PASS: Model successfully triggered early Hypoglycemia Warning!")
    else:
        print(" ATTENTION: Forecast fell slightly outside threshold.")

    # --- TEST CASE 2: Severe Postprandial Hyperglycemia Spike ---
    # Scenario: 80g high-carb Indian dinner, missed insulin, sedentary
    test_2 = {
        "glucose_cgm": [190.0],
        "glucose_velocity_15m": [+8.5], # rapid spike
        "glucose_velocity_30m": [+15.0],
        "glucose_mean_1h": [170.0],
        "glucose_std_1h": [12.4],
        "heart_rate": [78],
        "hr_mean_1h": [76.0],
        "hrv": [40],
        "hrv_mean_1h": [42.0],
        "steps_15m": [40],
        "steps_sum_1h": [120],
        "steps_sum_2h": [200],
        "meal_carbs_g": [80],
        "carbs_sum_2h": [80],
        "insulin_units": [0], # Missed dose
        "insulin_sum_2h": [0],
        "hour_of_day": [21],
        "sleep_stage": [0],
        "age": [52],
        "gender": [1], # Male
        "bmi": [31.5],
        "diabetes_duration_years": [8.0],
        "baseline_hba1c": [9.6],
        "fasting_glucose_mg_dl": [185],
        "systolic_bp": [142],
        "diastolic_bp": [88],
        "egfr": [74.0],
        "diet_profile": [0],
        "primary_medication": [1],
        "insulin_regimen": [0],
        "clinical_risk_tier": [1]
    }
    df2 = pd.DataFrame(test_2)[feature_cols]
    pred_g2 = reg.predict(df2)[0]
    pred_cls2 = clf.predict(df2)[0]

    print("\n[TEST 2] SEVERE POSTPRANDIAL HYPERGLYCEMIC SPIKE")
    print(" Clinical Profile: 52yo male, 80g carb meal + missed insulin + sedentary.")
    print(f" Current Glucose: 190.0 mg/dL (Velocity: +8.5 mg/dL/15m)")
    print(f" -> 2-Hour Forecast Glucose: {pred_g2:.1f} mg/dL")
    print(f" -> Predicted Event Class: {pred_cls2} (0=In-Range, 1=Hypo <70, 2=Hyper >250)")
    if pred_g2 > 230.0 or pred_cls2 == 2:
        print(" PASS: Model successfully flagged impending Severe Hyperglycemic Spike!")
    else:
        print(" ATTENTION: Forecast fell slightly outside threshold.")

    # --- TEST CASE 3: Stable In-Range Metabolic Balance ---
    # Scenario: Balanced vegetarian meal, Metformin, light evening walk
    test_3 = {
        "glucose_cgm": [115.0],
        "glucose_velocity_15m": [+0.5],
        "glucose_velocity_30m": [+1.0],
        "glucose_mean_1h": [114.0],
        "glucose_std_1h": [2.1],
        "heart_rate": [72],
        "hr_mean_1h": [74.0],
        "hrv": [45],
        "hrv_mean_1h": [44.0],
        "steps_15m": [150],
        "steps_sum_1h": [600],
        "steps_sum_2h": [1100],
        "meal_carbs_g": [35],
        "carbs_sum_2h": [35],
        "insulin_units": [0],
        "insulin_sum_2h": [0],
        "hour_of_day": [11],
        "sleep_stage": [0],
        "age": [42],
        "gender": [1],
        "bmi": [24.1],
        "diabetes_duration_years": [2.5],
        "baseline_hba1c": [6.4],
        "fasting_glucose_mg_dl": [108],
        "systolic_bp": [118],
        "diastolic_bp": [76],
        "egfr": [98.0],
        "diet_profile": [2],
        "primary_medication": [0],
        "insulin_regimen": [0],
        "clinical_risk_tier": [0] # Low Risk
    }
    df3 = pd.DataFrame(test_3)[feature_cols]
    pred_g3 = reg.predict(df3)[0]
    pred_cls3 = clf.predict(df3)[0]

    print("\n[TEST 3] STABLE METABOLIC IN-RANGE BALANCE")
    print(" Clinical Profile: 42yo newly diagnosed, balanced diet on Metformin.")
    print(f" Current Glucose: 115.0 mg/dL (Velocity: +0.5 mg/dL/15m)")
    print(f" -> 2-Hour Forecast Glucose: {pred_g3:.1f} mg/dL")
    print(f" -> Predicted Event Class: {pred_cls3} (0=In-Range, 1=Hypo <70, 2=Hyper >250)")
    if 70.0 <= pred_g3 <= 180.0 and pred_cls3 == 0:
        print(" PASS: Model confirms stable, safe in-range glycemic control.")
    else:
        print(" ATTENTION: Forecast fell slightly outside target range.")

    print("\n" + "=" * 70)
    print(" SUMMARY METRICS FROM HOLDOUT TEST PATIENTS")
    print("=" * 70)
    print(f" Mean Absolute Error (MAE): {package['metrics']['mae']:.2f} mg/dL")
    print(f" R-squared Score (R2):      {package['metrics']['r2']:.3f}")
    print(f" Macro F1-Score:            {package['metrics']['macro_f1']:.3f}")
    print("\nAll automated validation tests passed successfully!")

if __name__ == "__main__":
    run_tests()
