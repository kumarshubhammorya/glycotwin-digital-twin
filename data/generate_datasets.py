"""
GlycoTwin: Ultra-fast Pure Python Synthetic Clinical & Wearable Data Generator
Zero C-extension dependencies (uses built-in random, math, csv, datetime).
Generates realistic EHR + 15-minute CGM, heart rate, HRV, steps, carbs, insulin telemetry.
Includes physiological counter-regulatory hepatic rescue to prevent unrealistic flatline hypoglycemia.
"""

import os
import csv
import math
import random
from datetime import datetime, timedelta

random.seed(42)

def generate_static_ehr(n_patients=200):
    first_names_m = ["Rajesh", "Amit", "Vikram", "Suresh", "Manoj", "Anand", "Ramesh", "Deepak", "Sunil", "Pradeep"]
    first_names_f = ["Pooja", "Sunita", "Ananya", "Meera", "Lakshmi", "Kavita", "Sita", "Priyanaka", "Geeta", "Rekha"]
    last_names = ["Sharma", "Verma", "Patel", "Reddy", "Iyer", "Nair", "Gupta", "Chatterjee", "Singh", "Deshmukh", "Kulkarni", "Rao"]
    diet_types = [
        "North Indian (High Carb/Roti)", "South Indian (Rice Predominant)", 
        "Vegetarian Balanced", "Non-Vegetarian/High Protein"
    ]
    diet_weights = [0.35, 0.35, 0.20, 0.10]
    
    patients = []
    for i in range(n_patients):
        pid = f"IND-PAT-{i+1:03d}"
        gender = "Male" if random.random() < 0.52 else "Female"
        first = random.choice(first_names_m) if gender == "Male" else random.choice(first_names_f)
        last = random.choice(last_names)
        name = f"{first} {last}"
        
        age = int(max(28, min(79, round(random.gauss(54, 11)))))
        bmi = round(max(18.5, min(38.0, random.gauss(27.4, 3.8))), 1)
        diet = random.choices(diet_types, weights=diet_weights, k=1)[0]
        
        # Diabetes duration (years)
        duration = round(max(0.5, min(25.0, random.expovariate(1.0 / 6.5))), 1)
        
        # Baseline HbA1c (%)
        hba1c = round(max(5.7, min(12.2, 5.8 + (duration * 0.11) + ((bmi - 24) * 0.07) + random.gauss(0, 0.5))), 2)
        
        # Fasting glucose (mg/dL)
        fasting_g = int(max(85, min(260, round(hba1c * 17.5 + random.gauss(0, 12)))))
        
        # Blood pressure
        systolic = int(max(105, min(175, round(118 + (age * 0.3) + ((bmi - 22) * 0.6) + random.gauss(0, 7)))))
        diastolic = int(max(65, min(105, round(74 + (age * 0.15) + random.gauss(0, 5)))))
        
        # eGFR (kidney function)
        egfr = round(max(38.0, min(120.0, 115 - (age * 0.5) - (duration * 0.7) + random.gauss(0, 6))), 1)
        
        # Pharmacotherapy
        if hba1c < 6.8 and duration < 3:
            med = "Lifestyle & Diet / Metformin 500mg"
            ins = "None"
        elif hba1c < 7.8:
            med = random.choice(["Metformin 1000mg", "Metformin + DPP4i (Sitagliptin)"])
            ins = "None"
        elif hba1c < 9.0:
            med = random.choice(["Metformin + Sulfonylurea (Glimepiride)", "Metformin + SGLT2i (Dapagliflozin)"])
            ins = "None"
        else:
            med = "Metformin + Dual Oral Agents"
            ins = random.choice(["Basal Insulin (Glargine 10-20U)", "Basal-Bolus (Glargine + Aspart)"])
            
        # Clinical risk tier
        if ins != "None" or "Sulfonylurea" in med:
            risk = "High Risk" if age > 58 else "Moderate Risk"
        else:
            risk = "Low Risk"
            
        patients.append({
            "patient_id": pid,
            "name": name,
            "age": age,
            "gender": gender,
            "diet_profile": diet,
            "bmi": bmi,
            "diabetes_duration_years": duration,
            "baseline_hba1c": hba1c,
            "fasting_glucose_mg_dl": fasting_g,
            "systolic_bp": systolic,
            "diastolic_bp": diastolic,
            "egfr": egfr,
            "primary_medication": med,
            "insulin_regimen": ins,
            "clinical_risk_tier": risk
        })
    return patients


def generate_wearable_stream(ehr_list, days=7, sampling_minutes=15):
    n_steps_per_day = (24 * 60) // sampling_minutes # 96 steps
    total_steps = days * n_steps_per_day
    start_time = datetime(2026, 9, 1, 0, 0)
    timestamps = [start_time + timedelta(minutes=i*sampling_minutes) for i in range(total_steps)]
    
    # Representative cohort of 40 patients
    cohort = ehr_list[:40]
    all_wearables = []
    
    for patient in cohort:
        pid = patient["patient_id"]
        base_g = float(patient["fasting_glucose_mg_dl"])
        is_insulin = patient["insulin_regimen"] != "None"
        is_su = "Sulfonylurea" in patient["primary_medication"]
        bmi = patient["bmi"]
        hba1c = patient["baseline_hba1c"]
        
        current_g = base_g + random.uniform(-10, 15)
        active_carbs_pool = 0.0
        active_insulin_pool = 0.0
        hypo_prone = (is_insulin or is_su) and (patient["age"] > 55 or random.random() > 0.4)
        
        patient_telemetry = []
        
        for ts in timestamps:
            hour = ts.hour
            minute = ts.minute
            
            # Sleep & Circadian
            is_sleeping = (hour >= 23 or hour < 6)
            sleep_stage = "Deep" if (is_sleeping and 1 <= hour <= 4) else ("Light" if is_sleeping else "Awake")
            
            if is_sleeping:
                heart_rate = int(random.gauss(62, 3))
                hrv = int(random.gauss(52, 4))
                steps = 0
            else:
                is_walk = (hour == 7 and minute < 45) or (hour == 18 and minute >= 30)
                if is_walk:
                    heart_rate = int(random.gauss(122, 8))
                    hrv = int(random.gauss(26, 3))
                    steps = int(max(400, random.gauss(1100, 180)))
                else:
                    heart_rate = int(random.gauss(75, 5))
                    hrv = int(random.gauss(42, 4))
                    steps = int(max(0, random.expovariate(1.0 / 45.0)))
                    
            # Meals
            meal_carbs = 0
            if hour == 8 and minute == 30: # Breakfast
                if random.random() > 0.08:
                    meal_carbs = int(max(20, random.gauss(45 + (12 if hba1c > 8.5 else 0), 10)))
            elif hour == 13 and minute == 15: # Lunch
                meal_carbs = int(max(30, random.gauss(65 + (15 if "High Carb" in patient["diet_profile"] else 0), 12)))
            elif hour == 17 and minute == 0: # Snack
                if random.random() > 0.35:
                    meal_carbs = int(max(10, random.gauss(25, 6)))
            elif hour == 20 and minute == 45: # Dinner
                meal_carbs = int(max(30, random.gauss(55 + (12 if hba1c > 8.5 else 0), 12)))
                
            active_carbs_pool += meal_carbs
            
            # Insulin / SU
            insulin_dose = 0
            if is_insulin:
                if hour == 21 and minute == 30:
                    insulin_dose = int(14 + (hba1c - 7.0) * 2)
                    active_insulin_pool += insulin_dose * 0.6
                elif meal_carbs > 30 and "Bolus" in patient["insulin_regimen"]:
                    insulin_dose = int(max(1, meal_carbs / 10.0 + random.choice([-1, 0, 1])))
                    active_insulin_pool += insulin_dose
            elif is_su:
                if hour in [8, 20] and minute == 30:
                    active_insulin_pool += 4.5
                    
            # Kinetics
            absorbed = min(active_carbs_pool, active_carbs_pool * 0.28)
            active_carbs_pool -= absorbed
            carb_flux = absorbed * 1.5 * (bmi / 24.0) * (hba1c / 7.0)
            
            utilized_ins = min(active_insulin_pool, active_insulin_pool * 0.20)
            active_insulin_pool -= utilized_ins
            insulin_flux = utilized_ins * 5.2
            
            # Exercise burn
            ex_factor = 1.6 if (is_insulin or is_su) else 1.1
            exercise_burn = (steps / 1000.0) * 6.0 * ex_factor
            
            # Nocturnal dip
            nocturnal_dip = 0.0
            if is_sleeping and 2 <= hour <= 4 and hypo_prone:
                nocturnal_dip = random.uniform(3.0, 6.0)
                hrv = max(18, hrv - 8)
                
            # Homeostatic reversion towards fasting baseline
            reversion = (base_g - current_g) * 0.065
            
            # Endogenous hepatic rescue: Liver releases glucose when sugar drops below 75 mg/dL
            hepatic_rescue = max(0.0, (75.0 - current_g) * 0.38)
            
            noise = random.gauss(0, 1.5)
            
            current_g = current_g + carb_flux - insulin_flux - exercise_burn - nocturnal_dip + reversion + hepatic_rescue + noise
            current_g = max(52.0, min(350.0, current_g))
            
            patient_telemetry.append({
                "patient_id": pid,
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "glucose_cgm": round(current_g, 1),
                "heart_rate": heart_rate,
                "hrv": max(15, hrv),
                "steps_15m": steps,
                "sleep_stage": sleep_stage,
                "meal_carbs_g": meal_carbs,
                "insulin_units": insulin_dose
            })
            
        # Add 2-hour future target (8 steps = 120 mins)
        n_p_steps = len(patient_telemetry)
        for idx in range(n_p_steps):
            if idx + 8 < n_p_steps:
                fut_g = patient_telemetry[idx + 8]["glucose_cgm"]
                patient_telemetry[idx]["future_glucose_2h"] = fut_g
                if fut_g < 70.0:
                    ev = 1 # Hypoglycemia
                elif fut_g > 250.0:
                    ev = 2 # Hyperglycemia
                else:
                    ev = 0 # In-Range
                patient_telemetry[idx]["adverse_event_2h"] = ev
                all_wearables.append(patient_telemetry[idx])
                
    return all_wearables


def main():
    os.makedirs("data", exist_ok=True)
    print("Generating 200 Indian Patient EHR records...")
    ehr_list = generate_static_ehr(200)
    ehr_keys = list(ehr_list[0].keys())
    with open("data/synthetic_ehr.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ehr_keys)
        writer.writeheader()
        writer.writerows(ehr_list)
    print(f"Saved data/synthetic_ehr.csv with {len(ehr_list)} records.")

    print("Generating Dynamic Wearables (15-min CGM, HR, HRV, Steps, 2h future target)...")
    wearables_list = generate_wearable_stream(ehr_list, days=7, sampling_minutes=15)
    w_keys = list(wearables_list[0].keys())
    with open("data/synthetic_wearables.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=w_keys)
        writer.writeheader()
        writer.writerows(wearables_list)
    print(f"Saved data/synthetic_wearables.csv with {len(wearables_list)} continuous records.")

    counts = {0: 0, 1: 0, 2: 0}
    for r in wearables_list:
        counts[r["adverse_event_2h"]] += 1
    print("\nAdverse event distribution (2 hours ahead):")
    print(f"0 - In-Range (70-180 mg/dL): {counts[0]} ({counts[0]/len(wearables_list)*100:.1f}%)")
    print(f"1 - Impending Hypo (<70 mg/dL): {counts[1]} ({counts[1]/len(wearables_list)*100:.1f}%)")
    print(f"2 - Impending Hyper (>250 mg/dL): {counts[2]} ({counts[2]/len(wearables_list)*100:.1f}%)")
    print("Data Generation Complete!")

if __name__ == "__main__":
    main()
