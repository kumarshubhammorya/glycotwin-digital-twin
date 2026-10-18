"""
GlycoTwin: Pure-Python Synthetic Clinical & Wearable Data Generator
Zero C-extension dependencies (uses built-in random, math, csv, datetime).

Generates two linked, fully synthetic data streams for 200 Indian T2D personas:
  1. Static EHR: demographics, past diagnoses, lab results, medications and a
     genetic marker (TCF7L2 rs7903146 risk-allele count).
  2. Dynamic wearables: 15-minute CGM, heart rate, HRV, steps, sleep stage
     (Awake/Light/Deep/REM), meal carbohydrates and insulin doses for 7 days.

The glucose simulator is a simple compartment model (carb absorption, insulin /
sulfonylurea action, exercise, hepatic counter-regulation). Several EHR fields
change the dynamics on purpose, so the static record carries real signal:
  - TCF7L2 risk alleles and HbA1c raise post-meal excursions,
  - insulin or sulfonylurea use and age set hypoglycaemia propensity, which
    also drives the recorded prior-hypo history.
  - an eGFR < 60 pathway (slower insulin clearance, higher hypo propensity)
    exists in the code, but no persona in the shipped seed has eGFR < 60, so it
    never fires in the shipped data.
Wearable physiology is coupled back too: short sleep raises next-day insulin
resistance and lowers HRV, and low glucose triggers an adrenergic response
(heart rate up, HRV down).

Known quirks (kept so the shipped data stays reproducible; see README Limitations):
the meal bolus is skipped when the basal-insulin slot coincides with a meal over
30 g, and for early wake-ups the short-night resistance lands one day late.

All parameters are illustrative. Population ranges are loosely informed by
published Indian T2D literature but this is NOT a calibrated or validated
physiological model, and no real patient data is used.
"""

import os
import csv
import math
import random
from datetime import datetime, timedelta

random.seed(42)

N_PATIENTS = 200
DAYS = 7
SAMPLING_MINUTES = 15
HORIZON_STEPS = 8  # 8 x 15 min = 2 hours
HYPO_MG_DL = 70.0
HYPER_MG_DL = 250.0


def generate_static_ehr(n_patients=N_PATIENTS):
    first_names_m = ["Rajesh", "Amit", "Vikram", "Suresh", "Manoj", "Anand", "Ramesh", "Deepak", "Sunil", "Pradeep"]
    first_names_f = ["Pooja", "Sunita", "Ananya", "Meera", "Lakshmi", "Kavita", "Sita", "Priyanka", "Geeta", "Rekha"]
    last_names = ["Sharma", "Verma", "Patel", "Reddy", "Iyer", "Nair", "Gupta", "Chatterjee", "Singh", "Deshmukh", "Kulkarni", "Rao"]
    diet_types = [
        "North Indian (High Carb/Roti)", "South Indian (Rice Predominant)",
        "Vegetarian Balanced", "Non-Vegetarian/High Protein"
    ]
    diet_weights = [0.35, 0.35, 0.20, 0.10]

    # Separate stream for history/genetics so the core profile stays reproducible.
    hist_rng = random.Random(7)

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
            med = random.choice(["Metformin 1000mg", "Metformin + DPP4i (Sitagliptin)", "Metformin + Sulfonylurea (Gliclazide)"])
            ins = "None"
        elif hba1c < 9.0:
            med = random.choice(["Metformin + Sulfonylurea (Glimepiride)", "Metformin + SGLT2i (Dapagliflozin)"])
            ins = "None"
        else:
            med = "Metformin + Dual Oral Agents"
            ins = random.choice(["Basal Insulin (Glargine 10-20U)", "Basal-Bolus (Glargine + Aspart)"])

        # Clinical risk tier (hypoglycaemia-causing drug classes)
        if ins != "None" or "Sulfonylurea" in med:
            risk = "High Risk" if age > 58 else "Moderate Risk"
        else:
            risk = "Low Risk"

        # --- Lab results beyond glycaemia ---
        ldl = int(max(60, min(210, round(hist_rng.gauss(118, 28)))))
        tg = int(max(70, min(420, round(hist_rng.gauss(165 + (bmi - 25) * 4, 45)))))

        # --- Genetic marker: TCF7L2 rs7903146 T-allele count (Hardy-Weinberg, q ~= 0.30) ---
        q = 0.30
        u = hist_rng.random()
        tcf7l2 = 0 if u < (1 - q) ** 2 else (1 if u < (1 - q) ** 2 + 2 * q * (1 - q) else 2)
        family_history = "Yes" if hist_rng.random() < (0.45 + 0.10 * tcf7l2) else "No"

        # --- Latent hypoglycaemia propensity (drives both history and the CGM simulator) ---
        on_hypo_drug = ins != "None" or "Sulfonylurea" in med
        hypo_propensity = 0.0
        if on_hypo_drug:
            hypo_propensity = 0.35 + (0.25 if ins != "None" else 0.0) + (0.15 if age > 65 else 0.0) + (0.20 if egfr < 60 else 0.0)
            hypo_propensity = min(1.0, hypo_propensity + hist_rng.uniform(-0.15, 0.15))
        prior_hypo = 0
        if on_hypo_drug:
            lam = 0.4 + 3.0 * hypo_propensity
            # Poisson draw (Knuth) for episodes in the last 12 months
            L, k, p = math.exp(-lam), 0, 1.0
            while True:
                p *= hist_rng.random()
                if p <= L:
                    break
                k += 1
            prior_hypo = k
        elif hist_rng.random() < 0.05:
            prior_hypo = 1

        # --- Past diagnoses (comorbidities consistent with the record) ---
        dx = []
        if systolic >= 140 or diastolic >= 90 or hist_rng.random() < 0.15:
            dx.append("Hypertension")
        if ldl >= 130 or tg >= 200:
            dx.append("Dyslipidaemia")
        if egfr < 60:
            dx.append("CKD stage 3")
        if duration > 8 and hist_rng.random() < 0.35:
            dx.append("Peripheral neuropathy")
        if duration > 10 and hist_rng.random() < 0.25:
            dx.append("Non-proliferative retinopathy")
        if bmi >= 30 and hist_rng.random() < 0.40:
            dx.append("NAFLD")
        if prior_hypo > 0:
            dx.append("Hypoglycaemia episode (past 12 m)")
        past_dx = "; ".join(dx) if dx else "None recorded"

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
            "clinical_risk_tier": risk,
            "ldl_cholesterol_mg_dl": ldl,
            "triglycerides_mg_dl": tg,
            "past_diagnoses": past_dx,
            "prior_hypo_episodes_12m": prior_hypo,
            "family_history_t2d": family_history,
            "tcf7l2_rs7903146_risk_alleles": tcf7l2,
            "_hypo_propensity": hypo_propensity,  # latent, not written to CSV
        })
    return patients


def _slot(rng, hour, minute, jitter_min):
    """A time-of-day in 15-min slots, jittered by +/- jitter_min minutes."""
    base = hour * 60 + minute + rng.randint(-jitter_min // 15, jitter_min // 15) * 15
    return max(0, min(24 * 60 - 15, base)) // 15


def _day_plan(rng, patient):
    """Per-day behaviour schedule as {slot_index: event} plus sleep window."""
    on_su = "Sulfonylurea" in patient["primary_medication"]
    on_ins = patient["insulin_regimen"] != "None"
    skip_boost = 0.06 if (on_su or on_ins) else 0.0
    heavy_day = rng.random() < 0.10  # festival / dining out
    carb_mult = 1.4 if heavy_day else 1.0
    high_carb = "High Carb" in patient["diet_profile"] or "Rice" in patient["diet_profile"]
    a1c_bump = 12 if patient["baseline_hba1c"] > 8.5 else 0

    meals = {}
    if rng.random() > 0.08 + skip_boost:
        meals[_slot(rng, 8, 30, 45)] = int(max(20, rng.gauss(45 + a1c_bump, 10)) * carb_mult)
    if rng.random() > 0.04 + skip_boost / 2:
        meals[_slot(rng, 13, 15, 45)] = int(max(30, rng.gauss(65 + (15 if high_carb else 0), 12)) * carb_mult)
    if rng.random() > 0.35:
        meals[_slot(rng, 17, 0, 60)] = int(max(10, rng.gauss(25, 6)))
    if rng.random() > 0.03:
        meals[_slot(rng, 20, 45, 45)] = int(max(30, rng.gauss(55 + a1c_bump, 12)) * carb_mult)

    walks = set()
    if rng.random() < 0.55:
        s = _slot(rng, 7, 15, 30)
        walks.update(range(s, s + rng.choice([2, 3, 3, 4])))
    if rng.random() < 0.50:
        s = _slot(rng, 18, 45, 45)
        walks.update(range(s, s + rng.choice([2, 3, 4])))

    poor_sleep = rng.random() < 0.15
    sleep_start = _slot(rng, 23, 0, 30) + (4 if poor_sleep else 0)   # minutes since midnight / 15
    wake = _slot(rng, 6, 0, 30) - (2 if poor_sleep else 0)
    return meals, walks, sleep_start, wake, poor_sleep


def _sleep_stage(rng, frac):
    """Sleep architecture: more deep sleep early in the night, more REM late."""
    u = rng.random()
    if frac < 0.35:
        return "Deep" if u < 0.45 else ("Light" if u < 0.90 else "REM")
    if frac < 0.65:
        return "Deep" if u < 0.20 else ("Light" if u < 0.70 else "REM")
    return "Deep" if u < 0.05 else ("Light" if u < 0.55 else ("REM" if u < 0.95 else "Awake"))


def generate_wearable_stream(ehr_list, days=DAYS, sampling_minutes=SAMPLING_MINUTES):
    steps_per_day = (24 * 60) // sampling_minutes  # 96 slots
    start_time = datetime(2026, 9, 1, 0, 0)
    all_wearables = []

    for patient in ehr_list:
        pid = patient["patient_id"]
        rng = random.Random(f"wear-{pid}")
        base_g = float(patient["fasting_glucose_mg_dl"])
        is_insulin = patient["insulin_regimen"] != "None"
        is_bolus = "Bolus" in patient["insulin_regimen"]
        is_su = "Sulfonylurea" in patient["primary_medication"]
        bmi = patient["bmi"]
        hba1c = patient["baseline_hba1c"]
        renal_slow = patient["egfr"] < 60
        tcf7l2 = patient["tcf7l2_rs7903146_risk_alleles"]
        propensity = patient["_hypo_propensity"]
        dawn = rng.random() < 0.4  # dawn phenomenon

        current_g = base_g + rng.uniform(-10, 15)
        carbs_pool = 0.0
        insulin_pool = 0.0
        insulin_decay = 0.15 if renal_slow else 0.20
        carb_sensitivity = 1.2 * (bmi / 24.0) * (hba1c / 7.0) * (1.0 + 0.08 * tcf7l2)
        prev_poor_sleep = False
        telemetry = []

        for day in range(days):
            meals, walks, sleep_start, wake, poor_sleep = _day_plan(rng, patient)
            resistance = 1.15 if prev_poor_sleep else 1.0
            basal_slot = _slot(rng, 21, 30, 30)
            su_slots = {_slot(rng, 8, 0, 30), _slot(rng, 20, 0, 30)}
            missed_basal = rng.random() < 0.07

            for s in range(steps_per_day):
                ts = start_time + timedelta(days=day, minutes=s * sampling_minutes)
                hour = ts.hour

                # Sleep window wraps midnight: asleep before `wake` or after `sleep_start`
                asleep = s < wake or s >= sleep_start
                if asleep:
                    if s < wake:
                        frac = (s + (steps_per_day - 92)) / max(1, wake + (steps_per_day - 92))
                    else:
                        frac = (s - sleep_start) / max(1, (steps_per_day - sleep_start) + wake)
                    stage = _sleep_stage(rng, max(0.0, min(1.0, frac)))
                    hr_mu = {"Deep": 58, "Light": 62, "REM": 66, "Awake": 70}[stage]
                    hrv_mu = {"Deep": 58, "Light": 50, "REM": 44, "Awake": 40}[stage]
                    heart_rate = rng.gauss(hr_mu, 3)
                    hrv = rng.gauss(hrv_mu - (8 if poor_sleep else 0), 4)
                    steps = 0
                elif s in walks:
                    stage = "Awake"
                    heart_rate = rng.gauss(118, 8)
                    hrv = rng.gauss(26, 3)
                    steps = int(max(400, rng.gauss(1100, 180)))
                else:
                    stage = "Awake"
                    heart_rate = rng.gauss(76, 5)
                    hrv = rng.gauss(42 - (6 if prev_poor_sleep else 0), 4)
                    steps = int(max(0, rng.expovariate(1.0 / 45.0)))

                meal_carbs = meals.get(s, 0)
                carbs_pool += meal_carbs

                insulin_dose = 0
                if is_insulin:
                    if s == basal_slot and not missed_basal:
                        insulin_dose = int(14 + (hba1c - 7.0) * 2)
                        insulin_pool += insulin_dose * 0.6
                    elif meal_carbs > 30 and is_bolus:
                        u = rng.random()
                        if u < 0.10:
                            insulin_dose = 0  # missed bolus
                        else:
                            dose = meal_carbs / 10.0 + rng.choice([-1, 0, 1])
                            if u > 0.92:
                                dose *= 1.5  # over-bolus
                            insulin_dose = int(max(1, dose))
                            insulin_pool += insulin_dose
                elif is_su and s in su_slots:
                    insulin_pool += 4.5 + 3.0 * propensity  # SU taken even when the meal is skipped

                absorbed = carbs_pool * 0.28
                carbs_pool -= absorbed
                carb_flux = absorbed * carb_sensitivity * resistance

                used_ins = insulin_pool * insulin_decay
                insulin_pool -= used_ins
                insulin_flux = used_ins * 5.2 * (1.0 + 0.6 * propensity)

                ex_factor = 1.6 if (is_insulin or is_su) else 1.1
                exercise_burn = (steps / 1000.0) * 6.0 * ex_factor

                nocturnal_dip = rng.uniform(1.0, 3.0) * propensity if (asleep and 1 <= hour <= 4) else 0.0
                dawn_rise = 1.6 if (dawn and 4 <= hour < 7) else 0.0
                reversion = (base_g - current_g) * 0.06
                hepatic_rescue = max(0.0, (70.0 - current_g) * 0.22)
                noise = rng.gauss(0, 1.6)

                current_g = (current_g + carb_flux - insulin_flux - exercise_burn - nocturnal_dip
                             + dawn_rise + reversion + hepatic_rescue + noise)
                current_g = max(40.0, min(400.0, current_g))

                # Adrenergic response to low glucose
                if current_g < 70:
                    heart_rate += 12 + (70 - current_g) * 0.4
                    hrv -= 10

                telemetry.append({
                    "patient_id": pid,
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "glucose_cgm": round(current_g, 1),
                    "heart_rate": int(round(heart_rate)),
                    "hrv": int(max(12, round(hrv))),
                    "steps_15m": steps,
                    "sleep_stage": stage,
                    "meal_carbs_g": meal_carbs,
                    "insulin_units": insulin_dose,
                })
            prev_poor_sleep = poor_sleep

        # Targets: glucose exactly 2 h ahead (regression) and whether ANY reading
        # in the next 2 h crosses a threshold (classification; hypo takes priority).
        n = len(telemetry)
        for idx in range(n - HORIZON_STEPS):
            window = [telemetry[j]["glucose_cgm"] for j in range(idx + 1, idx + HORIZON_STEPS + 1)]
            row = telemetry[idx]
            row["future_glucose_2h"] = telemetry[idx + HORIZON_STEPS]["glucose_cgm"]
            if min(window) < HYPO_MG_DL:
                row["adverse_event_2h"] = 1
            elif max(window) > HYPER_MG_DL:
                row["adverse_event_2h"] = 2
            else:
                row["adverse_event_2h"] = 0
            all_wearables.append(row)

    return all_wearables


def main():
    os.makedirs("data", exist_ok=True)
    print(f"Generating {N_PATIENTS} synthetic Indian T2D EHR records...")
    ehr_list = generate_static_ehr(N_PATIENTS)
    ehr_keys = [k for k in ehr_list[0].keys() if not k.startswith("_")]
    with open("data/synthetic_ehr.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ehr_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ehr_list)
    print(f"Saved data/synthetic_ehr.csv with {len(ehr_list)} records.")

    print(f"Generating {DAYS}-day 15-min wearable streams for all {N_PATIENTS} patients...")
    wearables_list = generate_wearable_stream(ehr_list)
    w_keys = list(wearables_list[0].keys())
    with open("data/synthetic_wearables.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=w_keys)
        writer.writeheader()
        writer.writerows(wearables_list)
    print(f"Saved data/synthetic_wearables.csv with {len(wearables_list)} rows.")

    counts = {0: 0, 1: 0, 2: 0}
    for r in wearables_list:
        counts[r["adverse_event_2h"]] += 1
    total = len(wearables_list)
    print("\nLabel distribution (event within the next 2 hours):")
    print(f"0 - No event (stays 70-250 mg/dL): {counts[0]} ({counts[0]/total*100:.1f}%)")
    print(f"1 - Hypoglycaemia (<70 mg/dL):     {counts[1]} ({counts[1]/total*100:.1f}%)")
    print(f"2 - Hyperglycaemia (>250 mg/dL):   {counts[2]} ({counts[2]/total*100:.1f}%)")
    print("Data generation complete.")


if __name__ == "__main__":
    main()
