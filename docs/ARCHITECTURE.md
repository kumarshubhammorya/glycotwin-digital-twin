# GlycoTwin: System Architecture & Technical Specifications

> **Happiest Health Reimagining & Reforming Healthcare in India Summit 2026 (Bengaluru)**  
> **Challenge:** Proof-of-Concept Digital Twin for Chronic Disease Management (Type 2 Diabetes)

---

## 1. Executive Summary

GlycoTwin is a localized, multimodal Digital Twin system designed for proactive management of Type 2 Diabetes in India. It solves the critical limitation of reactive healthcare by fusing two disjoint data streams:
1. **Static / Historical Electronic Health Records (EHR)**: Clinical baselines, pharmacotherapy, organ biomarkers, and lifestyle context.
2. **Dynamic / Real-Time Wearable & IoT Telemetry**: Continuous Glucose Monitoring (CGM), Heart Rate, Heart Rate Variability (HRV), accelerometer steps, and meal inputs.

By analyzing temporal physiological velocity and patient baseline risk, GlycoTwin forecasts **acute adverse glycemic events (Hypoglycemia $< 70$ mg/dL and Hyperglycemia $> 250$ mg/dL) 2 hours in advance**, enabling timely clinical and dietary interventions before dangerous clinical emergencies occur.

---

## 2. High-Level Architecture Diagram

```
+-----------------------------------------------------------------------------------+
|                              DATA INGESTION LAYER                                 |
+----------------------------------------+------------------------------------------+
|      STATIC CLINICAL STREAM (EHR)      |       DYNAMIC SENSOR STREAM (IoT)        |
|  - Patient Demographics (Age, BMI)     |  - CGM Glucose (every 15 mins)           |
|  - Baseline Labs (HbA1c, Fasting Glu)  |  - Photoplethysmography (HR & HRV)       |
|  - Renal Function (eGFR, Creatinine)   |  - 3-Axis Accelerometer (Steps/Activity) |
|  - Medications (Metformin, SU, Insulin)|  - Patient Meal & Bolus Event Logging    |
|  - Dietary Archetype (Indian regional) |  - Sleep Staging (Deep, Light, REM)      |
+----------------------------------------+------------------------------------------+
                                     |
                                     v
+-----------------------------------------------------------------------------------+
|                     MULTIMODAL FUSION & TEMPORAL ENGINE                           |
+-----------------------------------------------------------------------------------+
|  * Temporal Windowing (15m, 60m, 120m rolling statistics)                         |
|  * Glucose Velocity Derivative (dG/dt) & Volatility Indices (Standard Deviation)  |
|  * Metabolic Exertion Score (Steps x Insulin Sensitivity Factor)                  |
|  * Autonomic Stress Biomarker (HRV suppression during nocturnal hypoglycemia)     |
|  * Categorical Encoding for Pharmacotherapy & Clinical Risk Tiers                 |
+-----------------------------------------------------------------------------------+
                                     |
                                     v
+-----------------------------------------------------------------------------------+
|                         GLYCOTWIN PREDICTIVE AI SUITE                             |
+-----------------------------------------------------------------------------------+
|  1. ADVERSE EVENT CLASSIFIER (Gradient Boosted Ensemble):                         |
|     - Class 0: Target In-Range (70 - 180 mg/dL)                                   |
|     - Class 1: Severe Hypoglycemia Alert (< 70 mg/dL) [Priority 1 Emergency]     |
|     - Class 2: Severe Hyperglycemia Alert (> 250 mg/dL) [Priority 2 Spike]       |
|                                                                                   |
|  2. CONTINUOUS TRAJECTORY REGRESSOR:                                             |
|     - Forecasts exact quantitative glucose trajectory (mg/dL) at t + 120 mins     |
|                                                                                   |
|  3. COUNTERFACTUAL "WHAT-IF" SIMULATION ENGINE:                                   |
|     - Simulates patient physiological response to variable carbs, exercise, and   |
|       insulin titration prior to real-world execution                             |
+-----------------------------------------------------------------------------------+
                                     |
                                     v
+-----------------------------------------------------------------------------------+
|                         CLINICAL DECISION SUPPORT SYSTEM                          |
+-----------------------------------------------------------------------------------+
|  * Interactive Web UI (Streamlit & Plotly Telemetry)                              |
|  * 2-Hour Predictive Alert Horizon with Automated Actionable Rescue Protocols     |
|  * Drag-and-Drop Ingestion for Real Wearable Exports (Dexcom, LibreView, Apple)   |
|  * Indian Cohort Risk Stratification (N=200 Patient Cohort)                      |
+-----------------------------------------------------------------------------------+
```

---

## 3. Data Schema & Feature Engineering

### Static Features ($X_{static}$)
* `age`: Patient chronological age (28–79 years).
* `bmi`: Body Mass Index ($\text{kg/m}^2$, calibrated for South Asian risk thresholds).
* `diabetes_duration_years`: Duration of diagnosed T2D.
* `baseline_hba1c`: Glycated hemoglobin percentage (5.7%–12.5%).
* `fasting_glucose_mg_dl`: Baseline morning glucose.
* `systolic_bp`, `diastolic_bp`: Hemodynamic baseline.
* `egfr`: Estimated Glomerular Filtration Rate ($\text{mL/min}/1.73\text{m}^2$).
* `diet_profile`: Regional dietary intake (North Indian Carb-rich, South Indian Rice-predominant, Balanced).
* `primary_medication`: Oral hypoglycemic agents (Metformin, Sulfonylurea, SGLT2i, DPP4i).
* `insulin_regimen`: Basal or Basal-Bolus insulin therapies.

### Dynamic Features ($X_{dynamic}$)
* `glucose_cgm`: Instantaneous sensor glucose ($\text{mg/dL}$).
* `glucose_velocity_15m` ($dG/dt$): Instantaneous slope of glucose change.
* `glucose_velocity_30m`: Extended slope indicator.
* `glucose_mean_1h`, `glucose_std_1h`: 1-hour moving mean and glucose volatility.
* `heart_rate`, `hr_mean_1h`: Sympathetic/parasympathetic tone.
* `hrv`, `hrv_mean_1h`: Autonomic suppression metric (critical during nocturnal hypoglycemia).
* `steps_15m`, `steps_sum_1h`, `steps_sum_2h`: Skeletal muscle glucose uptake drivers.
* `meal_carbs_g`, `carbs_sum_2h`: Exogenous carbohydrate glycemic load.
* `insulin_units`, `insulin_sum_2h`: Exogenous insulin pharmacokinetics.

---

## 4. Real-World Ingestion Pathways

To bridge prototype demonstration with clinical production, GlycoTwin is architected to support three enterprise ingestion pipelines:

1. **Healthcare Provider Integration (EHR)**:
   * **HL7 / FHIR (Fast Healthcare Interoperability Resources)**: Ingests patient observation resources via standard RESTful JSON interfaces (`Observation/glucose`, `MedicationStatement`).
2. **Consumer Wearable Cloud APIs**:
   * **Abbott LibreLinkUp API & Dexcom Web API**: Automated webhook receivers polling 5-to-15 minute glucose telemetry packets.
   * **Apple HealthKit & Google Health Connect**: Native mobile integration aggregating steps, heart rate, and sleep logs.
3. **Local Clinical Portal Ingestion**:
   * Direct CSV parsing for Abbott LibreView and Dexcom Clarity export files for offline or disconnected clinical settings.

---

## 5. Compliance & Data Privacy (DPDP Act & HIPAA)
* All datasets utilized in this proof-of-concept are strictly synthetic, anonymized, and statistically modeled to respect India's Digital Personal Data Protection (DPDP) Act 2023 and global HIPAA standards.
* The system architecture employs zero persistent storage of personally identifiable medical credentials.
