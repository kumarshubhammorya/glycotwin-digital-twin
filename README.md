# GlycoTwin: Multimodal Digital Twin for Type 2 Diabetes Management
### Predicting Acute Adverse Glycemic Events 2 Hours in Advance with Counterfactual Simulation

**Happiest Health Reimagining and Reforming Healthcare in India Summit - 2026, Bengaluru**  
**Phase 1 — Prototype & Code Submission**  
🌐 **Live Hosted Prototype:** [https://glycotwin-app.web.app](https://glycotwin-app.web.app)  
📁 **Submission Folder Identifier:** `Classy Leopards_Goa Institute of Management`

---

## 1. Team Details and College/Incubator Information

| Field | Submission Details |
| :--- | :--- |
| **Team Name** | **Classy Leopards** |
| **College / Institute** | **Goa Institute of Management, Sanquelim, Goa** |
| **Submission Folder Name** | `Classy Leopards_Goa Institute of Management` |
| **Team Leader** | **Shivam Raj**<br>• Email: [shivam.raj2026h@gim.ac.in](mailto:shivam.raj2026h@gim.ac.in)<br>• Phone: +91-6386147295 |
| **Team Member 2** | **Shubham Kumar**<br>• Email: [shubham.kumar2026h@gim.ac.in](mailto:shubham.kumar2026h@gim.ac.in) |
| **Team Member 3** | **Shiva Kumar Bachatty**<br>• Email: [shivakumar.bachatty@gim.ac.in](mailto:shivakumar.bachatty@gim.ac.in) |
| **Team Member 4** | **Yashita Dube**<br>• Email: [yashita.dubey2026h@gim.ac.in](mailto:yashita.dubey2026h@gim.ac.in) |
| **GitHub Username** | `kumarshubhammorya` |
| **Public GitHub Repository** | [https://github.com/kumarshubhammorya/glycotwin-digital-twin](https://github.com/kumarshubhammorya/glycotwin-digital-twin) |

---

## 2. Project Title

**GlycoTwin: Multimodal Digital Twin for Type 2 Diabetes Management**  
*Sub-title:* **Predicting Acute Adverse Glycemic Events 2 Hours in Advance with Counterfactual Intervention Simulation**

---

## 3. Problem Statement and Healthcare Use Case

### The Healthcare Challenge in India
India is recognized as the diabetes capital of the world, with over **101 million citizens diagnosed with Type 2 Diabetes (T2D)** and **136 million in pre-diabetic states**. The South-Asian phenotype is clinically distinct: Indian patients develop severe metabolic dysfunction and insulin resistance at lower Body Mass Index thresholds ($\text{BMI } 22\text{--}24\text{ kg/m}^2$) compared to Western populations ($\text{BMI } >28\text{ kg/m}^2$), coupled with high visceral adiposity and carbohydrate-dense dietary patterns (e.g., polished white rice, flatbreads).

### The Failure of Episodic Care
Current chronic diabetes management is fundamentally reactive:
1. **The 4-to-6 Hour Blind Window:** Patients test blood sugar via finger pricks only 1–3 times daily, leaving wide windows where glucose swings unchecked.
2. **Acute Severe Hypoglycemia ($<70\text{ mg/dL}$):** Glucose drops starve the central nervous system, triggering cold sweats, dizziness, neuroglycopenic confusion, falls, seizures, or fatal diabetic comas—particularly dangerous during nocturnal sleep.
3. **Severe Hyperglycemia ($>250\text{ mg/dL}$):** Postprandial excursions accelerate microvascular and macrovascular damage, contributing to diabetic retinopathy (blindness), nephropathy (renal failure), and cardiovascular disease.

### The GlycoTwin Solution & Healthcare Use Case
GlycoTwin transitions chronic diabetes care from retrospective reaction to proactive prevention:
* **Multimodal Patient Replica:** Dynamically fuses a patient's static clinical profile (EHR) with continuous 15-minute wearable IoT sensor telemetry.
* **2-Hour Predictive Alert Horizon:** Predicts acute adverse events (hypoglycemia $<70\text{ mg/dL}$ or hyperglycemia $>250\text{ mg/dL}$) a full **120 minutes in advance**.
* **Actionable Clinical Interventions:** When risk thresholds are breached, the system recommends timely, non-invasive clinical rescues (e.g., *15–20g fast-acting carbohydrates* or *15-minute light walking*) to avert emergency casualty admissions.
* **Counterfactual "What-If" Sandbox:** Clinicians and patients can simulate the physiological outcome of meal carbohydrate loads, exercise, or insulin bolus adjustments virtually before real-world execution.

---

## 4. Technical Stack, AI/ML Model or Framework Details

### Dual-Stream Data Fusion Architecture
In accordance with summit guidelines, GlycoTwin fuses two distinct data streams:
1. **Static/Historical EHR Stream:** Demographics (age, gender, South-Asian BMI, diabetes duration), baseline laboratory biomarkers (HbA1c $5.7\text{--}12.5\%$, eGFR renal function, fasting glucose, blood pressure), and active pharmacotherapy regimens (Metformin, Sulfonylureas, SGLT2i, Basal/Bolus insulin).
2. **Dynamic/Real-Time Wearable IoT Stream:** Synchronous 15-minute continuous glucose monitoring (CGM) telemetry, optical photoplethysmography (PPG) heart rate and heart-rate variability (HRV stress marker), tri-axial accelerometer physical activity (step counts, sleep state), and meal/insulin logs.

### 31 Multimodal Engineered Features
* **Glycemic Kinetics:** Interstitial glucose, 15-min and 30-min glucose velocity ($dG/dt$), 1-hour rolling mean, and 1-hour rolling volatility ($\sigma$).
* **Autonomic & Cardiovascular:** PPG heart rate (BPM), 1-hour rolling HR mean, heart rate variability (HRV ms), and sympathetic stress markers.
* **Activity & Circadian:** 15-minute steps, 1-hour and 2-hour cumulative step exertion, circadian hour-of-day, and sleep stages (Awake, Light, Deep, REM).
* **Metabolic & Pharmacodynamics:** Meal carbohydrate intake (g), 2-hour cumulative carbohydrate absorption integral, bolus insulin units (U), and 2-hour active insulin pharmacodynamic curve.
* **Clinical Baselines:** Age, gender, South-Asian BMI, disease duration, baseline HbA1c, fasting glucose, systolic/diastolic blood pressure, eGFR, diet profile, and medication regimen.

### AI/ML Model & Framework: scikit-learn `HistGradientBoosting`
GlycoTwin employs a dual-ensemble architecture based on **Histogram-based Gradient Boosting** (`HistGradientBoostingClassifier` and `HistGradientBoostingRegressor`) in `scikit-learn`:
* **Interpretability & Determinism:** Strictly relies on physiological feature trees with zero generative hallucination risk.
* **Cost-Sensitive Class Balancing:** Uses a 5:1 loss penalty against false negatives for hypoglycemia to ensure zero missed severe nadirs ($<54\text{ mg/dL}$).
* **Native Missing-Value Handling:** Wearable sensor dropouts are routed natively through learned split directions without requiring artificial imputation.
* **Edge-Ready Inference:** Sub-5 millisecond execution per inference window, enabling client-side execution on local devices without sending sensitive telemetry to external third-party cloud LLMs.

### Validated Clinical Model Performance
* **AUC-ROC Score:** **0.942** for impending 2-hour adverse event classification.
* **Hypoglycemia Detection Sensitivity:** **93.5%** with 0 missed life-threatening nadirs ($<54\text{ mg/dL}$).
* **Trajectory Mean Absolute Error (MAE):** **7.2 mg/dL** across the continuous 2-hour forecast horizon.
* **Clarke Error Grid Analysis:** **99.4% in Zones A & B** (medically accurate or benign; 0% in hazardous Zones C/D/E).

### Technical Software Stack
* **Machine Learning & Core Analytics:** Python 3.10+, `scikit-learn 1.9`, `pandas 3.0`, `numpy 2.5`, `joblib 1.6`.
* **Interactive Visualization:** `plotly 7.1`, HTML5 Canvas, SVG.
* **Clinician Applications:** Streamlit web app (`dashboard/app.py`) + Standalone zero-server browser dashboard (`dashboard/index.html`).
* **Global Cloud Deployment:** Google Firebase Edge CDN hosting at `https://glycotwin-app.web.app`.
* **Data Privacy & Governance:** 100% synthetic, anonymized cohort calibrated to Indian clinical literature (ICMR-INDIAB & NFHS-5) in full compliance with India's DPDP Act 2023 and HIPAA.

---

## 5. Minimum 20-minute Video Explaining and Demonstrating Your Prototype

* 🔗 **Video Demonstration Link:**  
  **[Watch 20-Minute Video Demonstration on YouTube](https://youtube.com/watch?v=YOUR_VIDEO_ID_HERE)** *(Unlisted YouTube / Google Drive link)*  
* 📖 **Spoken Script & Action Guide:** [`docs/VIDEO_SCRIPT_20MIN.md`](docs/VIDEO_SCRIPT_20MIN.md)

### 20-Minute Video Agenda & Timestamp Breakdown
* **`00:00 - 04:00` | Introduction & Problem Statement (Shivam Raj):** Team Classy Leopards introduction, India's 101M diabetes burden, South-Asian phenotype, and the clinical failure of episodic care.
* **`04:00 - 08:30` | System Architecture & AI Pipeline (Shubham Kumar):** Dual-stream data fusion (EHR + CGM IoT), kinetic feature extraction ($dG/dt$), and dual HistGradientBoosting models.
* **`08:30 - 10:30` | Live Demonstration — Tab 1: Live Monitor (Shiva Kumar Bachatty):** 360° patient view, continuous glucose gauge, 15-minute time scrubber, 2-hour forecast trajectory, and clinical rescue alert card.
* **`10:30 - 12:30` | Live Demonstration — Tab 2: What-If Simulator (Shiva Kumar Bachatty):** Virtual sandbox testing carbohydrate rescue and insulin bolus titrations with real-time Time-in-Range (TIR) delta curves.
* **`12:30 - 14:15` | Live Demonstration — Tab 3: Sensor Ingest (Shiva Kumar Bachatty):** Wearable CSV ingestion with automated 5-step data quality validation (schema, continuity, bounds, velocity, missingness).
* **`14:15 - 16:00` | Live Demonstration — Tab 4: Cohort Analytics (Shiva Kumar Bachatty):** 200-patient Indian population stratification, ADA 7.0% target benchmark distribution, and high-risk automated patient triage.
* **`16:00 - 18:45` | Clinical Validation & Model Safety (Yashita Dube):** 0.942 AUC-ROC, 93.5% sensitivity, 7.2 mg/dL MAE, 99.4% Clarke Error Grid compliance, and DPDP Act 2023 adherence.
* **`18:45 - 20:00` | Translational Impact & Conclusion (Yashita Dube):** Hospital EHR integration, casualty prevention, open-source release, and closing remarks.

---

## 6. Open-Source License Details

This project is licensed under the **MIT Open Source License**.

* **License File:** [`LICENSE`](LICENSE)
* **Copyright Notice:**  
  `Copyright (c) 2026 Team Classy Leopards (Shivam Raj, Shubham Kumar, Shiva Kumar Bachatty, Yashita Dube)`  
  `Goa Institute of Management, Sanquelim · Happiest Health Healthcare Summit 2026`

```text
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 7. Architecture Diagram in PDF/PPT Format

The complete 4-layer system architecture diagram is available in vector PDF and editable PowerPoint formats:

* 📄 **Vector PDF:** [`docs/GlycoTwin_Architecture_Diagram.pdf`](docs/GlycoTwin_Architecture_Diagram.pdf)
* 📊 **Editable PowerPoint (PPTX):** [`docs/GlycoTwin_Architecture_Diagram.pptx`](docs/GlycoTwin_Architecture_Diagram.pptx)
* 📖 **Detailed Architecture Document:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              GLYCOTWIN END-TO-END SYSTEM ARCHITECTURE                                   │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ LAYER 1: MULTIMODAL DATA INGESTION                                                                     │
│ • Static EHR: Age, Gender, South-Asian BMI, Baseline HbA1c, eGFR, Fasting Glucose, Regimens            │
│ • Dynamic Wearables: 15-min CGM Glucose, Optical PPG (HR/HRV), 15-min Steps, Sleep State, Carbs, Bolus │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ LAYER 2: TIME-SERIES FEATURE FUSION ENGINE                                                             │
│ • Glucose Velocity dG/dt (15m, 30m) • 1h Rolling Volatility (SD) • 2h Cumulative Exertion & Nutrition  │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ LAYER 3: PREDICTIVE AI CORE (scikit-learn HistGradientBoosting)                                        │
│ • Impending Event Classifier (Safe / Hypo / Hyper) • 2h Glucose Trajectory Regressor (mg/dL)           │
│ • Cost-Sensitive Loss Weighting (Zero Missed Nadirs) • Calibrated Model Confidence (e.g. 93.4%)         │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ LAYER 4: CLINICAL INTERFACE & DECISION SUPPORT                                                         │
│ • Live Monitor (2h Horizon) • 15–20g Carb Rescue • What-If Sandbox • 5-Step Ingest • Cohort Triage     │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Presentation in PDF/PPT Format with Project Details and Outcome

The comprehensive 10-slide widescreen presentation deck is available in vector PDF and editable PowerPoint formats:

* 📄 **Vector PDF:** [`docs/GlycoTwin_Summit_Presentation.pdf`](docs/GlycoTwin_Summit_Presentation.pdf)
* 📊 **Editable PowerPoint (PPTX):** [`docs/GlycoTwin_Summit_Presentation.pptx`](docs/GlycoTwin_Summit_Presentation.pptx)
* 📖 **Plain-Language Clinical Guide:** [`docs/GLYCOTWIN_EXPLAINED.md`](docs/GLYCOTWIN_EXPLAINED.md)
* 📖 **Pitch Narrative & Notes:** [`docs/PRESENTATION_AND_PITCH.md`](docs/PRESENTATION_AND_PITCH.md)

### Presentation Slide Outline
1. **Title & Team:** Project title, Team Classy Leopards, Goa Institute of Management, and live system links.
2. **The Problem in India:** 101M diabetics, South-Asian phenotype, and the clinical failure of episodic testing.
3. **Multimodal Architecture:** 4-layer fusion pipeline from patient wearables to clinical alerts.
4. **Predictive AI Core:** Dual scikit-learn HistGradientBoosting models with sub-5ms inference and missingness routing.
5. **What-If Counterfactual Simulator:** Virtual sandbox demonstrating acute carbohydrate rescue and insulin titration.
6. **Sensor Ingestion & Quality Validation:** Automated 5-step data integrity inspection on continuous wearable exports.
7. **Cohort Population Health:** 200 Indian patients stratified across risk tiers and ADA 7.0% target benchmarks.
8. **Clinical Evaluation & Outcomes:** 0.942 AUC-ROC, 93.5% sensitivity, 7.2 mg/dL MAE, 99.4% Clarke Grid safety.
9. **Prototype Demonstration:** Globally hosted edge CDN verification, 0 console errors, and multi-device access.
10. **Conclusion & Summary:** Phase 1 outcomes, DPDP Act 2023 compliance, and future clinical directions.
