# 🎙️ GlycoTwin: 20-Minute Video Presentation & Demonstration Script
**Competition:** Happiest Health Healthcare Summit 2026 · Bengaluru, India  
**Project Title:** GlycoTwin — Multimodal Digital Twin for Type 2 Diabetes Management  
**Team Name:** Classy Leopards  
**Institution:** Goa Institute of Management, Sanquelim  
**Live Production URL:** [https://glycotwin-app.web.app](https://glycotwin-app.web.app)  
**GitHub Repository:** [https://github.com/kumarshubhammorya/glycotwin-digital-twin](https://github.com/kumarshubhammorya/glycotwin-digital-twin)  
**Target Video Duration:** 20 Minutes (0:00 – 20:00)

---

## 👥 Speaker Roles & Division of Time

| Time Window | Speaker / Role | Section | Visual on Screen |
|---|---|---|---|
| **0:00 – 4:00** | **Shivam Raj** (Team Leader) | Introduction, Problem Statement & Indian Context | Slide 1 (Title) & Slide 2 (Epidemic Challenge) |
| **4:00 – 8:30** | **Shubham Kumar** | Multimodal Architecture & AI/ML Engine | Slides 3, 4, 5 & Architecture Diagram |
| **8:30 – 16:00** | **Shiva Kumar Bachatty** | Complete Live System Demonstration (All 4 Tabs) | Live Web App: `https://glycotwin-app.web.app` |
| **16:00 – 20:00** | **Yashita Dube** | Clinical Validation, DPDP Compliance & Closing | Slides 7, 8, 9 & Slide 10 (Summary) |

*(Note: If recording individually or as a single presenter, one speaker can smoothly present the entire 20 minutes following the exact cues below.)*

---

## ⏱️ MINUTE-BY-MINUTE SCRIPT & SCREEN ACTION GUIDE

---

### PART 1: INTRODUCTION & THE INDIAN CLINICAL CONTEXT (0:00 – 4:00)
**Speaker:** Shivam Raj (Team Leader)  
**Screen Display:** `docs/GlycoTwin_Summit_Presentation.pdf` (Slide 1 & Slide 2)

#### 0:00 – 1:30 | Slide 1: Title & Team Introduction
- **[Screen Action]:** Fullscreen display of Slide 1 (*GlycoTwin: Multimodal Digital Twin for Type 2 Diabetes Management*).
- **[Spoken Words]:**
  > "Respected judges, clinical leaders, and the organizing committee of the Happiest Health Healthcare Summit 2026 — good day!
  >
  > We are **Team Classy Leopards** from the **Goa Institute of Management, Sanquelim**. My name is **Shivam Raj**, and along with my team members **Shubham Kumar**, **Shiva Kumar Bachatty**, and **Yashita Dube**, we are deeply honored to present **GlycoTwin** — our clinical-grade, multimodal digital twin system designed to revolutionize Type 2 Diabetes care in India.
  >
  > Type 2 Diabetes is no longer an isolated chronic ailment; it is an urgent public health crisis. Today, we will walk you through our end-to-end prototype: from the clinical physiology of the South-Asian diabetic phenotype, to our dual-ensemble machine learning architecture, followed by a live interactive demonstration on our globally deployed production platform, and concluding with our clinical validation and data compliance framework."

#### 1:30 – 4:00 | Slide 2: The Indian Challenge & Episodic Care Failure
- **[Screen Action]:** Advance to Slide 2 (*The Problem: India's 101 Million Diabetic Crisis & The Failure of Episodic Care*). Point cursor to the 3 stat cards: `101M+`, `< 70 mg/dL`, and `4–6 Hours`.
- **[Spoken Words]:**
  > "Let us examine why diabetes care in India is failing its patients. India is the recognized diabetes capital of the world, with over **101 million citizens diagnosed with Type 2 Diabetes** and another **136 million in pre-diabetic states**. 
  >
  > Crucially, the South-Asian phenotype presents a unique clinical dilemma: Indian patients develop significant metabolic dysfunction and insulin resistance at much lower BMI thresholds — typically between 22 and 24 kg/m² — coupled with high visceral adiposity and a diet heavy in refined carbohydrates.
  >
  > Yet, how is diabetes managed today? Through **episodic care**. A patient visits an endocrinologist once every three months, checks an HbA1c test, and pricks their finger once or twice a week. That leaves a massive blind window. When a patient takes bolus insulin or sulfonylureas, their glucose can plunge below 70 mg/dL in under two hours. By the time they experience cold sweats, tremors, and confusion, it is too late — severe neuroglycopenia has already set in, resulting in falls, emergency room admissions, or even fatal diabetic comas.
  >
  > Conversely, undetected postprandial spikes exceeding 250 mg/dL accelerate microvascular damage, leading to diabetic retinopathy and nephropathy.
  >
  > What Indian patients and healthcare providers need is not a retrospective diary. They need an **advance predictive radar**. They need a Digital Twin that constantly assimilates their continuous glucose monitor, smartwatch vitals, and EHR profile to predict acute events **2 hours before they occur**, granting a golden window for preventative action. 
  >
  > I will now hand over to **Shubham Kumar** to explain how we engineered this architecture."

---

### PART 2: SYSTEM ARCHITECTURE & AI/ML ENGINE (4:00 – 8:30)
**Speaker:** Shubham Kumar  
**Screen Display:** Slides 3, 4, 5, and `docs/GlycoTwin_Architecture_Diagram.pdf`

#### 4:00 – 5:30 | Slide 3 & 4: Multimodal Data Ingestion & Feature Kinetics
- **[Screen Action]:** Advance to Slide 3 (*Multimodal Architecture*) and Slide 4 (*Feature Kinetics*).
- **[Spoken Words]:**
  > "Thank you, Shivam. A biological digital twin is only as robust as the signals it captures. GlycoTwin fuses two synchronous data streams:
  >
  > First, the **Static Clinical EHR Baseline**: capturing patient age, gender, South-Asian BMI, baseline HbA1c, renal clearance via eGFR, fasting glucose, and active pharmacotherapy — such as Metformin, Sulfonylureas, SGLT2 inhibitors, and Basal/Bolus insulin regimens.
  >
  > Second, the **Dynamic Sensor IoT Stream**: sampled at continuous 15-minute intervals. This captures interstitial glucose from Continuous Glucose Monitors like Abbott Freestyle Libre, optical PPG heart rate and heart-rate variability (HRV) for physiological stress, and tri-axial accelerometer step counts for metabolic exertion.
  >
  > In our **Layer 2 Time-Series Feature Engineering**, we compute rolling first and second derivatives of glucose velocity ($dG/dt$ over 15 and 30 minutes), rolling glycemic volatility, and rolling cumulative physical exertion. Crucially, we model digestive and pharmacodynamic absorption windows using 2-hour exponentially decaying integrals for carbohydrate ingestion and active bolus insulin units."

#### 5:30 – 7:00 | Slide 5: The Dual AI Predictive Engine (HistGradientBoosting)
- **[Screen Action]:** Advance to Slide 5 (*The Dual AI Engine*). Highlight the model details: `scikit-learn HistGradientBoosting`.
- **[Spoken Words]:**
  > "When designing the AI core for clinical deployment, we prioritized two non-negotiable principles: **interpretability** and **computational reliability**. 
  >
  > Unlike opaque deep neural networks that require heavy GPU infrastructure and are prone to hallucinations, GlycoTwin utilizes a dual-ensemble of **scikit-learn HistGradientBoosting models**:
  >
  > 1. **The Adverse Event Classifier:** A cost-sensitive tree ensemble trained with class-weighted penalties ($5:1$ against false-negative hypoglycemia). This predicts whether the patient will cross critical thresholds within the next 120 minutes: severe hypoglycemia ($<70$ mg/dL) or acute hyperglycemia ($>250$ mg/dL).
  > 2. **The Trajectory Regressor:** A gradient-boosted regressor that projects the patient's continuous glycemic trajectory point-by-point across the 2-hour forward horizon.
  >
  > Why HistGradientBoosting? First, it natively handles missing wearable sensor data through optimal branching paths without introducing imputation artifacts. Second, it executes inference in under **5 milliseconds**, enabling zero-latency predictions right inside the client browser or mobile edge without transmitting sensitive biometric data to third-party cloud servers."

#### 7:00 – 8:30 | Architecture Diagram: End-to-End Clinical Dataflow
- **[Screen Action]:** Switch to `docs/GlycoTwin_Architecture_Diagram.pdf` showing the 4-layer architecture. Trace the flow from Layer 1 to Layer 4 with cursor.
- **[Spoken Words]:**
  > "Looking at our system architecture diagram:
  > - **Layer 1** continuously ingests the multimodal EHR and IoT streams.
  > - **Layer 2** aligns timestamps and derives physiological kinetics.
  > - **Layer 3** evaluates the feature vector through the calibrated AI core.
  > - **Layer 4** surfaces the Clinical Interface: firing automated alerts when risk exceeds 70%, proposing concrete clinical protocols (such as a 15-gram fast-acting carbohydrate rescue), empowering the clinician with a What-If counterfactual simulator, and validating external sensor files through a 5-step data integrity engine.
  >
  > Now, let us see GlycoTwin in action. I hand over to **Shiva Kumar Bachatty** for the live demonstration."

---

### PART 3: LIVE SYSTEM DEMONSTRATION (8:30 – 16:00)
**Speaker:** Shiva Kumar Bachatty  
**Screen Display:** Web Browser opened to `https://glycotwin-app.web.app` (fullscreen)

#### 8:30 – 10:30 | Live Demonstration — Tab 1: Live Patient 360° Monitor
- **[Screen Action]:** Open browser at `https://glycotwin-app.web.app`. Show the header: *GlycoTwin · Clinical Digital Twin for Type 2 Diabetes*. Ensure Tab 1 (*Live Monitor*) is selected.
- **[Spoken Words]:**
  > "Thank you, Shubham! We are now looking at the live, production-deployed GlycoTwin system, running globally at `https://glycotwin-app.web.app`. Anyone can access this URL right now on any phone, tablet, or PC without installing any software.
  >
  > In the left sidebar, we have our patient selector. We are looking at Patient **GT-1042**, a 54-year-old male with a South-Asian BMI of 24.8, baseline HbA1c of 7.6%, on Metformin and Glimepiride. Notice how the clinical chip tags show their current medication, kidney function (eGFR 78), and fasting baseline.
  >
  > At the top of Tab 1, we see the primary **Glycemic Gauge**. Our current reading is **82 mg/dL**, trending downwards at $-1.2$ mg/dL per minute.
  >
  > Look at the **Predictive Risk Pill** on the right: it displays an urgent Amber Alert — **'Imminent Hypoglycemia Risk: 78.4% (2-Hour Horizon)'**. 
  >
  > Below the gauge is our interactive **Time-Horizon Scrubber**. When I drag the slider back 45 minutes, we observe the historical 8-hour glucose trajectory alongside step exertion and heart rate. When I scrub forward to the present, the dashed projection line shows our HistGradientBoosting forecast: if unaddressed, the patient's glucose will nadir at **58 mg/dL** in approximately 75 minutes — entering severe neuroglycopenic hypoglycemia.
  >
  > Notice the dynamic **Clinical Recommendation Card**: GlycoTwin doesn't just sound an alarm; it provides an evidence-based clinical action plan: *'Administer 15g fast-acting carbohydrate (e.g., 100 mL fruit juice or 3 glucose tablets). Re-check sensor in 15 minutes.'* 
  >
  > There is an EHR audit log button labelled `(Demo)` that simulates direct FHIR transmission to the hospital's electronic records."

#### 10:30 – 12:30 | Live Demonstration — Tab 2: What-If Counterfactual Simulator
- **[Screen Action]:** Click on Tab 2 (*What-If Simulator*). Point to the interactive controls for Carbohydrate Intake and Bolus Insulin Titration.
- **[Spoken Words]:**
  > "Now, let us navigate to Tab 2: the **What-If Counterfactual Simulator**. This is where GlycoTwin acts as an authentic digital sandbox for precision medicine.
  >
  > Rather than trial-and-error adjustments on the living patient, a clinician or diabetic educator can simulate interventions virtually before administering them.
  >
  > Look at the dual sliders:
  > 1. **Immediate Carbohydrate Rescue (grams)**: currently set to 0.
  > 2. **Bolus Insulin Adjustment (units)**: currently set to 0.
  >
  > Notice the baseline prediction: the solid red line dives below the green target band (70–180 mg/dL), reaching a nadir of 58 mg/dL with a projected Time-in-Range of only 42%.
  >
  > Watch what happens as I drag the **Carbohydrate Rescue slider from 0 grams to 20 grams**...
  > 
  > *(Drag slider to 20g on screen)*
  > 
  > Instantly, in real time, the digital twin recalculates the trajectory! The dashed blue counterfactual line shifts upward. The predicted nadir moves from 58 mg/dL up to **94 mg/dL**, safely avoiding hypoglycemia altogether.
  >
  > The **Projected Time-in-Range (TIR) increases from 42% to 88%**, representing a **+46% clinical improvement**! 
  >
  > Furthermore, if the patient were hyperglycemic at 260 mg/dL, an endocrinologist can simulate a 2-unit or 4-unit rapid-acting insulin bolus to observe the exact rate of descent, preventing the deadly phenomenon of 'insulin stacking' where consecutive doses cause post-treatment hypoglycemic crashes."

#### 12:30 – 14:15 | Live Demonstration — Tab 3: Sensor Ingest & 5-Step Clinical Data Quality Engine
- **[Screen Action]:** Click on Tab 3 (*Data Ingest*). Show the file uploader and the 'Load Pre-packaged Clinical Sample' button. Click the button to trigger validation.
- **[Spoken Words]:**
  > "Next, let us explore Tab 3: **Sensor Ingestion & Data Integrity**.
  >
  > In real-world telemedicine, patient-uploaded CSV files from wearables are frequently corrupted, contain sensor dropouts, or suffer from transmission lag. Deploying AI models on noisy sensor data without validation can lead to catastrophic medical errors.
  >
  > To solve this, GlycoTwin features an automated **5-Step Sensor Quality Engine**:
  > 1. **Schema & Header Verification:** Validates that required clinical timestamps and glucose fields match standards like Abbott LibreView or Dexcom Studio.
  > 2. **Temporal Continuity Check:** Detects missing packets or sampling gaps exceeding the 15-minute expected delta.
  > 3. **Physiological Range Bounds:** Flags physiologically impossible values ($<20$ mg/dL or $>600$ mg/dL) caused by sensor electrode detachment.
  > 4. **Rate-of-Change Feasibility ($dG/dt$):** Evaluates if glucose velocity exceeds 5 mg/dL per minute — an impossible rate under human glucose kinetics, indicating sensor compression artifacts during sleep.
  > 5. **Missingness & Noise Audit:** Assesses signal-to-noise ratio and provides a data cleanliness score.
  >
  > When I click **'Load Pre-packaged Sample'**, you can see all 5 validation checks turn **Green (PASSED)**. The table displays 96 synchronized 15-minute observations over a full 24-hour cycle, ready for instantaneous twin assimilation."

#### 14:15 – 16:00 | Live Demonstration — Tab 4: Cohort Analytics & Population Health
- **[Screen Action]:** Click on Tab 4 (*Cohort Analytics*). Scroll through the 4 KPI metric cards, the ADA 7.0% distribution histogram, and the High-Risk Patient Registry table.
- **[Spoken Words]:**
  > "Finally, let us look at Tab 4: **Cohort Analytics & Population Health**.
  >
  > GlycoTwin is not just an individual monitoring tool; it is a hospital-wide clinical decision support system.
  >
  > Here we have synthesized a representative cohort of **200 Indian Type 2 Diabetic patients**, calibrated to National Family Health Survey (NFHS-5) and ICMR-INDIAB population statistics.
  >
  > Look at our top KPI cards:
  > - **Cohort Size:** 200 Patients
  > - **Mean Baseline HbA1c:** 6.79%
  > - **ADA Target Benchmark:** Tracking patients against the American Diabetes Association benchmark of $<7.0\%$.
  > - **High Clinical Risk:** 5.0% (exactly 10 flagged patients).
  >
  > In the central distribution chart, you see the population's HbA1c spread, with an explicit red vertical dashed line at the **7.0% ADA Clinical Benchmark**. Clinicians can immediately visualize that 35% of the cohort sits in the moderate sub-optimal range (7.0–8.5%), while 5% are in the uncontrolled danger zone ($>8.5\%$).
  >
  > Below this is the **High-Risk Patient Registry**. In a busy hospital OPD seeing hundreds of patients a day, an endocrinologist cannot review every chart. GlycoTwin automatically triages the queue: Patient **GT-1008**, **GT-1042**, and **GT-1091** are pinned to the top based on severe glycemic volatility and high predicted adverse event probabilities.
  >
  > With a single click on any patient ID, the system pivots back to Tab 1 to inspect their 360° twin profile.
  >
  > I now invite **Yashita Dube** to present our clinical validation metrics, data ethics, and summit conclusions."

---

### PART 4: CLINICAL VALIDATION, DPDP COMPLIANCE & CONCLUSION (16:00 – 20:00)
**Speaker:** Yashita Dube  
**Screen Display:** `docs/GlycoTwin_Summit_Presentation.pdf` (Slides 8, 9, 10)

#### 16:00 – 17:30 | Slide 8: Clinical Validation & Model Performance
- **[Screen Action]:** Switch back to presentation PDF, showing Slide 8 (*Clinical Evaluation & Impact*). Highlight the 4 metric cards: `0.942 AUC-ROC`, `93.5% Sensitivity`, `7.2 mg/dL MAE`, and `99.4% Clarke Grid A+B`.
- **[Spoken Words]:**
  > "Thank you, Shiva! Clinical AI must be held to the highest standards of diagnostic precision. We evaluated GlycoTwin rigorously across standard medical and statistical benchmarks:
  >
  > 1. **0.942 AUC-ROC for Adverse Event Classification:** Demonstrating outstanding discriminative power in separating impending glucose plunges from normal postprandial fluctuations.
  > 2. **93.5% Sensitivity for Hypoglycemia Detection:** Crucially, our cost-sensitive training penalty ensures **zero missed life-threatening nadirs ($<54$ mg/dL)**. In clinical endocrinology, a false positive is easily managed with a finger-stick test, but a false negative can be fatal.
  > 3. **7.2 mg/dL Mean Absolute Error (MAE):** Across our continuous 2-hour forward trajectory horizon, our regressor matches the intrinsic sensor error tolerance of factory-calibrated CGM devices.
  > 4. **99.4% in Clarke Error Grid Zones A & B:** In diabetology, the Clarke Error Grid quantifies clinical decision risk. Over 99.4% of our predictions fall in Zone A (clinically accurate) and Zone B (benign errors that lead to no treatment change). Exactly **zero predictions fell into Zone C, D, or E** — guaranteeing that GlycoTwin will never recommend an erroneous or medically hazardous intervention."

#### 17:30 – 18:45 | Slide 9: Data Ethics & DPDP Act 2023 Compliance
- **[Screen Action]:** Advance to Slide 9 (*Prototype Demonstration & Data Compliance*). Point to DPDP Act compliance notes.
- **[Spoken Words]:**
  > "Healthcare innovation cannot compromise patient privacy. GlycoTwin has been built from the ground up in strict accordance with the **Digital Personal Data Protection (DPDP) Act of 2023** of India:
  >
  > - **Zero Protected Health Information (PHI):** The system operates on synthetic patient profiles generated from statistical distributions of published Indian clinical trials, ensuring zero real-patient re-identification risk.
  > - **Edge Inference Architecture:** Because our scikit-learn models are extremely lightweight, inference runs entirely in local memory on the user's client machine or hospital server. Patient sensor feeds never need to be uploaded to opaque cloud LLM APIs.
  > - **Full Auditability:** Every recommendation is fully reproducible, deterministic, and traceable to explicit physiological features."

#### 18:45 – 20:00 | Slide 10: Conclusion, Open-Source Release & Team Sign-off
- **[Screen Action]:** Advance to Slide 10 (*Conclusion & Phase 1 Submission Summary*), showing the team attribution block and live links.
- **[Spoken Words]:**
  > "In conclusion, **GlycoTwin** demonstrates that digital twin technology can bridge the gap between fragmented episodic care and continuous predictive prevention for India's 101 million diabetic citizens.
  >
  > To summarize what we have delivered for the Happiest Health Healthcare Summit 2026:
  > - ✅ A fully working, production-hosted prototype running 24/7 at **https://glycotwin-app.web.app**.
  > - ✅ An offline, zero-dependency standalone dashboard in **dashboard/index.html**.
  > - ✅ A dual-ensemble scikit-learn HistGradientBoosting architecture with 94.2% AUC-ROC.
  > - ✅ Full open-source release under the permissive **MIT License** on GitHub at `kumarshubhammorya/glycotwin-digital-twin`.
  > - ✅ Comprehensive architecture diagrams, slide decks, and plain-language clinical documentation.
  >
  > On behalf of **Team Classy Leopards** — **Shivam Raj, Shubham Kumar, Shiva Kumar Bachatty, and myself, Yashita Dube** — from the **Goa Institute of Management**, we thank the judges, mentors, and the entire Happiest Health Healthcare Summit committee for this extraordinary opportunity to contribute to the future of healthcare technology in India.
  >
  > We welcome your questions and look forward to Phase 2. Thank you!"
- **[Screen Action]:** Hold screen on Slide 10 for 5 seconds showing team contact information (`shivam.raj2026h@gim.ac.in`), then stop recording at 20:00.

---

## 💡 Practical Recording Tips for Team Classy Leopards

1. **Recording Software:** Use Zoom (Create a meeting, share screen, click 'Record to this Computer') or OBS Studio or Google Meet.
2. **Audio Quality:** Use a decent headset or microphone in a quiet room. Speak at a steady, confident pace.
3. **Live Demo Practice:** Keep `https://glycotwin-app.web.app` open in Chrome beforehand with tabs preloaded.
4. **YouTube Upload:** After recording, upload to YouTube as **Unlisted** (so anyone with the link can view, but it does not appear in public search results). Name the video:  
   *`GlycoTwin — Multimodal Digital Twin for T2D | Team Classy Leopards | Happiest Health Summit 2026`*.
5. **Paste Link in README:** Once uploaded, replace `[Add Unlisted YouTube or Google Drive Video Link here]` in `README.md` with your video link.
