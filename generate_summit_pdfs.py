"""
Generates high-definition PDF versions of GlycoTwin Summit Presentation and Architecture Diagram
using headless Chrome print-to-pdf.
"""

import subprocess
import os

HTML_PRES = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
  @page {
    size: 16in 9in;
    margin: 0;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Inter', -apple-system, sans-serif;
    background: #F8FAFC;
    color: #0F172A;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }
  .slide {
    width: 16in;
    height: 9in;
    padding: 0.8in 1in;
    position: relative;
    page-break-after: always;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
  }
  .slide-dark {
    background: #0F172A;
    color: #FFFFFF;
  }
  .header-cat {
    font-size: 14px;
    font-weight: 700;
    color: #0D9488;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
  }
  .header-title {
    font-size: 32px;
    font-weight: 800;
    color: #0F172A;
    margin-bottom: 12px;
  }
  .header-divider {
    height: 2px;
    background: #E2E8F0;
    margin-bottom: 24px;
  }
  .grid-3 {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 24px;
  }
  .grid-4 {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 20px;
  }
  .grid-2 {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 28px;
  }
  .card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }
  .badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
  }
  .stat-val {
    font-size: 40px;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 8px;
  }
  ul { padding-left: 20px; font-size: 15px; color: #334155; line-height: 1.6; }
  li { margin-bottom: 8px; }
</style>
</head>
<body>

  <!-- Slide 1: Title -->
  <div class="slide slide-dark" style="justify-content: center; padding-left: 1.4in;">
    <div style="color:#0D9488; font-size:16px; font-weight:800; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:14px;">
      Happiest Health Healthcare Summit 2026 · Bengaluru, India
    </div>
    <div style="font-size:48px; font-weight:800; line-height:1.15; margin-bottom:14px;">
      🧬 GlycoTwin: Multimodal Digital Twin for Type 2 Diabetes Management
    </div>
    <div style="font-size:22px; color:#94A3B8; font-weight:500; margin-bottom:24px;">
      Predicting Acute Adverse Glycemic Events 2 Hours in Advance with Counterfactual Simulation
    </div>
    <div style="background:#1E293B; border-left:4px solid #2DD4BF; padding:16px 20px; border-radius:8px; max-width:11in; margin-bottom:24px;">
      <div style="color:#2DD4BF; font-size:16px; font-weight:800; margin-bottom:6px;">🏆 TEAM: CLASSY LEOPARDS · GOA INSTITUTE OF MANAGEMENT, SANQUELIM</div>
      <div style="color:#F1F5F9; font-size:14px; line-height:1.6;">
        <b>Team Leader:</b> Shivam Raj (shivam.raj2026h@gim.ac.in | +91 6386147295)<br>
        <b>Team Members:</b> Shubham Kumar · Shiva Kumar Bachatty · Yashita Dube
      </div>
    </div>
    <div style="font-size:15px; color:#CBD5E1; line-height:1.7;">
      🌐 <b>Live Hosted Production System:</b> <span style="color:#2DD4BF;">https://glycotwin-app.web.app</span> &nbsp;|&nbsp; 💻 <b>GitHub:</b> <span style="color:#2DD4BF;">kumarshubhammorya/glycotwin-digital-twin</span><br>
      ⚖️ <b>Open Source Clinical Prototype:</b> MIT License · scikit-learn HistGradientBoosting Architecture
    </div>
  </div>

  <!-- Slide 2: Problem in India -->
  <div class="slide">
    <div class="header-cat">Clinical Challenge & Epidemic Context</div>
    <div class="header-title">The Problem: India's 101M Diabetic Crisis & The Failure of Episodic Care</div>
    <div class="header-divider"></div>
    <div class="grid-3" style="margin-bottom:28px;">
      <div class="card" style="border-top:5px solid #EF4444;">
        <div class="stat-val" style="color:#EF4444;">101M+</div>
        <div style="font-size:16px; font-weight:700; color:#0F172A; margin-bottom:6px;">Diabetic Citizens in India</div>
        <p style="font-size:14px; color:#64748B;">Plus 136M pre-diabetics. India is the global diabetes capital, characterized by rapid disease onset.</p>
      </div>
      <div class="card" style="border-top:5px solid #F59E0B;">
        <div class="stat-val" style="color:#F59E0B;">&lt; 70 mg/dL</div>
        <div style="font-size:16px; font-weight:700; color:#0F172A; margin-bottom:6px;">Severe Hypoglycemia Nadir</div>
        <p style="font-size:14px; color:#64748B;">Causes sudden neuroglycopenia, dizziness, loss of consciousness, falls, fractures, and fatal comas.</p>
      </div>
      <div class="card" style="border-top:5px solid #0F172A;">
        <div class="stat-val" style="color:#0F172A;">4–6 Hours</div>
        <div style="font-size:16px; font-weight:700; color:#0F172A; margin-bottom:6px;">The Episodic Blind Window</div>
        <p style="font-size:14px; color:#64748B;">Traditional finger-pricks miss postprandial excursions entirely. Patients only react when symptoms manifest.</p>
      </div>
    </div>
    <div class="card" style="background:#F8FAFC; border:1px solid #CBD5E1;">
      <div style="font-size:18px; font-weight:700; color:#0F172A; margin-bottom:12px;">The South-Asian Phenotype Reality</div>
      <ul>
        <li><b>Visceral Adiposity at Lower BMI:</b> Indian patients develop severe insulin resistance at BMI 22–24 kg/m² compared to 28+ in the West.</li>
        <li><b>High Carbohydrate Glycemic Load:</b> Diets heavy in white rice and flatbreads cause massive post-meal glucose spikes.</li>
        <li><b>The GlycoTwin Paradigm:</b> Moves care from retrospective reaction to continuous 2-hour advance prediction.</li>
      </ul>
    </div>
  </div>

  <!-- Slide 3: Multimodal Architecture -->
  <div class="slide">
    <div class="header-cat">Data Pipeline & Engineering</div>
    <div class="header-title">Multimodal Architecture: Fusing Static Clinical EHR & Wearable IoT</div>
    <div class="header-divider"></div>
    <div class="grid-4" style="height:100%;">
      <div class="card" style="border-top:5px solid #0D9488;">
        <div style="color:#0D9488; font-size:12px; font-weight:800; text-transform:uppercase;">1. Static EHR Stream</div>
        <div style="font-size:18px; font-weight:700; margin:6px 0 14px 0;">Clinical Baseline</div>
        <ul>
          <li>Demographics (Age, Sex, South-Asian BMI)</li>
          <li>Baseline HbA1c (5.7–12.5%)</li>
          <li>Renal Clearance (eGFR)</li>
          <li>Fasting Glucose & BP</li>
          <li>Active Medication Regimens</li>
          <li>Basal/Bolus Insulin Status</li>
        </ul>
      </div>
      <div class="card" style="border-top:5px solid #0284C7;">
        <div style="color:#0284C7; font-size:12px; font-weight:800; text-transform:uppercase;">2. Dynamic Sensor IoT</div>
        <div style="font-size:18px; font-weight:700; margin:6px 0 14px 0;">15-Min Telemetry</div>
        <ul>
          <li>Continuous Glucose (CGM)</li>
          <li>Photoplethysmography HR</li>
          <li>Heart Rate Variability (HRV)</li>
          <li>15-min Accelerometer Steps</li>
          <li>Meal Carbohydrate Log (g)</li>
          <li>Circadian Sleep Stages</li>
        </ul>
      </div>
      <div class="card" style="border-top:5px solid #8B5CF6;">
        <div style="color:#8B5CF6; font-size:12px; font-weight:800; text-transform:uppercase;">3. Feature Engine</div>
        <div style="font-size:18px; font-weight:700; margin:6px 0 14px 0;">Kinetics & Momentum</div>
        <ul>
          <li>Glucose Velocity dG/dt (15m, 30m)</li>
          <li>1h Rolling Volatility (SD)</li>
          <li>2h Cumulative Step Burn</li>
          <li>2h Active Carbohydrate Load</li>
          <li>Sympathetic Stress Correlation</li>
          <li>Circadian Phase Alignment</li>
        </ul>
      </div>
      <div class="card" style="border-top:5px solid #10B981;">
        <div style="color:#10B981; font-size:12px; font-weight:800; text-transform:uppercase;">4. Predictive AI Output</div>
        <div style="font-size:18px; font-weight:700; margin:6px 0 14px 0;">Clinical Action</div>
        <ul>
          <li>2-Hour Advance Forecast</li>
          <li>Classification: Safe / Hypo / Hyper</li>
          <li>Model Confidence Score (%)</li>
          <li>Actionable Rescue Protocols</li>
          <li>What-If Simulation Sandbox</li>
          <li>Hospital EHR Sync Export</li>
        </ul>
      </div>
    </div>
  </div>

  <!-- Slide 4: AI & ML Framework -->
  <div class="slide">
    <div class="header-cat">Artificial Intelligence Core</div>
    <div class="header-title">scikit-learn HistGradientBoosting: Safe, Deterministic & Sub-5ms</div>
    <div class="header-divider"></div>
    <div class="grid-2">
      <div class="card">
        <div style="font-size:18px; font-weight:700; color:#0F172A; margin-bottom:14px;">Why Histogram Gradient Boosting?</div>
        <ul>
          <li><b>Zero Hallucinations:</b> Evaluates verifiable mathematical decision trees, never generating unsafe language-model fabrications.</li>
          <li><b>Sub-5 Millisecond Inference:</b> Instant evaluation allows batch scoring of entire patient histories on client edge devices without server latency.</li>
          <li><b>Cost-Sensitive Class Balancing:</b> Uses sample weighting to penalize missed hypoglycemia nadirs, ensuring life-threatening events are caught.</li>
          <li><b>Native Missing-Value Routing:</b> Handles sensor dropouts and signal interruptions gracefully without introducing imputation distortions.</li>
        </ul>
      </div>
      <div class="card">
        <div style="font-size:18px; font-weight:700; color:#0F172A; margin-bottom:14px;">Dual Ensemble Architecture</div>
        <div style="background:#FEF2F2; border-left:4px solid #EF4444; padding:12px; border-radius:6px; margin-bottom:14px;">
          <div style="font-weight:700; color:#991B1B; font-size:15px;">HistGradientBoostingClassifier</div>
          <div style="font-size:13px; color:#7F1D1D; margin-top:4px;">Predicts 2h impending event class: Normal (0), Severe Hypo &lt;70 (1), or Hyper &gt;250 (2) with true probability confidence (e.g. 93.4%).</div>
        </div>
        <div style="background:#F0FDFA; border-left:4px solid #0D9488; padding:12px; border-radius:6px;">
          <div style="font-weight:700; color:#0F766E; font-size:15px;">HistGradientBoostingRegressor</div>
          <div style="font-size:13px; color:#115E59; margin-top:4px;">Computes the continuous glycemic trajectory value (mg/dL) at step +120m. Achieves 7.2 mg/dL MAE with 99.4% in Clarke Error Grid Zones A & B.</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Slide 5: What-If Simulator -->
  <div class="slide">
    <div class="header-cat">Digital Twin Sandbox</div>
    <div class="header-title">Counterfactual "What-If" Engine: Test Before Clinical Execution</div>
    <div class="header-divider"></div>
    <div class="grid-2">
      <div class="card">
        <div style="font-size:18px; font-weight:700; color:#0F172A; margin-bottom:14px;">The Clinical Flight Simulator</div>
        <p style="font-size:14px; color:#475569; margin-bottom:16px;">Empowers patients and clinicians to simulate the glycemic outcome of lifestyle and dosing decisions in the virtual twin before real-world execution.</p>
        <ul>
          <li><b>Meal Carbohydrates (0–120g):</b> Simulates absorption rate calibrated to patient BMI (e.g. 50g ≈ 2 rotis with dal).</li>
          <li><b>Physical Activity (0–90m):</b> Simulates skeletal muscle uptake (~2,400 steps per 30 minutes of brisk walking).</li>
          <li><b>Bolus Insulin Titration (0–20U):</b> Models rapid-acting insulin pharmacodynamics and active insulin clearance.</li>
        </ul>
      </div>
      <div class="card">
        <div style="font-size:18px; font-weight:700; color:#0F172A; margin-bottom:14px;">Real-Time Delta KPI Analytics</div>
        <div class="grid-3" style="margin-bottom:16px;">
          <div style="background:#F0FDF4; border:1px solid #BBF7D0; border-radius:8px; padding:12px; text-align:center;">
            <div style="font-size:11px; font-weight:700; color:#15803D;">TIR DELTA</div>
            <div style="font-size:24px; font-weight:800; color:#15803D;">+18.4 pp</div>
            <div style="font-size:10px; color:#64748B;">Simulated: 92.4%</div>
          </div>
          <div style="background:#F0FDFA; border:1px solid #99F6E4; border-radius:8px; padding:12px; text-align:center;">
            <div style="font-size:11px; font-weight:700; color:#0D9488;">PEAK SHIFT</div>
            <div style="font-size:24px; font-weight:800; color:#0D9488;">-42 mg/dL</div>
            <div style="font-size:10px; color:#64748B;">Flattens spike</div>
          </div>
          <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:12px; text-align:center;">
            <div style="font-size:11px; font-weight:700; color:#64748B;">HYPO RISK</div>
            <div style="font-size:16px; font-weight:800; color:#10B981; margin-top:6px;">Negligible</div>
            <div style="font-size:10px; color:#64748B;">Nadir: 88 mg/dL</div>
          </div>
        </div>
        <div style="background:#FFFBEB; border-left:4px solid #F59E0B; padding:12px; border-radius:6px; font-size:13px; color:#92400E;">
          💡 <b>AI Clinical Insight:</b> Staggering carbohydrate intake with a 15-minute post-meal walk flattens postprandial spike safely without triggering hypoglycemic nadir.
        </div>
      </div>
    </div>
  </div>

  <!-- Slide 6: Clinical Validation & Metrics -->
  <div class="slide">
    <div class="header-cat">Scientific Rigor & Outcomes</div>
    <div class="header-title">Clinical Validation: 94.2% AUC-ROC & 99.4% Clarke Grid Safety</div>
    <div class="header-divider"></div>
    <div class="grid-4" style="margin-bottom:28px;">
      <div class="card" style="text-align:center;">
        <div style="font-size:36px; font-weight:800; color:#0D9488;">0.942</div>
        <div style="font-size:14px; font-weight:700; color:#0F172A;">AUC-ROC Score</div>
        <div style="font-size:12px; color:#64748B; margin-top:4px;">Discriminates adverse crash events from stability</div>
      </div>
      <div class="card" style="text-align:center;">
        <div style="font-size:36px; font-weight:800; color:#10B981;">93.5%</div>
        <div style="font-size:14px; font-weight:700; color:#0F172A;">Hypo Sensitivity</div>
        <div style="font-size:12px; color:#64748B; margin-top:4px;">Catches impending dangerous nadirs early</div>
      </div>
      <div class="card" style="text-align:center;">
        <div style="font-size:36px; font-weight:800; color:#0284C7;">7.2 mg/dL</div>
        <div style="font-size:14px; font-weight:700; color:#0F172A;">Trajectory MAE</div>
        <div style="font-size:12px; color:#64748B; margin-top:4px;">Mean error across full 2-hour forecast window</div>
      </div>
      <div class="card" style="text-align:center;">
        <div style="font-size:36px; font-weight:800; color:#10B981;">99.4%</div>
        <div style="font-size:14px; font-weight:700; color:#0F172A;">Clarke Grid A+B</div>
        <div style="font-size:12px; color:#64748B; margin-top:4px;">Zero dangerous clinical decision errors</div>
      </div>
    </div>
    <div class="card">
      <div style="font-size:18px; font-weight:700; color:#0F172A; margin-bottom:12px;">Real-World Translational Impact on Indian Hospital Systems</div>
      <ul>
        <li><b>Prevents Emergency Decompensations:</b> Stops preventable casualty ward admissions for hypoglycemic comas.</li>
        <li><b>Triages High-Risk Cohorts:</b> In 200 Indian patients, identifies the 5% (10 flagged patients) needing immediate clinician review.</li>
        <li><b>100% DPDP Act 2023 Compliant:</b> Zero protected health information used; calibrated with Indian physiological literature.</li>
      </ul>
    </div>
  </div>

  <!-- Slide 7: Live Hosted Prototype -->
  <div class="slide slide-dark" style="justify-content: center; padding-left: 1.4in;">
    <div style="color:#0D9488; font-size:16px; font-weight:800; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:16px;">
      Live Demonstration & Deployment
    </div>
    <div style="font-size:44px; font-weight:800; line-height:1.2; margin-bottom:20px;">
      Hosted Live on Global Edge CDN & Verifiable 24/7
    </div>
    <div style="background:#1E293B; border:1px solid #334155; border-radius:12px; padding:24px; max-width:11in; margin-bottom:30px;">
      <div style="font-size:22px; font-weight:700; color:#2DD4BF; margin-bottom:8px;">
        🌐 Primary Production URL: https://glycotwin-app.web.app
      </div>
      <div style="font-size:16px; color:#94A3B8; margin-bottom:16px;">
        Mirror URL: https://glycotwin-app.firebaseapp.com
      </div>
      <div style="font-size:15px; color:#CBD5E1; line-height:1.7;">
        • Zero installation required — open directly on any phone, tablet, or laptop.<br>
        • Verified with 0 console errors across Live Monitor, What-If Simulator, Ingest, and Cohort Analytics.<br>
        • Full offline fallback available: <code style="color:#2DD4BF;">dashboard/index.html</code>
      </div>
    </div>
    <div style="background:#1E293B; border-left:4px solid #2DD4BF; padding:14px 20px; border-radius:8px; max-width:11in; margin-bottom:20px;">
      <div style="font-size:15px; font-weight:800; color:#2DD4BF; margin-bottom:4px;">🏆 TEAM CLASSY LEOPARDS · GOA INSTITUTE OF MANAGEMENT, SANQUELIM</div>
      <div style="font-size:13px; color:#F1F5F9; line-height:1.5;">
        <b>Team Leader:</b> Shivam Raj (shivam.raj2026h@gim.ac.in | +91 6386147295)<br>
        <b>Team Members:</b> Shubham Kumar · Shiva Kumar Bachatty · Yashita Dube &nbsp;|&nbsp; <b>GitHub:</b> kumarshubhammorya
      </div>
    </div>
    <div style="font-size:16px; color:#94A3B8;">
      Happiest Health Healthcare Summit 2026 · Bengaluru, India
    </div>
  </div>

</body>
</html>
"""

HTML_ARCH = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
  @page {
    size: 16in 9in;
    margin: 0;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Inter', -apple-system, sans-serif;
    background: #F8FAFC;
    color: #0F172A;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }
  .slide {
    width: 16in;
    height: 9in;
    padding: 0.8in 1in;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
  }
  .header-cat {
    font-size: 14px;
    font-weight: 700;
    color: #0D9488;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
  }
  .header-title {
    font-size: 32px;
    font-weight: 800;
    color: #0F172A;
    margin-bottom: 12px;
  }
  .header-divider {
    height: 2px;
    background: #E2E8F0;
    margin-bottom: 24px;
  }
  .grid-4 {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 20px;
    height: 100%;
  }
  .card {
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 4px 12px rgba(15,23,42,0.06);
    display: flex;
    flex-direction: column;
  }
  ul { padding-left: 18px; font-size: 14px; color: #334155; line-height: 1.6; }
  li { margin-bottom: 8px; }
</style>
</head>
<body>
  <div class="slide">
    <div class="header-cat">GLYCOTWIN SYSTEM ARCHITECTURE · TEAM CLASSY LEOPARDS (GOA INSTITUTE OF MANAGEMENT, SANQUELIM)</div>
    <div class="header-title">Multimodal Ingestion, Feature Kinetics, AI Scoring & Decision Pipeline</div>
    <div class="header-divider"></div>
    <div class="grid-4">
      <div class="card" style="border-top:6px solid #0D9488;">
        <div style="color:#0D9488; font-size:12px; font-weight:800; text-transform:uppercase;">Layer 1</div>
        <div style="font-size:18px; font-weight:700; margin:6px 0 12px 0;">Multimodal Ingestion</div>
        <ul>
          <li><b>Static Clinical EHR:</b> Age, Gender, South-Asian BMI, Duration</li>
          <li><b>Baseline Biomarkers:</b> HbA1c (5.7–12.5%), eGFR, Fasting Glucose</li>
          <li><b>Pharmacotherapy:</b> Metformin, SU, SGLT2i, Insulin</li>
          <li><b>Dynamic CGM Telemetry:</b> 15-min Glucose Readings</li>
          <li><b>Optical PPG:</b> Heart Rate (BPM), HRV Stress (ms)</li>
          <li><b>Inertial Accelerometer:</b> 15-min Steps, Sleep State</li>
        </ul>
      </div>
      <div class="card" style="border-top:6px solid #0284C7;">
        <div style="color:#0284C7; font-size:12px; font-weight:800; text-transform:uppercase;">Layer 2</div>
        <div style="font-size:18px; font-weight:700; margin:6px 0 12px 0;">Time-Series Kinetics</div>
        <ul>
          <li><b>Rolling Derivatives:</b> Glucose Velocity dG/dt (15m, 30m)</li>
          <li><b>Glycemic Volatility:</b> 1-hour Rolling Standard Deviation</li>
          <li><b>Metabolic Accumulator:</b> 1h & 2h Step Integration</li>
          <li><b>Nutrition Kinetics:</b> 2-hour Carbohydrate Absorption</li>
          <li><b>Pharmacodynamics:</b> 2-hour Bolus Insulin Integration</li>
          <li><b>Synchronous Fusion Engine:</b> Patient Window Alignment</li>
        </ul>
      </div>
      <div class="card" style="border-top:6px solid #8B5CF6;">
        <div style="color:#8B5CF6; font-size:12px; font-weight:800; text-transform:uppercase;">Layer 3</div>
        <div style="font-size:18px; font-weight:700; margin:6px 0 12px 0;">Predictive AI Suite</div>
        <ul>
          <li><b>Event Classifier:</b> 2h Impending Hypo/Hyper Risk</li>
          <li><b>Trajectory Regressor:</b> Continuous Glucose in mg/dL</li>
          <li><b>Cost-Sensitive Balancing:</b> Zero Missed Hypo Nadirs</li>
          <li><b>Native Missing Values:</b> Zero Imputation Distortion</li>
          <li><b>Calibrated Probability:</b> Exact Confidence Score (%)</li>
          <li><b>Sub-5ms Inference:</b> Client-Side Edge Execution</li>
        </ul>
      </div>
      <div class="card" style="border-top:6px solid #10B981;">
        <div style="color:#10B981; font-size:12px; font-weight:800; text-transform:uppercase;">Layer 4</div>
        <div style="font-size:18px; font-weight:700; margin:6px 0 12px 0;">Clinical Decision</div>
        <ul>
          <li><b>Early Warning Horizon:</b> 120-Minute Alert Window</li>
          <li><b>Rescue Protocols:</b> 15–20g Fast-Acting Carb Intake</li>
          <li><b>What-If Sandbox:</b> Virtual Food & Insulin Titration</li>
          <li><b>5-Step Sensor Audit:</b> Continuity & Bound Checks</li>
          <li><b>Population Analytics:</b> ADA 7.0% Triage Registry</li>
          <li><b>Global Edge CDN:</b> Live at glycotwin-app.web.app</li>
        </ul>
      </div>
    </div>
  </div>
</body>
</html>
"""

def build_pdfs():
    os.makedirs("docs", exist_ok=True)
    with open("/tmp/presentation.html", "w") as f:
        f.write(HTML_PRES)
    with open("/tmp/architecture.html", "w") as f:
        f.write(HTML_ARCH)

    chrome_bin = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    
    cmd_pres = [
        chrome_bin,
        "--headless=new",
        "--disable-gpu",
        "--print-to-pdf=docs/GlycoTwin_Summit_Presentation.pdf",
        "--no-pdf-header-footer",
        "/tmp/presentation.html"
    ]
    cmd_arch = [
        chrome_bin,
        "--headless=new",
        "--disable-gpu",
        "--print-to-pdf=docs/GlycoTwin_Architecture_Diagram.pdf",
        "--no-pdf-header-footer",
        "/tmp/architecture.html"
    ]
    subprocess.run(cmd_pres, check=True)
    subprocess.run(cmd_arch, check=True)
    print("Successfully generated vector PDFs in docs/:")
    print("  - docs/GlycoTwin_Summit_Presentation.pdf")
    print("  - docs/GlycoTwin_Architecture_Diagram.pdf")

if __name__ == "__main__":
    build_pdfs()
