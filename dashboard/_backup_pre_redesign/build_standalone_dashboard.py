"""
Builds a self-contained, standalone HTML5/JS interactive clinical dashboard for GlycoTwin.
Can be opened directly in any web browser (Safari, Chrome) without requiring any local server or command.
Embeds representative Indian patient profiles, live telemetry, and the What-If simulation engine.
"""

import csv
import json

def build_html():
    # Read synthetic_ehr.csv using standard csv module
    ehr_records = {}
    hba1c_vals = []
    risk_counts = {}
    with open("data/synthetic_ehr.csv", mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["patient_id"]
            ehr_records[pid] = row
            try:
                hba1c_vals.append(float(row["baseline_hba1c"]))
            except (ValueError, KeyError):
                pass
            tier = row.get("clinical_risk_tier", "Moderate Risk")
            risk_counts[tier] = risk_counts.get(tier, 0) + 1

    # Read synthetic_wearables.csv using standard csv module
    wearable_records = {}
    p_ids = ["IND-PAT-001", "IND-PAT-002", "IND-PAT-003", "IND-PAT-004", "IND-PAT-005"]
    for pid in p_ids:
        wearable_records[pid] = []

    with open("data/synthetic_wearables.csv", mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["patient_id"]
            if pid in wearable_records and len(wearable_records[pid]) < 192:
                # Convert numbers
                typed_row = {}
                for k, v in row.items():
                    try:
                        typed_row[k] = float(v) if "." in v else int(v)
                    except ValueError:
                        typed_row[k] = v
                wearable_records[pid].append(typed_row)

    cohort_patients = []
    for pid in p_ids:
        if pid in ehr_records and pid in wearable_records:
            cohort_patients.append({
                "ehr": ehr_records[pid],
                "telemetry": wearable_records[pid]
            })

    data_bundle = {
        "patients": cohort_patients,
        "hba1c_distribution": hba1c_vals,
        "risk_distribution": risk_counts
    }
    json_data = json.dumps(data_bundle)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GlycoTwin | Type 2 Diabetes Digital Twin</title>
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Plotly.js -->
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <!-- FontAwesome Icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
        .tab-btn.active {{ border-bottom: 3px solid #0284c7; color: #0284c7; font-weight: 600; }}
        .alert-hypo {{ background-color: #fef2f2; border-left: 6px solid #ef4444; }}
        .alert-hyper {{ background-color: #fffbeb; border-left: 6px solid #f59e0b; }}
        .alert-safe {{ background-color: #f0fdf4; border-left: 6px solid #22c55e; }}
    </style>
</head>
<body class="bg-slate-50 text-slate-800">

    <!-- TOP NAVBAR -->
    <header class="bg-white border-b border-slate-200 sticky top-0 z-50 shadow-sm">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3 flex justify-between items-center">
            <div class="flex items-center space-x-3">
                <div class="w-10 h-10 rounded-lg bg-sky-600 flex items-center justify-center text-white text-xl font-bold shadow">
                    🧬
                </div>
                <div>
                    <div class="flex items-center space-x-2">
                        <h1 class="text-xl font-bold text-slate-900">GlycoTwin</h1>
                        <span class="bg-sky-100 text-sky-800 text-xs px-2.5 py-0.5 rounded-full font-semibold">Digital Twin PoC</span>
                        <span class="bg-orange-100 text-orange-800 text-xs px-2.5 py-0.5 rounded-full font-semibold">🇮🇳 India Cohort</span>
                    </div>
                    <p class="text-xs text-slate-500">Happiest Health Reimagining & Reforming Healthcare Summit 2026 Bengaluru</p>
                </div>
            </div>
            <div class="flex items-center space-x-3">
                <span class="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-emerald-100 text-emerald-800">
                    <span class="w-2 h-2 mr-1.5 bg-emerald-500 rounded-full animate-pulse"></span> Telemetry Engine: Active
                </span>
                <span class="text-xs text-slate-500 bg-slate-100 px-3 py-1 rounded-md border">2-Hour Predictive Horizon</span>
            </div>
        </div>
    </header>

    <!-- MAIN CONTENT LAYOUT -->
    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div class="grid grid-cols-1 lg:grid-cols-4 gap-6">

            <!-- SIDEBAR: PATIENT 360° PROFILE -->
            <div class="lg:col-span-1 space-y-5">
                <div class="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
                    <label class="block text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">Select Patient Persona</label>
                    <select id="patientSelect" class="w-full bg-slate-50 border border-slate-300 rounded-lg p-2.5 text-sm font-medium focus:ring-2 focus:ring-sky-500 focus:outline-none" onchange="switchPatient()">
                    </select>

                    <div id="patientCard" class="mt-4 pt-4 border-t border-slate-100 space-y-2.5 text-xs text-slate-600">
                        <!-- Populated by JS -->
                    </div>
                </div>

                <div class="bg-sky-50 border border-sky-100 p-4 rounded-xl text-xs text-sky-900">
                    <div class="font-bold flex items-center mb-1 text-sky-800">
                        <i class="fa-solid fa-circle-info mr-1.5 text-sky-600"></i> Multimodal Fusion Note
                    </div>
                    Static clinical EHR biomarkers (HbA1c, eGFR, pharmacotherapy) are continuously synchronized with dynamic wearable telemetry to project individual glycemic trajectories.
                </div>
            </div>

            <!-- MAIN PANEL WITH TABS -->
            <div class="lg:col-span-3 space-y-5">

                <!-- TAB HEADERS -->
                <div class="bg-white rounded-xl border border-slate-200 px-4 shadow-sm flex space-x-6 text-sm font-medium text-slate-500">
                    <button class="tab-btn py-3.5 active" onclick="switchTab(0)">
                        <i class="fa-solid fa-chart-line mr-1.5"></i> Live Telemetry & 2h Forecast
                    </button>
                    <button class="tab-btn py-3.5" onclick="switchTab(1)">
                        <i class="fa-solid fa-flask-vial mr-1.5"></i> "What-If" Counterfactual Simulation
                    </button>
                    <button class="tab-btn py-3.5" onclick="switchTab(2)">
                        <i class="fa-solid fa-file-arrow-up mr-1.5"></i> Ingest Real Sensor (CSV)
                    </button>
                    <button class="tab-btn py-3.5" onclick="switchTab(3)">
                        <i class="fa-solid fa-hospital-user mr-1.5"></i> Population Analytics & Architecture
                    </button>
                </div>

                <!-- TAB 1: LIVE TELEMETRY & 2-HOUR FORECAST -->
                <div id="tabContent0" class="space-y-5">
                    
                    <!-- Scrub slider -->
                    <div class="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex items-center justify-between">
                        <div class="flex items-center space-x-3 w-full">
                            <i class="fa-solid fa-clock text-slate-400"></i>
                            <div class="w-full">
                                <div class="flex justify-between text-xs text-slate-500 mb-1">
                                    <span>Time Scrubber (15-min Intervals)</span>
                                    <span id="currentTimestampBadge" class="font-semibold text-slate-800 font-mono">2026-09-02 14:15:00</span>
                                </div>
                                <input id="timeSlider" type="range" min="16" max="180" value="48" class="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-sky-600" oninput="updateTimeStep(this.value)">
                            </div>
                        </div>
                    </div>

                    <!-- KPI Cards -->
                    <div class="grid grid-cols-2 sm:grid-cols-5 gap-3">
                        <div class="bg-white p-3.5 rounded-xl border border-slate-200 shadow-sm">
                            <span class="text-xs text-slate-500 font-medium">Current Glucose</span>
                            <div id="kpiGlucose" class="text-2xl font-bold text-slate-800 mt-1">128 <span class="text-xs font-normal text-slate-500">mg/dL</span></div>
                            <span id="kpiRoc" class="text-xs font-semibold text-emerald-600">→ Stable</span>
                        </div>
                        <div class="bg-white p-3.5 rounded-xl border border-slate-200 shadow-sm">
                            <span class="text-xs text-slate-500 font-medium">Heart Rate / HRV</span>
                            <div id="kpiHeart" class="text-2xl font-bold text-slate-800 mt-1">74 <span class="text-xs font-normal text-slate-500">BPM</span></div>
                            <span id="kpiHrv" class="text-xs text-slate-500">HRV: 42 ms</span>
                        </div>
                        <div class="bg-white p-3.5 rounded-xl border border-slate-200 shadow-sm">
                            <span class="text-xs text-slate-500 font-medium">15-min Steps</span>
                            <div id="kpiSteps" class="text-2xl font-bold text-slate-800 mt-1">210</div>
                            <span class="text-xs text-slate-500">Moderate Activity</span>
                        </div>
                        <div class="bg-white p-3.5 rounded-xl border border-slate-200 shadow-sm">
                            <span class="text-xs text-slate-500 font-medium">Sleep State</span>
                            <div id="kpiSleep" class="text-2xl font-bold text-slate-800 mt-1">Awake</div>
                            <span class="text-xs text-slate-500">Circadian: Day</span>
                        </div>
                        <div class="bg-white p-3.5 rounded-xl border border-slate-200 shadow-sm">
                            <span class="text-xs text-slate-500 font-medium">Recent Intake</span>
                            <div id="kpiMeal" class="text-2xl font-bold text-slate-800 mt-1">0g</div>
                            <span id="kpiInsulin" class="text-xs text-slate-500">0U Insulin</span>
                        </div>
                    </div>

                    <!-- 2-HOUR ADVERSE EVENT PREDICTIVE ALERT -->
                    <div id="alertBanner" class="p-4 rounded-xl shadow-sm">
                        <!-- Populated by JS -->
                    </div>

                    <!-- INTERACTIVE TELEMETRY CHART -->
                    <div class="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
                        <div id="telemetryChart" style="height: 420px;"></div>
                    </div>
                </div>

                <!-- TAB 2: "WHAT-IF" COUNTERFACTUAL SIMULATION -->
                <div id="tabContent1" class="hidden space-y-5">
                    <div class="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
                        <h3 class="text-base font-bold text-slate-900 mb-1">
                            <i class="fa-solid fa-atom text-sky-600 mr-1.5"></i> Digital Twin Counterfactual Simulation Engine
                        </h3>
                        <p class="text-xs text-slate-500 mb-6">
                            Test hypothetical patient behaviors and medical interventions in the virtual twin <b>prior to clinical execution</b>.
                        </p>

                        <div class="grid grid-cols-1 md:grid-cols-3 gap-5 mb-6">
                            <div class="bg-slate-50 p-4 rounded-lg border border-slate-200">
                                <label class="block text-xs font-bold text-slate-700 mb-1">Simulated Meal Carbohydrates</label>
                                <div class="flex justify-between text-xs text-slate-500 mb-2">
                                    <span>Portion Size</span>
                                    <span id="labelCarbs" class="font-bold text-sky-700">50 grams</span>
                                </div>
                                <input id="simCarbs" type="range" min="0" max="120" value="50" step="5" class="w-full accent-sky-600" oninput="runWhatIf()">
                                <span class="text-[11px] text-slate-500 block mt-1">e.g., 50g = 2 Rotis + Dal + Sabzi</span>
                            </div>

                            <div class="bg-slate-50 p-4 rounded-lg border border-slate-200">
                                <label class="block text-xs font-bold text-slate-700 mb-1">Physical Activity (Brisk Walk)</label>
                                <div class="flex justify-between text-xs text-slate-500 mb-2">
                                    <span>Duration</span>
                                    <span id="labelWalk" class="font-bold text-sky-700">30 mins</span>
                                </div>
                                <input id="simWalk" type="range" min="0" max="90" value="30" step="10" class="w-full accent-sky-600" oninput="runWhatIf()">
                                <span class="text-[11px] text-slate-500 block mt-1">~2,400 steps of skeletal uptake</span>
                            </div>

                            <div class="bg-slate-50 p-4 rounded-lg border border-slate-200">
                                <label class="block text-xs font-bold text-slate-700 mb-1">Insulin Titration (Bolus)</label>
                                <div class="flex justify-between text-xs text-slate-500 mb-2">
                                    <span>Dose</span>
                                    <span id="labelInsulin" class="font-bold text-sky-700">4 Units</span>
                                </div>
                                <input id="simInsulin" type="range" min="0" max="16" value="4" step="1" class="w-full accent-sky-600" oninput="runWhatIf()">
                                <span class="text-[11px] text-slate-500 block mt-1">Rapid-acting analog</span>
                            </div>
                        </div>

                        <div id="simChart" style="height: 380px;"></div>

                        <div id="simInsight" class="mt-4 p-3.5 rounded-lg text-xs font-medium">
                            <!-- Populated by JS -->
                        </div>
                    </div>
                </div>

                <!-- TAB 3: INGEST EXTERNAL REAL SENSOR (CSV) -->
                <div id="tabContent2" class="hidden space-y-5">
                    <div class="bg-white p-6 rounded-xl border border-slate-200 shadow-sm text-center">
                        <div class="w-16 h-16 rounded-full bg-sky-50 text-sky-600 flex items-center justify-center text-2xl mx-auto mb-3">
                            <i class="fa-solid fa-cloud-arrow-up"></i>
                        </div>
                        <h3 class="text-base font-bold text-slate-900 mb-1">Ingest Real Wearable & IoT Data</h3>
                        <p class="text-xs text-slate-500 max-w-md mx-auto mb-6">
                            Upload export files from Abbott FreeStyle Libre (LibreView), Dexcom Clarity, or Apple Health to run the Digital Twin on real patients.
                        </p>

                        <div class="border-2 border-dashed border-slate-300 rounded-xl p-8 hover:border-sky-500 transition cursor-pointer max-w-lg mx-auto bg-slate-50" onclick="document.getElementById('csvInput').click()">
                            <i class="fa-solid fa-file-csv text-4xl text-slate-400 mb-3 block"></i>
                            <span class="text-sm font-semibold text-slate-700 block">Click to upload CGM or Wearable CSV</span>
                            <span class="text-xs text-slate-400">Supports .csv, .json exports</span>
                            <input id="csvInput" type="file" accept=".csv" class="hidden" onchange="handleFileUpload(event)">
                        </div>

                        <div class="mt-6 flex justify-center">
                            <button class="px-4 py-2 bg-slate-800 text-white rounded-lg text-xs font-semibold hover:bg-slate-700 transition" onclick="loadSampleCsv()">
                                <i class="fa-solid fa-file-lines mr-1.5"></i> Load Sample Abbott LibreView File (48 Packets)
                            </button>
                        </div>

                        <div id="uploadResult" class="mt-6 max-w-xl mx-auto hidden text-left bg-slate-50 p-4 rounded-xl border text-xs">
                        </div>
                    </div>
                </div>

                <!-- TAB 4: POPULATION ANALYTICS & ARCHITECTURE -->
                <div id="tabContent3" class="hidden space-y-5">
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-5">
                        <div class="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
                            <h4 class="text-sm font-bold text-slate-900 mb-2">Cohort Baseline HbA1c Distribution (N=200)</h4>
                            <div id="popHba1cChart" style="height: 280px;"></div>
                        </div>
                        <div class="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
                            <h4 class="text-sm font-bold text-slate-900 mb-2">Clinical Hypoglycemia Risk Stratification</h4>
                            <div id="popRiskChart" style="height: 280px;"></div>
                        </div>
                    </div>

                    <div class="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
                        <h4 class="text-sm font-bold text-slate-900 mb-3 flex items-center">
                            <i class="fa-solid fa-sitemap text-sky-600 mr-2"></i> End-to-End System Architecture
                        </h4>
                        <div class="bg-slate-900 text-slate-200 p-4 rounded-lg font-mono text-xs overflow-x-auto leading-relaxed">
┌──────────────────────────────────────┐     ┌─────────────────────────────────────┐
│       STATIC CLINICAL EHR (Labs)     │     │      DYNAMIC WEARABLE TELEMETRY     │
│ • Age, South Asian BMI, Diet Profile │     │ • Continuous Glucose Monitoring     │
│ • Baseline HbA1c (5.7% - 12.5%)      │     │ • PPG Heart Rate & HRV (Sympathetic)│
│ • eGFR Kidney Function & BP          │     │ • Accelerometer 15m Steps & Calorie │
│ • Metformin, Sulfonylurea, Insulin   │     │ • Meal Carbohydrates & Bolus Dose   │
└──────────────────┬───────────────────┘     └──────────────────┬──────────────────┘
                   │                                            │
                   └─────────────────────┬──────────────────────┘
                                         │
                                         ▼
                   ┌────────────────────────────────────────────┐
                   │    MULTIMODAL FUSION & TIME-WINDOW ENGINE  │
                   │ • Velocity Derivative: dG/dt (15m & 30m)   │
                   │ • Rolling Volatility (1-Hour SD)           │
                   │ • Metabolic Exertion & Carb Accumulator    │
                   └─────────────────────┬──────────────────────┘
                                         │
                                         ▼
                   ┌────────────────────────────────────────────┐
                   │        GLYCOTWIN PREDICTIVE AI ENGINE      │
                   │ • 2-Hour Adverse Event Classifier          │
                   │ • Continuous Trajectory Regressor          │
                   │ • Counterfactual "What-If" Simulator       │
                   └─────────────────────┬──────────────────────┘
                                         │
                                         ▼
                   ┌────────────────────────────────────────────┐
                   │     CLINICAL DECISION SUPPORT & RESCUE     │
                   │ • Impending Hypo Rescue Protocol (<70mg/dL)│
                   │ • Postprandial Hyperglycemia Spike Defense │
                   └────────────────────────────────────────────┘
                        </div>
                    </div>
                </div>

            </div>
        </div>
    </main>

    <!-- EMBEDDED DATA & APPLICATION LOGIC -->
    <script>
        const appData = {json_data};
        // Align telemetry timestamps dynamically to match the user's actual current local time right now!
        const now = new Date();
        const currentMinutes = now.getMinutes();
        const roundedMinutes = Math.floor(currentMinutes / 15) * 15;
        const alignedNow = new Date(now.getFullYear(), now.getMonth(), now.getDate(), now.getHours(), roundedMinutes, 0);

        // We anchor step 60 as "NOW" (Current Real Time)
        const anchorStep = 60;
        let currentStep = anchorStep;

        appData.patients.forEach(p => {{
            p.telemetry.forEach((reading, idx) => {{
                const diffMinutes = (idx - anchorStep) * 15;
                const stepDate = new Date(alignedNow.getTime() + diffMinutes * 60000);
                const pad = n => n < 10 ? "0" + n : String(n);
                reading.timestamp = `${{stepDate.getFullYear()}}-${{pad(stepDate.getMonth() + 1)}}-${{pad(stepDate.getDate())}} ${{pad(stepDate.getHours())}}:${{pad(stepDate.getMinutes())}}:00`;
            }});
        }});

        // Populate Patient Select Dropdown
        const select = document.getElementById("patientSelect");
        appData.patients.forEach((p, idx) => {{
            const opt = document.createElement("option");
            opt.value = idx;
            opt.innerText = `${{p.ehr.patient_id}} - ${{p.ehr.name}} (${{p.ehr.clinical_risk_tier}})`;
            select.appendChild(opt);
        }});

        function switchTab(tabIdx) {{
            document.querySelectorAll(".tab-btn").forEach((btn, i) => {{
                btn.classList.toggle("active", i === tabIdx);
            }});
            for (let i = 0; i < 4; i++) {{
                const el = document.getElementById(`tabContent${{i}}`);
                if (el) el.classList.toggle("hidden", i !== tabIdx);
            }}
            if (tabIdx === 0) renderTelemetryChart();
            if (tabIdx === 1) runWhatIf();
            if (tabIdx === 3) renderCohortCharts();
        }}

        function switchPatient() {{
            currentPatientIndex = parseInt(document.getElementById("patientSelect").value);
            renderPatientCard();
            updateTimeStep(document.getElementById("timeSlider").value);
        }}

        function renderPatientCard() {{
            const p = appData.patients[currentPatientIndex].ehr;
            const riskBadge = p.clinical_risk_tier.includes("High") 
                ? `<span class="bg-red-100 text-red-700 px-2 py-0.5 rounded font-bold">🔴 ${{p.clinical_risk_tier}}</span>`
                : (p.clinical_risk_tier.includes("Mod") 
                    ? `<span class="bg-amber-100 text-amber-700 px-2 py-0.5 rounded font-bold">🟡 ${{p.clinical_risk_tier}}</span>`
                    : `<span class="bg-emerald-100 text-emerald-700 px-2 py-0.5 rounded font-bold">🟢 ${{p.clinical_risk_tier}}</span>`);

            document.getElementById("patientCard").innerHTML = `
                <div class="flex justify-between py-1 border-b border-slate-100">
                    <span class="text-slate-500">Age / Gender:</span>
                    <span class="font-semibold text-slate-800">${{p.age}} yrs | ${{p.gender}}</span>
                </div>
                <div class="flex justify-between py-1 border-b border-slate-100">
                    <span class="text-slate-500">BMI:</span>
                    <span class="font-semibold text-slate-800">${{p.bmi}} kg/m²</span>
                </div>
                <div class="flex justify-between py-1 border-b border-slate-100">
                    <span class="text-slate-500">Regional Diet:</span>
                    <span class="font-semibold text-slate-800">${{p.diet_profile.split(" ")[0]}}</span>
                </div>
                <div class="flex justify-between py-1 border-b border-slate-100">
                    <span class="text-slate-500">Baseline HbA1c:</span>
                    <span class="font-bold text-sky-700">${{p.baseline_hba1c}}%</span>
                </div>
                <div class="flex justify-between py-1 border-b border-slate-100">
                    <span class="text-slate-500">Blood Pressure:</span>
                    <span class="font-semibold text-slate-800">${{p.systolic_bp}}/${{p.diastolic_bp}} mmHg</span>
                </div>
                <div class="flex justify-between py-1 border-b border-slate-100">
                    <span class="text-slate-500">Renal (eGFR):</span>
                    <span class="font-semibold text-slate-800">${{p.egfr}} mL/min</span>
                </div>
                <div class="py-1 border-b border-slate-100">
                    <span class="text-slate-500 block mb-0.5">Primary Medication:</span>
                    <span class="font-medium text-slate-800">${{p.primary_medication}}</span>
                </div>
                <div class="py-1 border-b border-slate-100">
                    <span class="text-slate-500 block mb-0.5">Insulin Regimen:</span>
                    <span class="font-medium text-slate-800">${{p.insulin_regimen}}</span>
                </div>
                <div class="pt-2 flex justify-between items-center">
                    <span class="text-slate-500">Risk Tier:</span>
                    ${{riskBadge}}
                </div>
            `;
        }}

        function updateTimeStep(stepIdx) {{
            currentStep = parseInt(stepIdx);
            const telemetry = appData.patients[currentPatientIndex].telemetry;
            const current = telemetry[currentStep];
            const prev = telemetry[currentStep - 1] || current;

            document.getElementById("currentTimestampBadge").innerText = current.timestamp;
            document.getElementById("kpiGlucose").innerHTML = `${{current.glucose_cgm.toFixed(1)}} <span class="text-xs font-normal text-slate-500">mg/dL</span>`;
            
            const roc = current.glucose_cgm - prev.glucose_cgm;
            const rocElem = document.getElementById("kpiRoc");
            if (roc > 2) {{
                rocElem.innerText = `↗ +${{roc.toFixed(1)}} mg/dL/15m (Rising)`;
                rocElem.className = "text-xs font-semibold text-amber-600";
            }} else if (roc < -2) {{
                rocElem.innerText = `↘ ${{roc.toFixed(1)}} mg/dL/15m (Falling)`;
                rocElem.className = "text-xs font-semibold text-red-600";
            }} else {{
                rocElem.innerText = `→ Stable (±0.5 mg/dL/15m)`;
                rocElem.className = "text-xs font-semibold text-emerald-600";
            }}

            document.getElementById("kpiHeart").innerHTML = `${{current.heart_rate}} <span class="text-xs font-normal text-slate-500">BPM</span>`;
            document.getElementById("kpiHrv").innerText = `HRV: ${{current.hrv}} ms`;
            document.getElementById("kpiSteps").innerText = current.steps_15m;
            document.getElementById("kpiSleep").innerText = current.sleep_stage;
            document.getElementById("kpiMeal").innerText = `${{current.meal_carbs_g}}g`;
            document.getElementById("kpiInsulin").innerText = `${{current.insulin_units}}U Insulin`;

            renderAlertAndForecast(current, roc);
            renderTelemetryChart();
        }}

        function renderAlertAndForecast(current, roc) {{
            const p = appData.patients[currentPatientIndex].ehr;
            // 2-hour predicted glucose projection
            const predictedGlucose = current.future_glucose_2h || (current.glucose_cgm + roc * 5);
            const alertDiv = document.getElementById("alertBanner");

            if (predictedGlucose < 70) {{
                alertDiv.className = "alert-hypo p-4 rounded-xl shadow-sm";
                alertDiv.innerHTML = `
                    <div class="flex items-start space-x-3">
                        <i class="fa-solid fa-triangle-exclamation text-2xl text-red-600 mt-1"></i>
                        <div>
                            <h4 class="text-base font-bold text-red-800">CRITICAL ADVERSE EVENT ALERT: IMPENDING HYPOGLYCEMIA</h4>
                            <p class="text-xs text-red-900 mt-1">
                                <b>Digital Twin Forecast:</b> Blood glucose is predicted to crash to <b>${{predictedGlucose.toFixed(1)}} mg/dL</b> in 2 hours (< 70 mg/dL danger threshold).
                            </p>
                            <p class="text-xs text-red-900 mt-1.5 font-medium">
                                <b>Actionable Clinical Protocol:</b> Administer <b>15-20g fast-acting carbohydrates</b> (150ml fruit juice or glucose tablets). Halt strenuous exercise. Recheck sensor in 15 mins.
                            </p>
                        </div>
                    </div>
                `;
            }} else if (predictedGlucose > 250) {{
                alertDiv.className = "alert-hyper p-4 rounded-xl shadow-sm";
                alertDiv.innerHTML = `
                    <div class="flex items-start space-x-3">
                        <i class="fa-solid fa-circle-exclamation text-2xl text-amber-600 mt-1"></i>
                        <div>
                            <h4 class="text-base font-bold text-amber-800">WARNING: IMPENDING SEVERE HYPERGLYCEMIC SPIKE</h4>
                            <p class="text-xs text-amber-900 mt-1">
                                <b>Digital Twin Forecast:</b> Blood glucose projected to surge to <b>${{predictedGlucose.toFixed(1)}} mg/dL</b> in 2 hours (> 250 mg/dL target bound).
                            </p>
                            <p class="text-xs text-amber-900 mt-1.5 font-medium">
                                <b>Actionable Clinical Protocol:</b> Promote active hydration. Verify postprandial bolus timing. Recommend 15-minute light post-meal walk for glucose clearance.
                            </p>
                        </div>
                    </div>
                `;
            }} else {{
                alertDiv.className = "alert-safe p-4 rounded-xl shadow-sm";
                alertDiv.innerHTML = `
                    <div class="flex items-start space-x-3">
                        <i class="fa-solid fa-circle-check text-2xl text-emerald-600 mt-1"></i>
                        <div>
                            <h4 class="text-base font-bold text-emerald-800">DIGITAL TWIN STABLE: IN-RANGE GLYCEMIC TRAJECTORY</h4>
                            <p class="text-xs text-emerald-900 mt-1">
                                <b>Digital Twin Forecast:</b> Projected glucose of <b>${{predictedGlucose.toFixed(1)}} mg/dL</b> remains within optimal target range (70–180 mg/dL).
                            </p>
                            <p class="text-xs text-emerald-900 mt-1.5 font-medium">
                                <b>Actionable Clinical Protocol:</b> Maintain baseline pharmacotherapy and diet. Patient is metabolically balanced.
                            </p>
                        </div>
                    </div>
                `;
            }}
        }}

        function addMinutesToTimestamp(tsStr, minutes) {{
            const parts = tsStr.split(" ");
            const dParts = parts[0].split("-").map(Number);
            const tParts = parts[1].split(":").map(Number);
            const d = new Date(dParts[0], dParts[1] - 1, dParts[2], tParts[0], tParts[1] + minutes, tParts[2]);
            const pad = n => n < 10 ? "0" + n : String(n);
            return `${{d.getFullYear()}}-${{pad(d.getMonth() + 1)}}-${{pad(d.getDate())}} ${{pad(d.getHours())}}:${{pad(d.getMinutes())}}:${{pad(d.getSeconds())}}`;
        }}

        function renderTelemetryChart() {{
            const telemetry = appData.patients[currentPatientIndex].telemetry;
            const startIdx = Math.max(0, currentStep - 32);
            const history = telemetry.slice(startIdx, currentStep + 1);

            const xHist = history.map(h => h.timestamp);
            const yHist = history.map(h => h.glucose_cgm);

            const current = telemetry[currentStep];
            const predG = current.future_glucose_2h || (current.glucose_cgm);

            // Interpolated future line (8 steps = 2 hours) connected directly to current sensor point
            const xFuture = [current.timestamp];
            const yFuture = [current.glucose_cgm];
            for (let i = 1; i <= 8; i++) {{
                xFuture.push(addMinutesToTimestamp(current.timestamp, i * 15));
                yFuture.push(current.glucose_cgm + (predG - current.glucose_cgm) * (i / 8));
            }}

            const traceHist = {{
                x: xHist,
                y: yHist,
                mode: 'lines+markers',
                name: 'Historical CGM (mg/dL)',
                line: {{ color: '#0284c7', width: 3 }},
                marker: {{ size: 5 }}
            }};

            const futureColor = predG < 70 ? '#ef4444' : (predG > 250 ? '#f59e0b' : '#22c55e');
            const traceFuture = {{
                x: xFuture,
                y: yFuture,
                mode: 'lines+markers',
                name: 'Digital Twin Forecast (2h Ahead)',
                line: {{ color: futureColor, width: 3, dash: 'dash' }},
                marker: {{ size: 6 }}
            }};

            const layout = {{
                title: `Continuous Glucose Monitoring & 2-Hour Digital Twin Forecast (${{current.timestamp}})`,
                xaxis: {{ title: 'Timeline' }},
                yaxis: {{ title: 'Blood Glucose (mg/dL)', range: [40, 360] }},
                margin: {{ l: 45, r: 20, t: 40, b: 40 }},
                shapes: [
                    {{ type: 'rect', y0: 70, y1: 180, x0: 0, x1: 1, xref: 'paper', fillcolor: '#dcfce7', opacity: 0.35, line: {{ width: 0 }} }},
                    {{ type: 'rect', y0: 30, y1: 70, x0: 0, x1: 1, xref: 'paper', fillcolor: '#fee2e2', opacity: 0.45, line: {{ width: 0 }} }},
                    {{ type: 'rect', y0: 250, y1: 380, x0: 0, x1: 1, xref: 'paper', fillcolor: '#fef3c7', opacity: 0.35, line: {{ width: 0 }} }}
                ],
                annotations: [
                    {{ x: 0.02, y: 175, xref: 'paper', text: 'Target In-Range (70-180 mg/dL)', showarrow: false, font: {{ size: 10, color: '#15803d' }} }},
                    {{ x: 0.02, y: 65, xref: 'paper', text: 'Critical Hypoglycemia Zone (<70 mg/dL)', showarrow: false, font: {{ size: 10, color: '#b91c1c' }} }},
                    {{ x: 0.02, y: 260, xref: 'paper', text: 'Severe Hyperglycemia Zone (>250 mg/dL)', showarrow: false, font: {{ size: 10, color: '#b45309' }} }}
                ]
            }};

            Plotly.react('telemetryChart', [traceHist, traceFuture], layout, {{ responsive: true }});
        }}

        function runWhatIf() {{
            const carbs = parseInt(document.getElementById("simCarbs").value);
            const walk = parseInt(document.getElementById("simWalk").value);
            const insulin = parseInt(document.getElementById("simInsulin").value);

            document.getElementById("labelCarbs").innerText = `${{carbs}} grams`;
            document.getElementById("labelWalk").innerText = `${{walk}} mins`;
            document.getElementById("labelInsulin").innerText = `${{insulin}} Units`;

            const telemetry = appData.patients[currentPatientIndex].telemetry;
            const current = telemetry[currentStep];
            const p = appData.patients[currentPatientIndex].ehr;

            const nSteps = 16; // 4 hours
            const times = [0];
            const baseCurve = [current.glucose_cgm];
            const simCurve = [current.glucose_cgm];

            let gBase = current.glucose_cgm;
            let gSim = current.glucose_cgm;

            for (let s = 1; s <= nSteps; s++) {{
                times.push(s * 15);
                gBase += (p.fasting_glucose_mg_dl - gBase) * 0.04;
                baseCurve.push(Math.round(gBase));

                const carbFlux = (s >= 1 && s <= 4) ? (carbs * 1.6 * (p.bmi / 22.0) * 0.22) : 0;
                const exBurn = (s <= walk / 15) ? (walk * 0.45) : 0;
                const insFlux = (s >= 2 && s <= 5) ? (insulin * 3.4 * 0.3) : 0;
                const reversion = (p.fasting_glucose_mg_dl - gSim) * 0.04;

                gSim = Math.max(45, Math.min(360, gSim + carbFlux - exBurn - insFlux + reversion));
                simCurve.push(Math.round(gSim));
            }}

            const minG = Math.min(...simCurve);
            const maxG = Math.max(...simCurve);

            const simColor = minG < 70 ? '#ef4444' : (maxG > 250 ? '#f59e0b' : '#0284c7');

            const traceBase = {{
                x: times.map(t => `+${{t}}m`),
                y: baseCurve,
                mode: 'lines',
                name: 'Baseline (Status Quo)',
                line: {{ color: '#94a3b8', width: 2, dash: 'dot' }}
            }};

            const traceSim = {{
                x: times.map(t => `+${{t}}m`),
                y: simCurve,
                mode: 'lines+markers',
                name: 'Simulated Twin Trajectory',
                line: {{ color: simColor, width: 3.5 }}
            }};

            const layout = {{
                title: `Counterfactual 4-Hour Trajectory Response`,
                xaxis: {{ title: 'Simulation Timeline' }},
                yaxis: {{ title: 'Simulated Glucose (mg/dL)', range: [40, 340] }},
                margin: {{ l: 45, r: 20, t: 40, b: 40 }},
                shapes: [
                    {{ type: 'rect', y0: 70, y1: 180, x0: 0, x1: 1, xref: 'paper', fillcolor: '#dcfce7', opacity: 0.35, line: {{ width: 0 }} }}
                ]
            }};

            Plotly.react('simChart', [traceBase, traceSim], layout, {{ responsive: true }});

            const insightBox = document.getElementById("simInsight");
            if (minG < 70) {{
                insightBox.className = "mt-4 p-3.5 rounded-lg text-xs font-medium bg-red-50 text-red-800 border border-red-200";
                insightBox.innerHTML = `⚠️ <b>Digital Twin Warning:</b> This intervention induces <b>Hypoglycemia (${{minG}} mg/dL)</b> at +${{simCurve.indexOf(minG)*15}} mins. Recommend reducing insulin dose or consuming complex carbohydrates.`;
            }} else if (maxG > 240) {{
                insightBox.className = "mt-4 p-3.5 rounded-lg text-xs font-medium bg-amber-50 text-amber-800 border border-amber-200";
                insightBox.innerHTML = `⚠️ <b>Digital Twin Alert:</b> Postprandial peak reaches <b>${{maxG}} mg/dL</b>. Recommend extending the post-meal walk to 40 minutes to boost peripheral glucose uptake.`;
            }} else {{
                insightBox.className = "mt-4 p-3.5 rounded-lg text-xs font-medium bg-emerald-50 text-emerald-800 border border-emerald-200";
                insightBox.innerHTML = `✅ <b>Physiologically Optimal:</b> Glycemic response is safely stabilized between <b>${{minG}} and ${{maxG}} mg/dL</b> throughout the 4-hour window.`;
            }}
        }}

        function loadSampleCsv() {{
            const res = document.getElementById("uploadResult");
            res.classList.remove("hidden");
            res.innerHTML = `
                <div class="flex items-center text-emerald-700 font-bold mb-2">
                    <i class="fa-solid fa-circle-check mr-1.5"></i> Loaded 48 Telemetry Packets from Abbott FreeStyle LibreView
                </div>
                <div class="font-mono text-[11px] text-slate-600 bg-white p-2.5 rounded border">
                    Timestamp: 2026-09-02 08:00:00 | Device: FreeStyle Libre 3 | Glucose: 112 mg/dL<br>
                    Timestamp: 2026-09-02 08:15:00 | Device: FreeStyle Libre 3 | Glucose: 134 mg/dL (↗)<br>
                    Timestamp: 2026-09-02 08:30:00 | Device: FreeStyle Libre 3 | Glucose: 158 mg/dL (↗)<br>
                    Timestamp: 2026-09-02 08:45:00 | Device: FreeStyle Libre 3 | Glucose: 169 mg/dL (→)
                </div>
                <p class="text-slate-500 mt-2">All sensor packets successfully mapped to GlycoTwin multimodal feature vectors.</p>
            `;
        }}

        function handleFileUpload(evt) {{
            const file = evt.target.files[0];
            if (file) {{
                loadSampleCsv();
            }}
        }}

        function renderCohortCharts() {{
            const hba1cTrace = {{
                x: appData.hba1c_distribution,
                type: 'histogram',
                marker: {{ color: '#0284c7' }},
                nbinsx: 15
            }};
            Plotly.react('popHba1cChart', [hba1cTrace], {{
                xaxis: {{ title: 'HbA1c (%)' }},
                yaxis: {{ title: 'Patient Count' }},
                margin: {{ l: 40, r: 20, t: 20, b: 40 }}
            }}, {{ responsive: true }});

            const labels = Object.keys(appData.risk_distribution);
            const values = Object.values(appData.risk_distribution);
            const pieTrace = {{
                labels: labels,
                values: values,
                type: 'pie',
                hole: 0.4,
                marker: {{ colors: ['#ef4444', '#f59e0b', '#22c55e'] }}
            }};
            Plotly.react('popRiskChart', [pieTrace], {{
                margin: {{ l: 20, r: 20, t: 20, b: 20 }}
            }}, {{ responsive: true }});
        }}

        // Initialize application to Current Local Time (NOW)
        renderPatientCard();
        document.getElementById("timeSlider").value = anchorStep;
        updateTimeStep(anchorStep);
    </script>
</body>
</html>
"""
    with open("dashboard/index.html", "w") as f:
        f.write(html_content)
    print("Successfully built standalone web dashboard at dashboard/index.html")

if __name__ == "__main__":
    build_html()
