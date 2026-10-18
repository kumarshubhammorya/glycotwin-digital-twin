# GlycoTwin Project Guide for Claude Code

## Project Overview
GlycoTwin is a clinical-grade multimodal digital twin for Type 2 Diabetes (T2D) management, built for the **Happiest Health Healthcare Summit 2026** (Bengaluru).
- **Core Problem:** India's 101M diabetic crisis, South-Asian phenotype (insulin resistance at lower BMI 22–24 kg/m²), and episodic care failure.
- **Goal:** Predict acute adverse events (<70 mg/dL hypoglycemia or >250 mg/dL hyperglycemia) 2 hours in advance with counterfactual "What-If" simulation.
- **Live Deployment:** https://glycotwin-app.web.app (Firebase Hosting)
- **GitHub Repository:** https://github.com/kumarshubhammorya/glycotwin-digital-twin
- **Team:** Classy Leopards · Goa Institute of Management, Sanquelim (Lead: Shivam Raj)

---

## Architecture & Codebase Map

- `dashboard/app.py`: Production Streamlit clinical decision support dashboard (4 modules: Live Monitor, What-If Simulator, Data Ingest, Cohort Analytics).
- `dashboard/build_standalone_dashboard.py`: Python generator that compiles `dashboard/index.html` (zero-server client-side HTML5/Plotly/JS engine).
- `public/index.html`: Web deployment build hosted on Firebase CDN.
- `models/train_fusion_model.py`: Multimodal feature engineering & dual HistGradientBoosting model training.
- `models/glycotwin_model.joblib`: Pretrained serialized scikit-learn models (classifier + regressor).
- `data/generate_datasets.py`: Compartmental synthetic data generator calibrated to Indian clinical literature (ICMR-INDIAB & NFHS-5).
- `data/synthetic_ehr.csv`: 200 Indian patient baseline profiles.
- `data/synthetic_wearables.csv`: 26,560 synchronized 15-minute sensor packets.
- `docs/VIDEO_SCRIPT_20MIN.md`: 20-minute spoken presentation & demonstration guide.
- `docs/GlycoTwin_Summit_Presentation.pptx` & `.pdf`: 10-slide summit pitch deck.
- `docs/GlycoTwin_Architecture_Diagram.pptx` & `.pdf`: 4-layer system architecture diagram.
- `generate_summit_docs.py` & `generate_summit_pdfs.py`: Automated generators for PPTX and PDF presentations.

---

## Strict Rules & Constraints

1. **AI/ML Model Policy:** The model is strictly `scikit-learn` `HistGradientBoostingClassifier` and `HistGradientBoostingRegressor`. Never claim or implement ODE, Kalman filters, or non-deterministic deep learning.
2. **Clinical Safety & Grounding:** No invented numbers or fake metrics. All displayed numbers in UI must be dynamically computed from data or model predictions.
3. **Demo Actions:** EHR audit export and external nudge buttons must be clearly labelled `(Demo)` to signify prototype safety.
4. **Data Privacy:** 100% compliant with India's DPDP Act 2023 and HIPAA. Never inject real patient health information (PHI).

---

## Key Terminal Commands

- **Run Streamlit Dashboard:**
  ```bash
  .venv_redesign/bin/streamlit run dashboard/app.py --server.port 8502
  ```
- **Regenerate Standalone Offline Dashboard (`dashboard/index.html`):**
  ```bash
  .venv_redesign/bin/python dashboard/build_standalone_dashboard.py
  cp dashboard/index.html public/index.html
  ```
- **Deploy to Firebase Hosting:**
  ```bash
  firebase deploy --only hosting
  ```
- **Retrain Models & Re-export:**
  ```bash
  .venv_redesign/bin/python models/train_fusion_model.py
  ```
- **Regenerate PPTX & PDF Presentations:**
  ```bash
  .venv_redesign/bin/python generate_summit_docs.py
  .venv_redesign/bin/python generate_summit_pdfs.py
  ```
