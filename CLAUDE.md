# GlycoTwin: notes for contributors and coding assistants

GlycoTwin is a research prototype for the Happiest Health Digital Twin Challenge 2026: a Type 2 diabetes digital twin that fuses a synthetic EHR with synthetic CGM and wearable streams to forecast hypoglycaemia (<70 mg/dL) and marked hyperglycaemia (>250 mg/dL) 2 hours ahead. Team Classy Leopards, Goa Institute of Management. See README.md for the full description.

## Pipeline and files

1. `data/generate_datasets.py` writes `data/synthetic_ehr.csv` (200 personas) and `data/synthetic_wearables.csv` (200 x 664 = 132,800 labelled 15-min windows). It is seeded and deterministic.
2. `models/train_fusion_model.py` builds the 34 features (18 wearable + 16 EHR) and splits by patient (150 train / 50 held-out). It trains a HistGradientBoosting classifier (`class_weight="balanced"`) and regressor, picks alert thresholds by F2 on grouped CV over the training patients, and writes `models/glycotwin_model.joblib`.
3. `models/evaluate.py` runs held-out scoring, 5-fold grouped CV, baselines, Clarke grid, episodes, ablations, the EHR-swap test and latency, and writes `models/metrics.json`.
4. `models/test_predictions.py` runs read-only PASS/FAIL checks and exits non-zero on failure.
5. `dashboard/app.py` is the Streamlit doctor dashboard; it runs the bundle live.
6. `dashboard/build_standalone_dashboard.py` writes `dashboard/index.html` and `dashboard/plotly.min.js`, with predictions precomputed in Python. Copy both to `public/` for Firebase Hosting.
7. `generate_summit_docs.py` generates the deck and the architecture diagram in `docs/`. It needs `requirements-dev.txt`.

## Commands (from the repository root)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # add requirements-dev.txt for the deck generator
python data/generate_datasets.py           # optional, deterministic
python models/train_fusion_model.py        # optional, overwrites the shipped bundle
python models/evaluate.py                  # writes models/metrics.json (about 5 min)
python models/test_predictions.py          # read-only checks
streamlit run dashboard/app.py             # or: python run_dashboard.py
python dashboard/build_standalone_dashboard.py && cp dashboard/index.html dashboard/plotly.min.js public/
firebase deploy --only hosting             # needs access to the Firebase project
```

## Rules

- **Numbers.** `models/metrics.json` is the only source of performance numbers. Do not type metrics into docs, slides or UI; read them from that file. If you retrain, rerun `evaluate.py` and update every place that quotes a number.
- **Encoding.** All inference goes through `load_data()` (which reads the EHR with `keep_default_na=False`, so the insulin regimen "None" stays a string) and `encode_features()` with the fixed `CATEGORY_LEVELS`. Never hand-code category integers.
- **Wording.**
  - Say "synthetic data only; research prototype, not a medical device; not for dosing decisions".
  - Classifier outputs are uncalibrated risk scores, never "probability" or "confidence".
  - Do not claim regulatory compliance. The privacy wording is: "uses only synthetic data, so no personal health data is processed (designed with DPDP Act 2023 principles in mind)".
  - The simulator is illustrative and not calibrated to ICMR-INDIAB, NFHS-5 or any dataset.
- **Clinical wording.**
  - Hypoglycaemia: Level 1 is <70 mg/dL and Level 2 is <54 mg/dL (ADA). >250 mg/dL is "marked hyperglycaemia". Time in range is 70–180 mg/dL.
  - Show rescue text (15 g fast carbohydrate, recheck in 15 min) only when the CGM is <70 now. For a predicted low, say "consider a snack / review medication timing with the care team".
  - Never output insulin dose numbers as advice.
- **What-If sandbox.** It is rule-based and separate from the ML model. Label it that way. There are two different rule sets: the Streamlit one (`dashboard/app.py`) relaxes towards fasting glucose and scales carbohydrate by BMI, while the standalone one (`SANDBOX_RULES` in `dashboard/build_standalone_dashboard.py`) starts from the model's +2 h forecast and uses the sulfonylurea/insulin flag. Describe each separately; do not claim they agree.
- **Charts.** Only the Streamlit Live Monitor draws a dotted interpolation line to the +2 h forecast. The standalone page shows the single +2 h point with ± held-out MAE. Only CGM and the risk scores are charted over 8 h; HR, HRV, steps and sleep appear as current values and short summaries.
- **CV vs held-out.** Forecast MAE and sensitivity are close, but alert precision differs (hypo 37.1% CV vs 52.3% held-out, hyper 51.8% vs 36.4%). Do not say CV "agrees" on precision.
- **Simulator quirks.** These are known and are listed in the README Limitations: the basal/bolus slot collision, the short-night resistance landing one day late for early wakes, and the eGFR < 60 pathway never firing. Do not describe the kidney pathway as exercised.
- **Demo patients.** Use held-out patients (IND-PAT-196, 080, 077, 066, 046). No insulin-treated persona is held out (3 of 200 use insulin, all in training). Label any insulin example as a training patient.
- **Demo buttons.** Buttons that would act in a real system (export, notify) are labelled "(Demo)".
- **Team details.** Do not edit the team names, emails or phone number in README.md without the team's confirmation.
