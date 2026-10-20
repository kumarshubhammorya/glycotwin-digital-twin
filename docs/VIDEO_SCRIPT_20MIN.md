# GlycoTwin video: recording script and demo plan

Happiest Health Digital Twin Challenge 2026, Phase 1. Team Classy Leopards, Goa Institute of Management, Sanquelim, Goa.

- **Required length:** at least 20 minutes. **Planned length:** about 28:11 at about 140 spoken words per minute, including about 3 minutes of clicks and typing; the spoken text alone is about 25:14. At 130 words per minute it runs to about 30:07; even rushed, at 160 words per minute, it is about 25:02. **Never stop the recording before 21:00 on the recorder's clock.**
- **Spoken text:** 3,532 words in the main script (only the `>` lines are spoken), plus 247 words of reserve passages for when you are running short (section 4).
- **Presenters:** Shivam Raj (team leader), Shubham Kumar, Shiva Kumar Bachatty, Yashita Dube.
- **What is shown:** the deck, a terminal and code editor, the Streamlit doctor dashboard (`dashboard/app.py`, running the trained model live), and the hosted standalone page [glycotwin-app.web.app](https://glycotwin-app.web.app).
- **Repository:** [github.com/kumarshubhammorya/glycotwin-digital-twin](https://github.com/kumarshubhammorya/glycotwin-digital-twin)

Rules for everyone who speaks:

1. Every performance number in this script is copied from `models/metrics.json` (section 6 lists the key for each one). Do not add, round up or "improve" any number while recording. If a number on screen ever differs from the script, read the screen and fix the script afterwards.
2. Values marked *expect* in the stage directions (CGM readings, risk scores, forecasts) are what the shipped model shows at that replay time. They are a check that you are on the right screen. They are not performance claims.
3. Words to use: "synthetic data", "research prototype, not a medical device", "risk score (uncalibrated)", "held-out patient", "marked hyperglycaemia (above 250)". Never call a risk score a "probability" or "confidence". Never say that the scores or the simulator are "calibrated" (only that they are not). Never call anything "clinical-grade", "compliant", "guaranteed" or "safe". Never say "diabetes capital", and never call our data "real patients".

---

## 1. Running order

| Time | Speaker | Segment | On screen |
| :--- | :--- | :--- | :--- |
| 0:00–3:39 | Shivam Raj | 1. Opening, the problem, why a digital twin | Deck: title and problem slides |
| 3:39–8:07 | Shubham Kumar | 2. Data streams, simulator, model, evaluation design | Terminal (starts `evaluate.py`), then deck: architecture diagram |
| 8:07–11:36 | Shubham Kumar | 3. Live code walkthrough | Code editor and terminal |
| 11:36–21:14 | Shiva Kumar Bachatty | 4. Live demo of the doctor dashboard | Streamlit app, then the hosted page |
| 21:14–26:18 | Yashita Dube | 5. Results, fusion evidence, limitations and safety | Streamlit Model Card |
| 26:18–28:11 | Shivam Raj | 6. Roadmap, impact and close | Deck: closing slide |

---

## 2. Pre-recording checklist

Do this on the machine that will record, on the day of recording. Allow 45 minutes, plus one full timed dry run.

### 2.1 Screen and machine

- [ ] Display at **1920 × 1080**, scaling 100%. Record the full screen at 1080p, 30 fps (OBS, or Zoom "Record to this computer").
- [ ] Browser zoom 100%. Hide the bookmarks bar. Close every other tab.
- [ ] Terminal font at least 18 pt; code editor zoomed so a line is readable on a phone (about 18 pt).
- [ ] Do Not Disturb on. Quit Slack, WhatsApp desktop, email and anything that pops up notifications.
- [ ] Each presenter's microphone tested; levels similar. Quiet room. If presenters are in different places, one person (Shiva) shares the screen for the whole recording and performs every click while the others speak.

### 2.2 Software and data

- [ ] All final changes are pushed to GitHub (README, decks, dashboards, `metrics.json`).
- [ ] **Folder A** (the main checkout, used for the dashboard and for showing code): `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
- [ ] **Folder B** (a fresh clone, used only to run `evaluate.py` live, so the committed `metrics.json` in folder A is never touched):
  ```bash
  git clone https://github.com/kumarshubhammorya/glycotwin-digital-twin.git ~/glycotwin-live
  cd ~/glycotwin-live && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
  ```
  Cloning from GitHub also proves the public repository is complete.
- [ ] In folder B, run `python models/test_predictions.py` once. It must end with `5 of 5 checks passed.` (about 7 seconds).
- [ ] In folder B, **time** `python models/evaluate.py` once (on a 4-core machine it took 5 min 48 s). Write the time down: Shubham starts it at the beginning of segment 2 and needs its output at 3D, about six and a half minutes later. If your run takes longer than that, use the fallback written under 3D. Afterwards run `git checkout -- models/metrics.json` in folder B so the on-camera `git diff` starts clean.
- [ ] In folder A, start the dashboard and leave it running: `streamlit run dashboard/app.py` (or `python run_dashboard.py`). Open `http://localhost:8501`.
- [ ] Warm the caches: click through all five views once (Cohort & Risk Board takes a few seconds the first time). Then **reload the browser tab (F5)**. A fresh tab must open on **Live Monitor**, patient **IND-PAT-196 · held-out (unseen) · Manoj Nair**, replay time **Day 7 · Mon 07 Sep 06:00**. Do not click anything else in that tab before segment 4.

### 2.3 Windows to have open, in this order

1. Deck in full-screen slideshow (PDF or PPTX).
2. Terminal in **folder B**, venv active, prompt clear.
3. Code editor on **folder A** with three tabs open: `models/train_fusion_model.py`, `models/metrics.json`, `dashboard/app.py`.
4. Browser tab 1: `http://localhost:8501` (fresh, see above). Browser tab 2: `https://glycotwin-app.web.app`.
5. A file-manager window showing `cgm_export_with_errors.csv` (section 2.4).

### 2.4 Prepare the broken file for the Data Ingest demo

- [ ] In the dashboard, select patient **IND-PAT-046**, open **Data Ingest**, click **Load sample: first 48 readings of IND-PAT-046**, then **Download this sample as CSV**. Then reload the tab (F5) so the demo starts fresh.
- [ ] Open the CSV in a text editor (not Excel, which rewrites the timestamps). Change the `glucose_cgm` value on **data row 20** (timestamp `2026-09-01 04:45:00`) to `900`. Delete the `glucose_cgm` value on **data row 30** (timestamp `2026-09-01 07:15:00`), leaving the field empty. Save as `cgm_export_with_errors.csv` on the Desktop.
- [ ] Check: uploading it must give **"Rejected: 2 of 5 checks failed"**, with 3. Physiological bounds FAIL, 4. Rate of change FAIL and 5. Missingness WARN. Reload the tab (F5) afterwards.

### 2.5 Check the hosted page

- [ ] `https://glycotwin-app.web.app` must show the **new** build: a black strip at the top reading "Synthetic data only. Research prototype, not a medical device.", a sidebar with **Live Monitor (replay)**, and the patient list starting with **IND-PAT-196 · Manoj Nair · held-out**.
- [ ] If it still shows the old site (patients named GT-…, "AUC 94.2%" or similar), redeploy first: `python dashboard/build_standalone_dashboard.py && cp dashboard/index.html dashboard/plotly.min.js public/ && firebase deploy --only hosting`. If you cannot deploy, open `dashboard/index.html` from folder A in the browser instead, and say "the page we host" rather than reading out the URL.
- [ ] Open it once in a private window to confirm it loads without being logged in.

### 2.6 Check the deck

- [ ] Use the regenerated deck. Before recording, page through every slide you will show and confirm none contains any of these: AUC 0.942 or "94.2%", 93.5% sensitivity, "0 missed nadirs", MAE 7.2, a Clarke figure without its source, "5:1", "sub-5 ms", "calibrated", "true probability", "clinical-grade", "DPDP/HIPAA compliant", "Full Regulatory Compliance", "100% medically safe", "diabetes capital", "calibrated to ICMR-INDIAB/NFHS-5". If any slide still does, do not show it; show the README or the Model Card instead.

### 2.7 Timed dry run

- [ ] Do one complete run-through with a stopwatch, performing every click. Write down the clock time at each segment start and compare with section 1.
- [ ] The dry run must reach the final "thank you" **after 21:30**. If it does not, plan to read the reserve passages in section 4 at the marked checkpoints.
- [ ] Reset everything after the dry run: reload the dashboard tab (F5), `git checkout -- models/metrics.json` in folder B, clear the terminal.

### 2.8 While recording

- [ ] Start recording, wait 2 seconds in silence on the title slide, then begin.
- [ ] Speak at a steady, unhurried pace. Pause one beat after each click so the screen settles.
- [ ] **Do not stop before 21:00 on the recorder's clock.** At the segment 5 handoff, check the clock against the checkpoint in section 4.
- [ ] If something breaks (an error, a frozen page), say what happened, reload, and continue. Do not restart the whole recording for a small slip.

---

## 3. Script

Format: lines starting with `>` are spoken. **[SCREEN]** says what is showing; **[DO]** is an action. *Expect* gives the values you should see, so you know you are on the right window.

### Segment 1: Opening and the problem (Shivam Raj)

#### 1A · 0:00–1:05 · Title and project overview

**[SCREEN]** Deck, title slide.

> Good day to the judges and to the Happiest Health team. Welcome to the prototype demonstration of GlycoTwin, a multimodal digital twin for Type 2 diabetes.
>
> GlycoTwin fuses a health record with continuous glucose monitor and wearable data, and forecasts two events two hours ahead: hypoglycaemia, a reading below 70 milligrams per decilitre, and marked hyperglycaemia, a reading above 250.
>
> One point applies to everything you will see. All our data is synthetic, generated by our own simulator. GlycoTwin is a research prototype, not a medical device, and not for dosing decisions. Every performance number in this video comes from one file, models slash metrics dot json, written by our evaluation script, which you will see run.

#### 1B · 1:05–2:37 · The problem, with sources

**[SCREEN]** Deck, problem slide (with its sources footer).

> Why India? The ICMR-INDIAB national study, published in the Lancet Diabetes and Endocrinology in 2023, estimated that in 2021 about 101 million people in India had diabetes, 11.4 percent of adults, and 136 million had prediabetes. That counts all diabetes found by testing, not only diagnosed Type 2, but the scale is clear.
>
> The phenotype differs too. In a large English cohort, South Asian adults with a BMI of about 24 had the same diabetes incidence as white adults with a BMI of 30, so weight alone is a weak guide. Genetics matters too: our twin includes the TCF7L2 risk allele, associated with Type 2 diabetes in Indian populations.
>
> The acute dangers happen between clinic visits. The American Diabetes Association defines Level 1 hypoglycaemia as glucose below 70 and Level 2 as below 54. Sulfonylureas and insulin are the main drug causes. At the other end, the international time-in-range consensus treats readings above 250 as level 2 time above range, and asks most adults to spend more than 70 percent of readings between 70 and 180.
>
> Yet care is episodic: a review every few months, a finger-stick now and then. Even with a CGM, someone has to notice a trend heading out of range in time to act.

#### 1C · 2:37–3:39 · Why a digital twin, and what we built

**[SCREEN]** Deck, the slide that summarises the solution (or stay on the problem slide).

> A CGM tells you where glucose is now. A digital twin combines that live stream with what we know about the person, their medicines, labs, history and genotype, and asks: what is likely to happen to this patient in the next two hours?
>
> That matches the three things the challenge asks for. Fusion of a static health record with a dynamic wearable stream. A working model that predicts a specific adverse event: for us, a low or a marked high within two hours, plus the glucose value two hours ahead. And a dashboard showing how a doctor works with the virtual patient.
>
> The intended use is earlier outreach, a phone call or a medication review, with the clinician deciding. Whether that reduces real episodes would need a prospective study, and we do not claim it. Shubham, over to you.

### Segment 2: Data, simulator, model and evaluation design (Shubham Kumar)

#### 2A · 3:39–4:14 · Start the evaluation

**[SCREEN]** Terminal in folder B. **[DO]** Type `python models/evaluate.py` and press Enter. Leave it running.

> Thank you, Shivam. This terminal is a fresh clone of our public repository, and I have just started our evaluation script. It scores the shipped model on patients it never saw, and retrains it many times for cross-validation and ablations, which takes about six minutes on a four-core laptop. So I will explain the design while it runs, and come back to its output.

#### 2B · 4:14–5:03 · The two data streams

**[SCREEN]** Switch to the deck: architecture diagram or data-streams slide.

> GlycoTwin has two synthetic data streams. Both come from one seeded script, data slash generate datasets dot py, so anyone can regenerate them exactly.
>
> The static record covers 200 personas: age, sex and diet; labs such as HbA1c, fasting glucose, blood pressure, eGFR and lipids; medication, from metformin alone up to a sulfonylurea or insulin; past diagnoses and hypo episodes in the last year; family history; and the TCF7L2 risk-allele count.
>
> The wearable stream records, every 15 minutes for seven days and for all 200 people, CGM glucose, heart rate, HRV, steps, sleep stage including REM, and logged carbohydrates and insulin. That makes 132,800 fifteen-minute windows.

#### 2C · 5:03–6:08 · How the simulator links the streams, honestly

> Why simulate, when the brief allows Synthea or MIMIC? We need both streams for the same person, aligned in time, with a known outcome every 15 minutes. Synthea has no 15-minute CGM, MIMIC is hospital data that cannot be redistributed, and we found no open Indian CGM dataset with a linked record.
>
> So our simulator links the streams on purpose. Sulfonylureas are taken even when breakfast is skipped, which causes lows. Insulin users sometimes miss a bolus or take too much. Low glucose raises heart rate and lowers HRV. Short sleep raises next-day insulin resistance. TCF7L2 risk alleles and a higher HbA1c raise post-meal peaks. Meal, walk and sleep times vary from day to day.
>
> Two caveats. The parameters are illustrative, loosely informed by Indian literature but not calibrated to any dataset. And because we wrote these links ourselves, the model finding them shows the pipeline works. It is not a medical discovery.

#### 2D · 6:08–7:04 · Labels and features

> Every window gets two labels. The event label is zero if every reading in the next two hours stays between 70 and 250, one if any reading drops below 70, and two if any goes above 250, with hypo taking priority. That gives 128,086 windows with no event, 1,012 hypo windows and 3,702 hyper windows. Events are rare, and that shapes everything that follows. The second label is the CGM reading exactly two hours ahead.
>
> The model sees 34 inputs. Eighteen come from the wearables: current glucose and its recent change, heart rate and HRV, steps, carbohydrates and insulin, sleep stage and hour of day. Sixteen come from the record, joined onto every window of that patient. The wearable features only look backwards, so nothing leaks in from the future.

#### 2E · 7:04–8:07 · The model and how we tested it

> The models are deliberately simple: scikit-learn HistGradientBoosting. A three-class classifier with balanced class weights, because events are rare, and a regressor for the two-hour glucose value. They suit mixed tabular data, train in under a minute on a laptop, and reproduce exactly with a fixed seed.
>
> How we test matters more than the model. We split by patient: 150 for training and 50 held out, never seen in training. The alert thresholds, 0.51 for hypo and 0.52 for hyper, were chosen by cross-validation inside the training patients only, maximising an F2 score, which weights sensitivity above precision. Then we score the 50 unseen patients, run five-fold cross-validation grouped by patient over all 200, compare with simple baselines, and remove groups of inputs to see what matters.
>
> The classifier outputs risk scores, not probabilities. Class weighting pushes them up on purpose, so we read them only against the thresholds.

### Segment 3: Live code walkthrough (Shubham Kumar)

The three editor views come first and the two terminal views last, so `evaluate.py` has about six and a half minutes to finish.

#### 3A · 8:07–9:00 · Where the design lives in the code

**[SCREEN]** Code editor, `models/train_fusion_model.py`. **[DO]** Scroll and point, in order, at: `WEARABLE_FEATURES` and `EHR_FEATURES`; `compute_wearable_features`; `split_patients`; `make_models` (`class_weight="balanced"`); `choose_thresholds` (`beta=2.0` in its signature, `GroupKFold`, `fbeta_score`); `encode_features` and `CATEGORY_LEVELS`.

> Here is that design in the code, in models slash train fusion model dot py. These are the wearable and EHR feature lists. compute wearable features builds every rolling feature per patient from past rows only. split patients does the patient-level split. make models sets the balanced class weights. And choose thresholds picks the alert thresholds by F2 score on out-of-fold predictions, grouped by patient, from training patients only.
>
> There is one encode features function with fixed category lists. Training, evaluation, the tests and both dashboards all use it, so a medication or a sleep stage is never encoded differently at prediction time.

#### 3B · 9:00–9:25 · metrics.json, the single source of numbers

**[SCREEN]** Code editor, `models/metrics.json`. **[DO]** Point at `holdout` → `regressor` → `mae`, then `baseline_persistence`, `classifier`, `episodes_hypo`, `rule_baseline_events`, `grouped_cv_5fold`, `ablation_grouped_cv_5fold`, `ehr_swap_test`.

> This is metrics dot json, as committed on GitHub: the held-out results and their baselines, sensitivity, precision, ROC-AUC and PR-AUC for each event, episodes, a simple rule baseline, cross-validation, ablations and an EHR-swap test. It is the only source of numbers in our README, our deck and this video.

#### 3C · 9:25–10:02 · How the dashboard loads the model

**[SCREEN]** Code editor, `dashboard/app.py`. **[DO]** Point at `get_model` (`joblib.load(MODEL_PATH)` and the checks below it), then `patient_predictions` (`encode_features`, `predict_proba`, `apply_alert_thresholds`, `regressor.predict`), then `get_metrics`.

> Next, dashboard slash app dot py. get model loads the trained bundle, glycotwin model dot joblib, and refuses to run if its keys or category lists do not match the training code. patient predictions calls the same helpers as training: encode, classify, apply the thresholds, then the regressor. get metrics reads metrics dot json, so the model card can never drift from the evaluation. If the bundle cannot load, the page shows an error, never the dataset's future labels.

#### 3D · 10:02–10:48 · The evaluation output and a reproducibility check

**[SCREEN]** Terminal in folder B. *Expect* the summary block ending in `Wrote models/metrics.json`, including `2 h forecast MAE 11.0 mg/dL (persistence 20.77), R2 0.803, Clarke A 91.81% / A+B 99.39%`.
If it is still running (you will see `Ablation …` lines appearing), open a second terminal tab in folder B with the venv active, do 3E there first (leave out its last sentence), then come back to this tab for 3D, and say "Shiva, over to you" at the end of 3D instead.

> The evaluation has finished. On the 50 held-out patients, the two-hour forecast error is 11.0 milligrams per decilitre, against 20.77 for simply assuming glucose stays the same. Yashita will interpret the rest. The latency line depends on the machine, so ignore it here.

**[DO]** Type `git diff --stat` and then `git diff models/metrics.json`. *Expect* `1 file changed, 3 insertions(+), 3 deletions(-)`, and only the three `latency` lines in the diff.

> Now a reproducibility check: git diff on metrics dot json, against the version committed on GitHub. The only lines that changed are the three latency figures. Every accuracy number reproduced exactly, from a fresh clone.

#### 3E · 10:48–11:36 · Prediction checks

**[SCREEN]** Terminal in folder B. **[DO]** Type `clear`, then `python models/test_predictions.py`. *Expect* five `[PASS]` lines and `5 of 5 checks passed.` after about 7 seconds.

> Last, our prediction checks, which run in seconds and write nothing. They confirm that the stored metrics match a fresh re-score; that held-out patient IND-PAT-196 gets a hypo alert before his first low, while his CGM is still in range; that IND-PAT-077 gets a hyper alert before his first reading above 250; that the stable windows of IND-PAT-046 get no alert; and that every category encodes correctly. Five of five pass, and you will meet all three patients in a minute. Shiva, over to you.

### Segment 4: Live demo of the doctor dashboard (Shiva Kumar Bachatty)

#### 4A · 11:36–14:49 · Live Monitor: a predicted low (IND-PAT-196)

**[SCREEN]** Browser tab 1, `http://localhost:8501`, fresh. *Expect* Live Monitor, IND-PAT-196 Manoj Nair, replay time "Day 7 · Mon 07 Sep 06:00", red pill "Model alert: hypoglycaemia risk within 2 h", current CGM 90.1, hypo score 0.85 against "alert ≥ 0.51", forecast 86 mg/dL at 08:00.

> Thank you, Shubham. This is the GlycoTwin doctor dashboard, running the trained model live on my laptop. The orange strip on every page says: synthetic data only, research prototype, not a medical device.
>
> On the left are the five views and the Patient 360 card. This is IND-PAT-196, Manoj Nair, a synthetic persona: 71, BMI 24.9, HbA1c 6.42, on metformin plus the sulfonylurea gliclazide. The teal chip says held-out: the model never saw him in training.
>
> The slider replays his seven-day record in 15-minute windows, and the model sees only data up to the selected time. We open at his latest alert: day 7, six in the morning. His CGM is 90.1, in range and steady. Yet the red card says hypoglycaemia risk within two hours: a risk score of 0.85 against a threshold of 0.51, labelled uncalibrated. The forecast for eight o'clock is 86. The alert covers any reading in the next two hours, the forecast one value at exactly two hours, so they can differ. For a predicted low, the suggestion is: consider a snack and review medication timing with the care team.
>
> Because he is held out, the chart also shows what actually happened, hidden from the model. The dashed line falls to about 50, and the check below says: alert was correct. The dotted line is just a link to the forecast, not a predicted path.

**[DO]** Click **+1 h**. *Expect* 07:00, CGM 79.7, "↘ -11.3 mg/dL per 15 min", heart rate 104, 1158 steps in 15 min.

> An hour later: 79.7 and falling fast. Heart rate 104, over a thousand steps in fifteen minutes. A morning walk, no breakfast logged, on a sulfonylurea.

**[DO]** Click **+1 h**. *Expect* 08:00, CGM 49.8, chip "Level 2 hypoglycaemia (<54)", red pill "Hypoglycaemia now (Level 2)" before the model-alert pill, card headline "Hypoglycaemia now (Level 2)" under "Current CGM reading · takes priority over the model", action text starting "CGM is 50 mg/dL now (Level 2 hypoglycaemia). Standard 15-15 rule …".

> Eight o'clock: 49.8, Level 2. Only because the CGM is below 70 now does the card show the rescue step: 15 grams of fast-acting carbohydrate, recheck in 15 minutes, inform the care team. GlycoTwin never suggests insulin doses.

**[DO]** Scroll down past the four wearable cards. Click **Nudge patient (Demo)** once (a toast says nothing was sent). Then open the expander **This week's model alerts for IND-PAT-196 (14)**.

> Buttons that would act in a real system are marked Demo and do nothing. Below are the risk score over the last eight hours and this week's alerts, with whether the event followed. Many rows say no: false alarm. Alarm burden is our main weakness, and Yashita will put numbers on it.

**[DO]** Scroll back to the top. Click **◀ Prev alert** three times. *Expect* "Day 5 · Sat 05 Sep 06:00", hypo score 0.92, retrospective check "next 2 h ranged 70–98 mg/dL … Verdict: false alarm."

> Here is one. Saturday, six in the morning, a high score, but his lowest reading in the next two hours was exactly 70, not below it. The low came fifteen minutes later. By our strict two-hour rule that is a false alarm, and we count it.

#### 4B · 14:49–16:36 · What-If Sandbox (rule-based)

**[DO]** Click **What-If Sandbox** in the sidebar. *Expect* the same replay time, "Day 5 · Sat 05 Sep 06:00" (the sandbox curve starts from his CGM there, 97.5); a red strip "ML model at this window (Sat 05 Sep 06:00): hypoglycaemia risk alert · hypo score 0.92 (alert ≥ 0.51) · hyper score 0.00 (alert ≥ 0.52) · forecast glucose at +2 h 88 mg/dL"; defaults "Extra meal carbohydrate (g)" 50, "Brisk walk after the meal (min)" 0, Horizon 4 h; "Time in 70–180 (simulated)" 82%; "Simulated peak" 192 mg/dL at +1 h; a dark diamond on the chart for "ML model forecast at +2 h". **[DO]** Scroll down a little until the note under the chart is in view. *Expect* a red note headed "ML model alert at this window (not a recommendation)", ending "Simulated peak 192 mg/dL is above the 180 mg/dL target at +1 h."

> The What-If sandbox says it in its title: rule-based, illustrative, not the ML model, not a dosing tool. It starts from the current CGM, uses only BMI and fasting glucose from the record, and ignores medication, so it can disagree with the model.
>
> That is why the red strip at the top shows what the ML model says for this same window: the hypo alert we just looked at, score 0.92, forecast 88. By default the sandbox adds a 50 gram meal and looks four hours ahead. The simulated peak is 192, above the 180 target, and the note under the chart puts the model's alert first.

**[DO]** Click the walk slider's handle and press the right-arrow key 6 times (30 min). *Expect* "Simulated peak" 172 mg/dL, "Time in 70–180 (simulated)" 100% in green, and the note still red and still headed "ML model alert at this window", now without the peak sentence.

> With a 30-minute walk, the simulated curve stays within 70 to 180 the whole time. But the note stays red. The model is still alerting at this window, and these rules know nothing about his sulfonylurea, so the page says it plainly: an in-range sandbox curve is not reassurance. We built it so that a comforting simulation can never hide a model alert.

**[DO]** Set the meal slider to 0 (click its handle, press Home). Then click the walk slider's handle again and press the right-arrow key 6 more times (60 min). *Expect* "Simulated lowest" 58 mg/dL with "Level 1 hypoglycaemia (54–69)", and the red note ending "Simulated lowest value 58 mg/dL (below 70) at +1h15. Review this scenario with the care team."

> With no meal but an hour's walk, the simulated lowest value is 58, below 70, and the note adds: review this scenario with the care team. For insulin users an illustrative insulin input appears too; it shows the shape of an effect, never a dose.

#### 4C · 16:36–18:01 · Live Monitor: marked hyperglycaemia (IND-PAT-077)

**[DO]** Click **Live Monitor**. Click the patient box in the sidebar, type `077`, press Enter. Drag the replay slider all the way to the left (Day 1 · Tue 01 Sep 00:00), then click **Next alert ▶**. *Expect* Day 1 11:00, CGM 176.9, amber pill "Model alert: marked hyperglycaemia risk within 2 h", hyper score 0.72, verdict "false alarm".

> Now hyperglycaemia. IND-PAT-077, Anand Rao: 70, BMI 33.5, HbA1c 7.12, metformin only, a high-carbohydrate North Indian diet, held out. The replay clock is shared, so I go back to day 1 and press Next alert. Eleven in the morning: CGM 177 and a hyper alert. The check says false alarm: he stayed below 250 for two hours.

**[DO]** Click **+1 h**. *Expect* 12:00, CGM 167.8 and falling, "0 g carbs, last 2 h", hyper score 0.95, forecast 243 mg/dL at 14:00, verdict "alert was correct".

> At noon he is at 168 and drifting down, no meal logged. Yet the hyper score is 0.95 and the two o'clock forecast is 243. This time the alert was correct: lunch came at one, and he went above 250. The model expected lunch partly from his profile and partly from the hour of day, because routines are regular in our simulator. Yashita will show how much it leans on that.

**[DO]** Click **+1 h** twice. *Expect* 14:00, CGM 264.0, amber pill "Marked hyperglycaemia now" before the model-alert pill, red chip "Marked hyperglycaemia (>250)", card headline "Marked hyperglycaemia now".

> At two o'clock he is at 264. The card says to follow his hyperglycaemia plan and review the meal, missed doses and hydration with the care team.

#### 4D · 18:01–18:34 · A stable patient (IND-PAT-046)

**[DO]** Patient box: type `046`, press Enter. *Expect* same time (Day 1 14:00), CGM 211.5 rising, amber chip "Above range (181–250)", green pill "No model alert", hyper score 0.01, verdict "correct: no event"; **◀ Prev alert** and **Next alert ▶** greyed out.

> For contrast, IND-PAT-046, Sunita Rao, 62, on lifestyle measures and low-dose metformin, also held out. At the same moment she is at 211 after lunch, above range and shown in amber, but there is no alert: the model expects her to stay below 250, and the check agrees. Her alert buttons are greyed out, because the model raised no alert for her all week.

#### 4E · 18:34–19:39 · Data Ingest and the five checks

**[DO]** Click **Data Ingest**. Click **Load sample: first 48 readings of IND-PAT-046**. *Expect* green "Accepted: sample: first 48 readings of IND-PAT-046 passed all 5 checks.", five PASS badges, and a green model card "No event predicted within 2 h".

> Data Ingest is how a clinic would bring in a device export: a generic CGM file, our full wearable format, or a LibreView-style file with a metadata line on top. Native vendor parsers are on the roadmap, not built. Here are twelve hours of this patient's data. Five checks run: schema, timestamp continuity, physiological bounds of 40 to 400, rate of change above 4 per minute, and missing values. All pass, and because this file has the full wearable format, the trained model scores its latest window: no event predicted.

**[DO]** Drag `cgm_export_with_errors.csv` from the file manager onto **Upload a wearable / CGM export**. *Expect* red "Rejected: 2 of 5 checks failed …", with 3. Physiological bounds FAIL, 4. Rate of change FAIL, 5. Missingness WARN.

> Now a file we broke on purpose: one glucose value of 900 and one blank. Rejected: bounds and rate of change fail, and the blank gives a warning. A rejected file is never scored.

#### 4F · 19:39–20:28 · Cohort and 2-hour risk board

**[DO]** Click **Cohort & Risk Board**. *Expect* tiles "Synthetic personas 200 · 200 with 7-day wearable streams · 50 held-out"; board time Tue 01 Sep 14:00; pills "0 hypoglycaemia-risk alerts" and "21 marked-hyperglycaemia-risk alerts". Switch on **Held-out (unseen) patients only**. *Expect* IND-PAT-077 at the top with "HYPER risk".

> The Cohort and Risk Board is the doctor's triage view: 200 personas, all with seven-day wearable streams, 50 held out. The risk board scores every patient's latest window at the shared replay time and sorts by risk score. With held-out patients only, Anand Rao is at the top.

**[DO]** In **Open a patient from the board**, choose **IND-PAT-066 · Vikram Verma · HYPER risk**, then click **Open in Live Monitor**. *Expect* Live Monitor, IND-PAT-066 at Day 1 14:00, CGM 233.7, hyper alert, verdict "false alarm".

> From the board I can open any patient. IND-PAT-066 opens at the same moment, with a hyper alert at 234, and the check says false alarm: high, but never above 250. That is alarm burden up close.

#### 4G · 20:28–21:14 · The hosted standalone page

**[DO]** Switch to browser tab 2, `https://glycotwin-app.web.app`. *Expect* black disclaimer strip; IND-PAT-196 at 2026-09-02 07:00 ("· day 2 of 7 · window …"); CGM 84.2 falling; "Hypo alert (next 2 h)"; hypo risk score 0.994; "What actually happened" box showing a reading below 70. **[DO]** Scroll to "7-day replay overview: tap a point to jump there" and click a red triangle.

> Last, the public page at glycotwin-app dot web dot app, which anyone can open, even on a phone. It does not run the model in the browser. It replays outputs that our Python pipeline computed when the page was built, for five held-out patients and one insulin-treated training patient, labelled as such. It opens on Manoj Nair half an hour before his first low on day 2, with the same "what actually happened" box, and a seven-day overview you can click to jump to any alert. Yashita, the numbers.

### Segment 5: Results, fusion evidence, limitations and safety (Yashita Dube)

#### 5A · 21:14–22:19 · The two-hour forecast against baselines

**[DO]** Switch to browser tab 1. Click **Model Card**. *Expect* tiles "+2 h forecast MAE (held-out) 11.0 mg/dL · persistence 20.8 · linear 35.2" and "Clarke zone A (held-out) 91.8% · A+B 99.4% · D 0.6%".

> Thank you, Shiva. The Model Card reads everything from metrics dot json. Held-out means the 50 patients the model never saw, 33,200 windows. And it is all synthetic, so these numbers describe how well the model learns our simulator, not how it would do on real patients.
>
> The two-hour forecast error is 11.0 milligrams per decilitre. Assuming glucose stays the same gives 20.8, and extending the last 15-minute trend gives 35.2. So the model roughly halves the error of the no-change baseline, with an R-squared of 0.803. On the Clarke error grid, 91.8 percent of forecasts fall in zone A and 7.6 percent in zone B, so 99.4 percent combined, from the same file. But zone D is 0.6 percent, not zero, and the no-change baseline already puts 79.1 percent in zone A, so A plus B alone does not separate models well.

#### 5B · 22:19–23:57 · Alerts, episodes and the alarm burden

**[DO]** Scroll to "2-h event alerts: held-out patients", then to "Episodes and alarm burden: held-out patients".

> Now the alerts, judged on every 15-minute window. For hypoglycaemia, sensitivity is 73.4 percent at 52.3 percent precision, with ROC-AUC 0.990 and PR-AUC 0.621. For marked hyperglycaemia, sensitivity is 86.2 percent at 36.4 percent precision, ROC-AUC 0.988 and PR-AUC 0.774. Events are only 1.4 and 1.7 percent of windows, so precision and PR-AUC say more than ROC-AUC. A simple rule, CGM below 90 and falling or above 200 and rising, catches only 42.8 and 40.8 percent. And when the patient is not already out of range, sensitivity drops to 63.2 percent for hypo and 77.9 for hyper.
>
> By episode, 34 of 39 hypo episodes, 87 percent, were warned 15 to 120 minutes before the first low, and 42 of 43 hyper episodes. The cost: about 0.36 hypo and 0.58 hyper alarms per patient per day, and 69 percent of hypo alarms and 88 percent of hyper alarms were not followed by the event. In a clinic that means alarm fatigue. Our thresholds favour sensitivity, and tuning that trade-off with clinicians is the first thing we would change.
>
> Cross-validation over all 200 patients gives a similar forecast error, 11.08, and similar sensitivity: 75.5 percent for hypo and 82.5 for hyper. Precision does not match: 37.1 percent for hypo and 51.8 for hyper, so precision depends on which patients are tested.

#### 5C · 23:57–25:02 · Does fusing the EHR help?

**[DO]** Scroll to "Ablation: which inputs matter" and "EHR-swap test".

> Does fusing the health record help? First, the ablation, retraining with groups of inputs removed. With wearables only, the forecast error rises from 11.08 to 11.45, and hypo PR-AUC falls from 0.548 to 0.530: the record improves the forecast and hypo detection, modestly. For hyper, the wearables alone rank risk slightly better, 0.855 against 0.820. The record alone, with hour of day, gives an error of 15.91. Without the CGM features it is 14.55, and without hour of day 13.86, the daily routine Shiva showed. Without heart rate, HRV and sleep it is 11.07: they add little independent signal here.
>
> Second, the EHR swap: we score each held-out patient with someone else's record. The error rises from 11.0 to 16.13, and hypo PR-AUC falls from 0.621 to 0.435. Give the twin the wrong record and it gets worse, so it really uses the record.

#### 5D · 25:02–26:18 · Limitations and safety

**[DO]** Scroll to "Data and runtime" and "Limitations".

> The limitations are on this page too. Synthetic data only: the high ROC-AUC values partly reflect a simulator far simpler than real physiology. The simulator is not calibrated, and the cohort is mostly well controlled. The scores are uncalibrated, and most alarms are false. Only 3 of 200 personas use insulin, none held out, so insulin users are untested; no persona has an eGFR below 60; and only 35 readings fall below 54, too few to test Level 2 separately. It is a single two-hour point forecast, and the sandbox is rule-based. Speed is not the problem: about 15 milliseconds per window on a CPU, on one machine.
>
> On safety: GlycoTwin is not a medical device and has had no clinical validation. Clinical use would need prospective validation on real data, clinicians in the loop, and regulatory review as software as a medical device. It uses only synthetic data, so no personal health data is processed, and we designed it with India's Digital Personal Data Protection Act of 2023 in mind. Shivam will close.

### Segment 6: Roadmap, impact and close (Shivam Raj)

#### 6A · 26:18–27:01 · Roadmap

**[SCREEN]** Deck, roadmap or closing slide.

> Thank you, Yashita. Our next steps. First, real data: test on open Type 2 diabetes CGM datasets such as ShanghaiT2DM or CGMacros, after checking their licences, and build the record side with Synthea. Second, alarm burden: set thresholds by how many alarms a care team can act on, and calibrate the risk scores. Third, forecast the whole two-hour path with its uncertainty, and explain each alert. Fourth, native import of Dexcom, LibreView, Apple Health and Health Connect exports. And fifth, a consented pilot with an Indian diabetes clinic, and only then a regulatory pathway.

#### 6B · 27:01–28:11 · Impact and close

**[SCREEN]** Deck, closing slide, repository and hosted page (and the video link once it exists).

> Our aim is not a new device. It is a way for a diabetologist or diabetes educator who looks after hundreds of CGM users to see, each morning, which few patients need a call before their glucose leaves range, and why. Whether GlycoTwin achieves that is a question for a clinical study, and we built it so that such a study is possible.
>
> We have submitted a public, MIT-licensed repository: a seeded simulator fusing a 200-person health record with a week of CGM and wearable data; a twin forecasting hypo- and hyperglycaemia two hours ahead, tested on 50 unseen patients against baselines; a live doctor dashboard and a hosted page; and an evaluation anyone can re-run with one command.
>
> Thank you for your time. We look forward to your questions.

**[DO]** Hold the closing slide for 5 seconds in silence, then stop recording. Check the recorder's clock first: if it shows less than 21:00, do not stop. Read reserve passage R2 (section 4), then repeat the thank-you line.

---

## 4. Reserve passages (read only if you are running short)

**Checkpoint:** when Yashita starts segment 5, the recorder should show at least **16:30**. If it shows less, read **R1** at the end of segment 5C (before 5D). If the final thank-you still arrives before 21:30, read **R2** before the thank-you. These passages are not counted in the main word count.

#### R1 · Yashita Dube · after 5C · about 1 minute · Why not a bigger model?

> A question we expect from the technical panel: why gradient-boosted trees and not a deep sequence model? Three reasons. Our inputs are tabular, a mix of numbers and categories from two sources, which is where boosted trees are strong. Our events are rare, so a large network would have very few examples to learn from. And for a clinical prototype we wanted something that trains in under a minute, gives identical results from the same seed, and can be re-checked by anyone with a laptop. We did not try a deep model, so we make no claim that it would do worse. With real CGM data and many more patients, a sequence model that forecasts the whole two-hour path would be a natural next experiment, and the evaluation script would compare it against exactly the same baselines.

#### R2 · Shiva Kumar Bachatty · before the thank-you · about 1 minute · A second sulfonylurea patient

**[DO]** Switch to browser tab 1, Live Monitor, patient box `080`, then click **Latest** and **◀ Prev alert** a few times.

> One more example while we are here. This is IND-PAT-080, Priyanka Sharma, 46, also on gliclazide, also held out, and with a BMI of only 20.7: in our simulator, as in the clinic, hypoglycaemia risk follows medication and routine, not weight. Like Manoj Nair, she gets an alert most mornings around six, and on several evenings. As I step back through her week, you can see alerts that were followed by a low and alerts that were not, each marked in the retrospective check. That mix of hits and false alarms, repeated across all 50 held-out patients, is exactly what the Model Card summarises. Back to Shivam for the close.

---

## 5. After recording

- [ ] Watch the whole recording once. Trim only dead air at the start and end. Confirm the final file is **longer than 21:00** (the plan above runs about 28 minutes, or about 25 if rushed).
- [ ] If segments were recorded separately, join them and check the **joined** duration.
- [ ] Upload to YouTube as **Unlisted**. Title: `GlycoTwin: multimodal digital twin for Type 2 diabetes | Team Classy Leopards, Goa Institute of Management | Happiest Health Digital Twin Challenge 2026`. Description: one paragraph, the repository link, and the line "Synthetic data only. Research prototype, not a medical device. Not for diagnosis or dosing decisions."
- [ ] Wait for processing to finish, then open the link in a **logged-out (private) browser window**. Confirm it plays, the duration shown is over 20:00, and the audio is clear.
- [ ] Paste the link into README section "5. Video (minimum 20 minutes)", replacing the line `**VIDEO LINK: paste the unlisted YouTube URL here before submitting**`.
- [ ] Do not edit the deck by hand. In **folder A** (its `metrics.json` is the committed one), run `python generate_summit_docs.py` (needs `pip install -r requirements-dev.txt` and LibreOffice). It reads the video link from README section 5 and regenerates both `docs/GlycoTwin_Summit_Presentation.pptx` and its PDF, so the link lands on the **closing slide** of both. Paste the full URL (`https://youtu.be/…` or `https://www.youtube.com/watch?v=…`; a `https://drive.google.com/…` link also works), because the script only recognises those forms. The run must **not** print `WARNING: no YouTube or Google Drive link found`; open the PDF and check that the closing slide shows the link.
- [ ] In folder B, discard the latency change (`git checkout -- models/metrics.json`). Folder A's `metrics.json` was never touched.
- [ ] Commit and push the README together with everything the generator rewrote in `docs/` (deck PPTX and PDF; the architecture diagram files are rebuilt too). Open the README on GitHub in a logged-out window and click the video link once more.

---

## 6. Where every number comes from

All keys are in `models/metrics.json`. "Held-out" = `holdout`, "CV" = `grouped_cv_5fold`, "ablation" = `ablation_grouped_cv_5fold`.

| Said in the video | Value | Key |
| :--- | :--- | :--- |
| Patients, windows | 200; 132,800 | `data.patients`, `data.windows` |
| Inputs | 34 = 18 wearable + 16 EHR | `data.features` |
| Window labels: none / hypo / hyper | 128,086 / 1,012 / 3,702 | `data.label_counts` |
| CGM readings below 54 | 35 | `data.cgm_readings_lt54` |
| Held-out patients, windows | 50; 33,200 | `holdout.test_patients`, `holdout.windows` |
| Alert thresholds | 0.51 hypo, 0.52 hyper | `holdout.alert_thresholds` |
| Forecast MAE: model / persistence / linear | 11.0 / 20.77 (20.8) / 35.2 mg/dL | `holdout.regressor.mae`, `holdout.baseline_persistence.mae`, `holdout.baseline_linear_extrapolation.mae` |
| R² | 0.803 | `holdout.regressor.r2` |
| Clarke A, B, D; A+B | 91.81, 7.58, 0.61; 99.39 (sum of A and B) | `holdout.regressor.clarke_pct` |
| Persistence Clarke A | 79.08 | `holdout.baseline_persistence.clarke_pct.A` |
| Hypo: sensitivity, positives, precision, ROC-AUC, PR-AUC, prevalence | 73.4% (336 of 458), 52.3%, 0.990, 0.621, 1.4% | `holdout.classifier.hypo_lt70` |
| Hyper: sensitivity, precision, ROC-AUC, PR-AUC, prevalence | 86.2%, 36.4%, 0.988, 0.774, 1.7% | `holdout.classifier.hyper_gt250` |
| Rule baseline sensitivity: hypo / hyper | 42.8% / 40.8% | `holdout.rule_baseline_events` |
| Sensitivity when CGM is 70–250 now: hypo / hyper | 63.2% / 77.9% | `holdout.classifier_not_already_in_event` |
| Hypo episodes warned 15–120 min ahead | 34 of 39 (87%); median lead 120 min | `holdout.episodes_hypo` |
| Hyper episodes warned | 42 of 43 | `holdout.episodes_hyper` |
| Alarms per patient-day: hypo / hyper | 0.36 / 0.58 | `…episodes_*.alarms_per_patient_day` |
| Alarms not followed by the event: hypo / hyper | 69% / 88% | `…episodes_*.false_alarm_fraction` |
| CV: MAE; hypo sensitivity / precision; hyper sensitivity / precision | 11.08; 75.5% / 37.1%; 82.5% / 51.8% | `grouped_cv_5fold.regressor`, `grouped_cv_5fold.classifier` |
| Ablation MAE: all / wearables only / EHR + hour / no CGM / no HR-HRV-sleep / no hour | 11.08 / 11.45 / 15.91 / 14.55 / 11.07 / 13.86 | `ablation_grouped_cv_5fold.*.regressor_mae` |
| Ablation PR-AUC: hypo all vs wearables only; hyper all vs wearables only | 0.548 vs 0.530; 0.820 vs 0.855 | `ablation_grouped_cv_5fold.*.hypo_pr_auc`, `…hyper_pr_auc` |
| EHR swap: MAE; hypo PR-AUC | 11.0 → 16.13; 0.621 → 0.435 | `holdout.ehr_swap_test` |
| Latency, one window | about 15 ms (median 14.59), CPU only | `latency.single_window_ms_median` |

Not from `metrics.json`, and not performance claims:

- Population and clinical facts in segment 1 are from the README references: ICMR-INDIAB-17 (Anjana et al., *Lancet Diabetes Endocrinol* 2023), Caleyachetty et al. 2021, ADA Standards of Care 2025 section 6, Battelino et al. 2019, and Chandak et al. 2007 for TCF7L2.
- "3 of 200 personas use insulin, none held out" is counted from `data/synthetic_ehr.csv` and the bundle's held-out list; the Model Card's "Data and runtime" table shows it live.
- Patient details and on-screen values in segment 4 are read off the dashboards at the stated replay times.
