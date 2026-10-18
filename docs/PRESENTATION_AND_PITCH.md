# GlycoTwin: Presentation Deck & 2-5 Minute Video Pitch Script

> **Summit:** Happiest Health Reimagining and Reforming Healthcare in India Summit 2026, Bengaluru  
> **Target Audience:** Technical Evaluation Committee, Medical Experts, Summit Jury  
> **Format:** Slide Outline + Verbatim Speaking Script for 2–5 Minute Video

---

## Part 1: Presentation Slide Deck Outline

### Slide 1: Title & Team
* **Title:** GlycoTwin: A Multimodal Digital Twin for Proactive Type 2 Diabetes Management
* **Sub-caption:** 2-Hour Predictive Adverse Event Horizon fusing EHR & Wearable IoT Streams
* **Team:** [Your Team Name] | [Your College / Incubator Name]
* **Event:** Happiest Health Summit 2026, Bengaluru

### Slide 2: The Healthcare Crisis in India
* **The Context:** India is the diabetes capital of the world with $>101$ million diagnosed individuals.
* **The Pain Point:** Current diabetes care is *reactive*. Patients only discover dangerous hypoglycemia or extreme sugar spikes when symptoms manifest—often leading to emergency room visits, falls, or diabetic ketoacidosis.
* **The Opportunity:** Wearables (CGMs, smartwatches) combined with clinical records can turn reactive treatment into **predictive prevention**.

### Slide 3: The Digital Twin Paradigm
* **What is a Digital Twin in Diabetes?** A continuous virtual counterpart of the patient's metabolic system.
* **Multimodal Fusion:**
  * *Static Stream (EHR):* Age, BMI, baseline HbA1c, renal function (eGFR), and medication regimen.
  * *Dynamic Stream (IoT):* 15-minute continuous glucose, heart rate, HRV stress biomarkers, steps, and meal inputs.
* **Core Breakthrough:** Forecasting adverse glycemic events **2 hours before they occur**.

### Slide 4: System Architecture & AI Pipeline
* **Ingestion:** FHIR-compatible EHR schema + Wearable stream processor.
* **Feature Engineering:** Rate of glucose change ($dG/dt$), 1h/2h rolling metabolic exertion, autonomic HRV suppression.
* **AI Engine:** Gradient Boosted Multimodal Classifiers and Regressors trained with class-balancing for rare medical events.
* **Validation:** Evaluating sensitivity for critical hypoglycemia and precision for postprandial spikes.

### Slide 5: The Interactive Clinician Dashboard (Live Demo)
* **Patient 360° Profile:** Comprehensive clinical card for Indian cohorts.
* **2-Hour Predictive Alert Horizon:** Red/Amber/Green alerts with actionable medical rescue protocols (e.g. "Consume 15g fast carbs now").
* **"What-If" Counterfactual Simulation:** Allowing doctors and patients to simulate meals, walks, and insulin doses before doing them in real life.
* **Real-World CSV Ingestion:** Ready for Abbott LibreView, Dexcom Clarity, and Apple Health exports.

### Slide 6: Clinical Impact, DPDP Compliance & Future Roadmap
* **Healthcare Impact:** Reduces nocturnal hypoglycemia emergencies, optimizes HbA1c, lowers hospitalization costs.
* **Data Privacy:** Synthetic data modeled under India's DPDP Act 2023.
* **Next Horizon:** Closed-loop automated insulin delivery and multi-organ twin expansion (Cardio-Renal-Metabolic Twin).

---

## Part 2: Verbatim Video Pitch Script (2 to 4 Minutes)

*(Tip: You can read this script aloud while recording your screen showing the PowerPoint slides and the interactive web dashboard!)*

---

### [0:00 - 0:45] Introduction & The Problem in India
> "Good day, esteemed members of the jury and healthcare leaders at the Happiest Health Summit 2026.
>
> India is facing an unprecedented chronic disease epidemic. With over 101 million people living with Type 2 Diabetes, our current healthcare delivery model is fundamentally **reactive**. Patients measure their blood sugar only after feeling dizzy, or arrive at hospitals after severe hypoglycemic shock or uncontrolled spikes.
>
> But what if a physician could foresee a life-threatening glucose crash **two hours before it happens**?
>
> Welcome to **GlycoTwin**—a personalized, multimodal Digital Twin proof-of-concept that turns reactive diabetes management into proactive, predictive care."

---

### [0:45 - 1:30] The Technology & Multimodal Fusion
> "To build a faithful Digital Twin, we cannot rely on wearables alone, nor can we rely on static hospital charts alone. 
>
> GlycoTwin demonstrates the true fusion of **two critical data streams**:
> 
> First, the **Static Clinical EHR Stream**: Capturing patient demographics, baseline HbA1c, kidney eGFR, and pharmacotherapy—whether they are on Metformin, Sulfonylureas, or Basal Insulin.
>
> Second, the **Dynamic Real-Time Wearable Stream**: Streaming continuous glucose monitor readings every 15 minutes, coupled with heart rate, heart rate variability, accelerometer steps, and meal inputs.
>
> Our machine learning engine aligns these streams in time, computing physiological derivatives like glucose rate of change and metabolic exertion. It then predicts two critical horizons:
> 1. Exact glucose trajectory 120 minutes into the future.
> 2. Automated classification of imminent adverse events—specifically flagging critical hypoglycemia under 70 mg/dL and severe hyperglycemia over 250 mg/dL."

---

### [1:30 - 3:00] Live Dashboard Demonstration
*(Switch screen to the running Streamlit web dashboard)*

> "Let us look at GlycoTwin in action on our clinician dashboard.
>
> Here in the sidebar, we can select from our cohort of 200 Indian patients. Notice how their complete clinical baseline—from diet profile to insulin regimen—is loaded into the twin.
>
> On the main screen, we see live continuous telemetry. As we scrub through time, notice the top metrics: real-time glucose, heart rate, and activity.
>
> Here is the magic: Look at this alert banner. The model detects a rapid downward velocity combined with recent physical activity and prior insulin administration. It sounds an early alarm: **'Critical Adverse Event: Impending Hypoglycemia in 90 minutes with projected glucose of 56 mg/dL.'** 
>
> More importantly, it gives the patient a rescue protocol: *'Consume 15 to 20 grams of fast-acting carbs and halt strenuous exercise.'* This prevents an emergency room visit before symptoms even begin.
>
> Now let’s look at the **'What-If' Simulation tab**. This is the core superpower of a Digital Twin. A doctor can test counterfactual scenarios: *'What happens if the patient eats a 60-gram carbohydrate dinner and takes a 30-minute walk?'* The virtual twin simulates the biological response curve, allowing personalized precision medicine without any risk to the patient.
>
> Furthermore, in our third tab, we provide seamless drag-and-drop ingestion for real-world exports from Abbott FreeStyle Libre and Dexcom devices."

---

### [3:00 - 3:45] Clinical Relevance, Ethics & Conclusion
> "In accordance with India's Digital Personal Data Protection Act and global HIPAA standards, our prototype was developed using rigorous synthetic modeling reflecting South Asian metabolic phenotypes.
>
> GlycoTwin proves that digital health in India can move from episodic check-ups to continuous, predictive care. We look forward to presenting this vision in person at the Happiest Health Summit in Bengaluru.
>
> Thank you!"
