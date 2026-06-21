# Battery Digital Twin + Warranty Life Prediction Dashboard

This Streamlit application upgrades the earlier diagnostic dashboard into a **warranty-cycle prediction dashboard**. It is designed for flooded tubular lead-acid batteries where early-cycle measurements must support a production QC decision: **will this battery survive 800, 1000, 1200 and 1500 cycles?**

## What changed in this version

### 1. Warranty Prediction module added
The app now includes a first tab called **Warranty Prediction**. It computes:

- Weibull characteristic life, `eta`
- Weibull shape factor, `beta`
- `P(>800 cycles)`
- `P(>1000 cycles)`
- `P(>1200 cycles)`
- `P(>1500 cycles)`
- PASS / REVIEW / HOLD decision
- early-life risk-driver chart

### 2. Early-cycle measurements are now explicitly used
The prediction uses early-cycle variables such as:

- capacity fade slope from early cycles
- Rct growth percentage from EIS/ECM
- SG drift per cycle
- voltage hysteresis
- formation temperature
- paste peak temperature
- spine Sb%
- packing density
- 4BS morphology proxy
- mud space
- PSoC hours
- water-loss risk
- corrosion, sulfation and shedding indices

### 3. 50-pack sample batch added
The included `sample_50_battery_pack_warranty_data.csv` contains 50 batteries from a demo batch. It intentionally includes good, borderline and bad early-life behaviour to make the warranty prediction logic easy to understand.

### 4. Test Description README tab added inside the dashboard
The dashboard now includes a **Test Description README** tab describing the data collection required across manufacturing, early cycling, EIS/ECM, accelerated life testing and end-of-life labelling.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Important technical note

The warranty model in this MVP is a transparent **Weibull survival surrogate**. It is not a calibrated production model. For production use, replace the surrogate with a trained survival model using real data:

```text
early-cycle measurements + manufacturing birth record + EIS/ECM features + actual failure cycle + censoring flag
```

Recommended model candidates:

- Weibull accelerated failure time model
- Cox proportional hazards model
- gradient-boosted survival model
- Bayesian survival model
- hybrid physics + ML survival model

## Production decision philosophy

The dashboard is built around this decision question:

> Given what I can measure in the first 20–50 cycles, what is the probability that this battery will survive the warranty cycle horizon?

The intended QC output is not only SOH, but:

- PASS: strong survival probability across warranty horizons
- REVIEW: borderline survival probability; needs engineering review or extended test
- HOLD: high risk of failing before warranty cycle target
