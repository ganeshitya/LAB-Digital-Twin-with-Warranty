"""
Advanced Battery Digital Twin & Warranty Life Prediction Dashboard
Flooded tubular lead-acid batteries | Streamlit MVP

Run:
    streamlit run app.py

Purpose:
    Converts early-cycle measurements and manufacturing process variables into
    warranty-horizon survival probabilities for 800 / 1000 / 1200 / 1500 cycles.

Important:
    This MVP uses a transparent Weibull-survival surrogate. For production,
    replace the surrogate with a calibrated survival model trained on:
    early-cycle features + manufacturing birth records + EIS/ECM features +
    actual failure-cycle labels + censoring flags.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# 0. Streamlit configuration and visual theme
# =============================================================================

st.set_page_config(
    page_title="Battery Digital Twin + Warranty",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

CYAN = "#00C8FF"
GREEN = "#00E5A0"
ORANGE = "#FF6B2B"
PURPLE = "#C084FC"
YELLOW = "#FFD700"
BG = "#071422"
CARD = "#0E2233"
TEXT = "#D7EAF5"
MUTED = "#7A9BB0"

st.markdown(
    f"""
    <style>
    .stApp {{background:{BG}; color:{TEXT};}}
    section[data-testid="stSidebar"] {{background:#091B2A;}}
    div[data-testid="stMetric"] {{
        background:{CARD};
        border:1px solid rgba(0,200,255,.18);
        padding:14px;
        border-radius:14px;
    }}
    .block-container {{padding-top:1.2rem;}}
    h1,h2,h3 {{color:{TEXT};}}
    .small-note {{color:{MUTED}; font-size:.9rem;}}
    .pass {{color:{GREEN}; font-weight:800;}}
    .review {{color:{YELLOW}; font-weight:800;}}
    .hold {{color:{ORANGE}; font-weight:800;}}
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# 1. Helper functions: Streamlit-safe chart rendering
# =============================================================================

# Streamlit 1.4x+ can throw StreamlitDuplicateElementId when the same figure is
# rendered in multiple tabs without a unique key. These wrappers enforce keys.

def chart_key(name: str, row=None) -> str:
    """Create stable unique keys for Streamlit elements."""
    try:
        battery_id = str(row.battery_id) if row is not None else "fleet"
    except Exception:
        battery_id = "fleet"
    return f"{name}_{battery_id}"


def show_chart(fig, key: str):
    """Render Plotly chart with mandatory unique key."""
    st.plotly_chart(fig, use_container_width=True, key=key)


def show_column_chart(column, fig, key: str):
    """Render Plotly chart inside a Streamlit column with mandatory unique key."""
    column.plotly_chart(fig, use_container_width=True, key=key)


def style_layout(fig):
    """Apply common dark theme to Plotly figures."""
    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font_color=TEXT,
        legend_title_text="",
    )
    return fig

# =============================================================================
# 2. Physics / surrogate degradation feature layer
# =============================================================================

def npclip(x, a, b):
    return np.clip(x, a, b)


def soc_from_sg(sg):
    """Approximate SG-to-SOC conversion for flooded LAB demo use."""
    return npclip((sg - 1.120) / (1.280 - 1.120) * 100, 0, 100)


def corrosion_index(sb, formation_temp, cycles):
    """Positive grid corrosion score from Sb%, formation temperature and cycles."""
    return npclip(
        100
        * (0.75 + 0.23 * sb)
        * np.exp(np.maximum(formation_temp - 45, 0) / 22)
        * np.sqrt(np.maximum(cycles, 1))
        / np.sqrt(500)
        / 2.8
        * 0.62,
        0,
        100,
    )


def sulfation_index(paste_temp, psoc_hours, cycles):
    """NAM sulfation score from paste temperature, PSoC exposure and cycles."""
    return npclip(
        np.maximum(paste_temp - 48, 0) * 4.5
        + np.log1p(psoc_hours) * 7.5
        + np.sqrt(np.maximum(cycles, 1)) * 0.8,
        0,
        100,
    )


def water_loss_pct(sb, float_v, cycles, ambient):
    """Electrolyte water-loss risk from Sb%, float voltage, temperature and cycles."""
    return npclip(
        sb * 1.7
        + np.maximum(float_v - 13.55, 0) * 20
        + np.maximum(ambient - 30, 0) * 0.45
        + cycles * 0.018,
        0,
        100,
    )


def shedding_risk(density, four_bs, mud, cycles):
    """Active material shedding risk from density, 4BS morphology and mud space."""
    return npclip(
        np.abs(density - 4.0) * 42
        + np.maximum(70 - four_bs, 0) * 0.75
        + np.maximum(18 - mud, 0) * 3.2
        + cycles * 0.015,
        0,
        100,
    )


def failure_probability(corr, sulf, water, shed, soh):
    """Short-term diagnostic probability from current risk indicators."""
    z = (
        -4.2
        + 0.022 * corr
        + 0.026 * sulf
        + 0.024 * water
        + 0.021 * shed
        + 0.045 * np.maximum(80 - soh, 0)
    )
    return 1 / (1 + np.exp(-z))


def dominant_mode(row):
    """Select dominant failure mode by maximum degradation score."""
    scores = {
        "Grid corrosion": row.corrosion_index,
        "Sulfation": row.sulfation_index,
        "Water loss": row.water_loss_pct,
        "Shedding": row.shedding_risk,
    }
    return max(scores, key=scores.get)

# =============================================================================
# 3. Warranty survival model layer
# =============================================================================

def weibull_survival(cycles, eta, beta):
    """Weibull survival probability S(N)=exp(-(N/eta)^beta)."""
    cycles = np.asarray(cycles, dtype=float)
    return np.exp(
        -np.power(
            np.maximum(cycles, 0) / np.maximum(eta, 1e-6),
            np.maximum(beta, 0.3),
        )
    )


def warranty_life_params(df):
    """
    Early-cycle feature -> Weibull eta/beta surrogate.

    Production replacement:
        Train Weibull AFT / Cox / gradient-boosted survival / Bayesian survival
        model on early-cycle data + actual failure cycle labels + censoring flags.
    """

    # Early stress score: higher means lower characteristic life.
    # These variables are measurable in first few cycles / formation / EIS.
    stress = (
        0.20 * df["corrosion_index"]
        + 0.18 * df["sulfation_index"]
        + 0.15 * df["water_loss_pct"]
        + 0.13 * df["shedding_risk"]
        + 0.22 * np.maximum(df["formation_temp_c"] - 55, 0) * 5
        + 0.16 * np.maximum(df["rct_growth_pct_early"], 0)
        + 0.13 * np.maximum(df["capacity_fade_slope_pct_per_cycle"] * 100, 0)
        + 0.10 * np.maximum(df["sg_drift_per_cycle"] * 10000, 0)
        + 0.08 * np.maximum(df["voltage_hysteresis_mv"], 0) / 10
    )
    stress = npclip(stress, 0, 160)

    # Characteristic life eta: healthy early behaviour shifts life towards 1500+.
    eta = 1700 - 6.2 * stress + 4.5 * (df["soh_pct"] - 85) - 1.2 * np.maximum(df["cycles"] - 50, 0)
    eta = npclip(eta, 350, 2200)

    # Shape beta: >1 indicates wear-out dominated behaviour.
    mode_factor = df["dominant_failure_mode"].map(
        {"Grid corrosion": 1.2, "Sulfation": 1.0, "Water loss": 0.9, "Shedding": 1.35}
    ).fillna(1.0)
    beta = 1.45 + 0.018 * stress + 0.18 * mode_factor
    beta = npclip(beta, 1.2, 4.8)

    return eta, beta, stress


def base_decision(p800, p1200, p1500):
    """Default warranty decision gates. Sidebar allows custom gates."""
    if p800 < 0.90 or p1200 < 0.75 or p1500 < 0.55:
        return "HOLD"
    if p800 < 0.95 or p1200 < 0.85 or p1500 < 0.65:
        return "REVIEW"
    return "PASS"

# =============================================================================
# 4. Data enrichment and demo data generation
# =============================================================================

def enrich(df):
    """Add derived SOC/SOH/risk/survival fields to uploaded or demo data."""
    defaults = {
        "battery_id": "UP-000001",
        "batch_id": "BATCH-UP",
        "variant": "12V-150Ah",
        "cycles": 25,
        "sg": 1.245,
        "voltage_v": 12.55,
        "current_a": 18,
        "ambient_temp_c": 33,
        "formation_temp_c": 49,
        "tank_temp_c": 47,
        "paste_peak_temp_c": 46.5,
        "spine_sb_pct": 3.0,
        "packing_density_gcc": 4.0,
        "four_bs_pct": 73,
        "mud_space_mm": 20,
        "formation_ah_input": 112,
        "float_voltage_v": 13.6,
        "psoc_hours": 48,
        "rct_mohm": 7,
        "r0_mohm": 3.2,
        "warburg_z": 1.8,
        "nominal_ah": 150,
        "capacity_0_ah": 150,
        "capacity_10_ah": 147.5,
        "capacity_25_ah": 144.0,
        "rct_0_mohm": 6.2,
        "rct_25_mohm": 7.0,
        "sg_0": 1.255,
        "sg_25": 1.245,
        "voltage_hysteresis_mv": 45,
    }

    df = df.copy()
    n = len(df)

    for col, value in defaults.items():
        if col not in df:
            if col == "battery_id":
                df[col] = [f"UP-{i + 1:06d}" for i in range(n)]
            else:
                df[col] = value

    # Type coercion
    text_cols = ["battery_id", "batch_id", "variant"]
    for col in df.columns:
        if col not in text_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(defaults.get(col, 0))

    # Diagnostic features
    df["soc_pct"] = soc_from_sg(df["sg"])
    df["corrosion_index"] = corrosion_index(df["spine_sb_pct"], df["formation_temp_c"], df["cycles"])
    df["sulfation_index"] = sulfation_index(df["paste_peak_temp_c"], df["psoc_hours"], df["cycles"])
    df["water_loss_pct"] = water_loss_pct(df["spine_sb_pct"], df["float_voltage_v"], df["cycles"], df["ambient_temp_c"])
    df["shedding_risk"] = shedding_risk(df["packing_density_gcc"], df["four_bs_pct"], df["mud_space_mm"], df["cycles"])

    # Early-cycle slope features
    df["capacity_fade_slope_pct_per_cycle"] = (
        ((df["capacity_0_ah"] - df["capacity_25_ah"]) / np.maximum(df["capacity_0_ah"], 1))
        * 100
        / np.maximum(df["cycles"], 1)
    )
    df["rct_growth_pct_early"] = (
        (df["rct_25_mohm"] - df["rct_0_mohm"]) / np.maximum(df["rct_0_mohm"], 0.1) * 100
    )
    df["sg_drift_per_cycle"] = (df["sg_0"] - df["sg_25"]) / np.maximum(df["cycles"], 1)

    # SOH proxy
    df["capacity_fade_pct"] = npclip(
        (100 - df["capacity_25_ah"] / np.maximum(df["capacity_0_ah"], 1) * 100)
        + 0.08 * df["corrosion_index"]
        + 0.07 * df["sulfation_index"],
        0,
        45,
    )
    df["soh_pct"] = npclip(
        100
        - df["capacity_fade_pct"]
        - 0.12 * df["water_loss_pct"]
        - 0.08 * df["shedding_risk"],
        40,
        100,
    )

    df["dominant_failure_mode"] = df.apply(dominant_mode, axis=1)
    df["failure_prob_90d"] = npclip(
        failure_probability(
            df["corrosion_index"],
            df["sulfation_index"],
            df["water_loss_pct"],
            df["shedding_risk"],
            df["soh_pct"],
        )
        * 1.75,
        0,
        1,
    )

    # Warranty life survival parameters
    eta, beta, stress = warranty_life_params(df)
    df["weibull_eta_cycles"] = eta
    df["weibull_beta"] = beta
    df["early_life_stress_score"] = stress

    for horizon in [800, 1000, 1200, 1500]:
        df[f"survival_{horizon}"] = weibull_survival(horizon, df["weibull_eta_cycles"], df["weibull_beta"])

    df["warranty_decision"] = [
        base_decision(p800, p1200, p1500)
        for p800, p1200, p1500 in zip(df["survival_800"], df["survival_1200"], df["survival_1500"])
    ]

    # Median life RUL proxy: cycle at S=0.5 minus current cycles
    df["rul_cycles_mean"] = npclip(
        df["weibull_eta_cycles"] * np.power(-np.log(0.5), 1 / df["weibull_beta"]) - df["cycles"],
        0,
        2000,
    )

    return df.round(4)


@st.cache_data(show_spinner=False)
def demo_data(n=50, seed=7):
    """Create 50-battery demo batch with good, borderline and bad early-life behaviour."""
    rng = np.random.default_rng(seed)
    variants = np.array([
        "12V-100Ah",
        "12V-150Ah",
        "12V-200Ah",
        "12V-250Ah",
        "2V-500Ah",
        "Telecom",
        "Solar Deep-Cycle",
    ])
    nominal = {
        "12V-100Ah": 100,
        "12V-150Ah": 150,
        "12V-200Ah": 200,
        "12V-250Ah": 250,
        "2V-500Ah": 500,
        "Telecom": 180,
        "Solar Deep-Cycle": 220,
    }

    variant = rng.choice(variants, n, p=[0.12, 0.22, 0.22, 0.16, 0.08, 0.10, 0.10])
    quality_group = rng.choice(["good", "borderline", "bad"], n, p=[0.50, 0.30, 0.20])

    def by_group(groups, good, borderline, bad):
        return np.array([
            rng.normal(*(good if g == "good" else borderline if g == "borderline" else bad))
            for g in groups
        ])

    df = pd.DataFrame({
        "battery_id": [f"FTLAB-B01-{i:03d}" for i in range(1, n + 1)],
        "batch_id": "BATCH-DEMO-001",
        "variant": variant,
        "cycles": rng.integers(18, 51, n),
        "sg_0": rng.normal(1.258, 0.006, n),
        "sg_25": by_group(quality_group, (1.250, 0.005), (1.238, 0.007), (1.224, 0.010)),
        "voltage_v": by_group(quality_group, (12.65, 0.12), (12.45, 0.18), (12.15, 0.25)),
        "current_a": rng.normal(18, 5, n).clip(4, 45),
        "ambient_temp_c": rng.normal(33, 4, n).clip(25, 46),
        "formation_temp_c": by_group(quality_group, (47, 3), (54, 3), (59, 3)).clip(38, 65),
        "tank_temp_c": by_group(quality_group, (46, 3), (52, 3), (58, 3)).clip(35, 63),
        "paste_peak_temp_c": by_group(quality_group, (45, 2.5), (49, 3), (53, 3)).clip(38, 60),
        "spine_sb_pct": by_group(quality_group, (2.85, 0.20), (3.15, 0.25), (3.45, 0.25)).clip(2.2, 3.9),
        "packing_density_gcc": by_group(quality_group, (4.00, 0.07), (3.90, 0.14), (3.75, 0.20)).clip(3.45, 4.4),
        "four_bs_pct": by_group(quality_group, (78, 5), (69, 7), (58, 8)).clip(40, 92),
        "mud_space_mm": by_group(quality_group, (21, 1.3), (18.5, 1.6), (16, 1.8)).clip(12, 26),
        "formation_ah_input": by_group(quality_group, (114, 5), (106, 7), (96, 8)).clip(80, 135),
        "float_voltage_v": by_group(quality_group, (13.55, 0.08), (13.68, 0.12), (13.86, 0.14)).clip(13.2, 14.2),
        "psoc_hours": by_group(quality_group, (25, 12), (70, 25), (135, 35)).clip(0, 220),
        "rct_0_mohm": by_group(quality_group, (6.0, 0.6), (6.6, 0.8), (7.4, 1.0)).clip(3, 12),
        "rct_25_mohm": by_group(quality_group, (6.7, 0.7), (8.5, 1.0), (11.0, 1.5)).clip(3.5, 18),
        "r0_mohm": rng.normal(3.2, 0.6, n).clip(1.5, 7),
        "warburg_z": rng.normal(1.8, 0.4, n).clip(0.5, 4),
        "voltage_hysteresis_mv": by_group(quality_group, (35, 12), (75, 20), (125, 30)).clip(10, 220),
        "early_quality_group": quality_group,
    })

    df["nominal_ah"] = [nominal[x] for x in df["variant"]]
    df["capacity_0_ah"] = df["nominal_ah"] * rng.normal(1.00, 0.015, n)
    df["capacity_10_ah"] = df["capacity_0_ah"] * (
        1 - by_group(quality_group, (0.006, 0.004), (0.018, 0.006), (0.035, 0.010)).clip(0, 0.08)
    )
    df["capacity_25_ah"] = df["capacity_0_ah"] * (
        1 - by_group(quality_group, (0.012, 0.006), (0.035, 0.010), (0.075, 0.020)).clip(0, 0.14)
    )
    df["sg"] = df["sg_25"]

    return enrich(df)


def load_data(uploaded_file):
    """Load uploaded CSV/XLSX or fall back to demo batch."""
    if uploaded_file is None:
        return demo_data()
    if uploaded_file.name.lower().endswith(".csv"):
        raw = pd.read_csv(uploaded_file)
    else:
        raw = pd.read_excel(uploaded_file)
    return enrich(raw)

# =============================================================================
# 5. Chart builders
# =============================================================================

def gauge(value, title, suffix="%", maxv=100, low_good=True):
    """Create gauge chart. low_good=True means red at high values."""
    if low_good:
        steps = [
            {"range": [0, 35], "color": "rgba(0,229,160,.18)"},
            {"range": [35, 65], "color": "rgba(255,215,0,.22)"},
            {"range": [65, maxv], "color": "rgba(255,107,43,.28)"},
        ]
    else:
        steps = [
            {"range": [0, 70], "color": "rgba(255,107,43,.28)"},
            {"range": [70, 85], "color": "rgba(255,215,0,.22)"},
            {"range": [85, maxv], "color": "rgba(0,229,160,.18)"},
        ]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(value),
            number={"suffix": suffix},
            title={"text": title},
            gauge={
                "axis": {"range": [0, maxv]},
                "bar": {"color": CYAN},
                "steps": steps,
            },
        )
    )
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor=BG, font_color=TEXT)
    return fig


def survival_curve(row):
    """Full warranty survival curve with confidence band and cycle-horizon markers."""
    x = np.arange(0, 1601, 25)
    s = weibull_survival(x, row.weibull_eta_cycles, row.weibull_beta)

    # Simple confidence band from early-stress uncertainty proxy.
    eta_low = row.weibull_eta_cycles * (1 - 0.10 - row.early_life_stress_score / 2500)
    eta_high = row.weibull_eta_cycles * (1 + 0.10)
    upper = weibull_survival(x, eta_high, max(row.weibull_beta * 0.92, 1.1))
    lower = weibull_survival(x, eta_low, row.weibull_beta * 1.08)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=upper, name="Upper confidence", line=dict(width=0), showlegend=False))
    fig.add_trace(
        go.Scatter(
            x=x,
            y=lower,
            name="Confidence band",
            fill="tonexty",
            fillcolor="rgba(0,200,255,.18)",
            line=dict(width=0),
        )
    )
    fig.add_trace(go.Scatter(x=x, y=s, name="Predicted survival", line=dict(color=CYAN, width=4)))

    for horizon, colour in [(800, GREEN), (1000, YELLOW), (1200, ORANGE), (1500, PURPLE)]:
        fig.add_vline(x=horizon, line_dash="dash", line_color=colour)
        fig.add_annotation(x=horizon, y=0.08, text=f"{horizon} cycles", showarrow=False, font=dict(color=colour))

    fig.update_yaxes(tickformat=".0%", range=[0, 1.02], title="Survival probability")
    fig.update_xaxes(title="Warranty cycle horizon")
    fig.update_layout(
        title="Long-term warranty survival curve from early-cycle measurements",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font_color=TEXT,
        height=470,
    )
    return fig

# =============================================================================
# 6. Test description embedded README
# =============================================================================

TEST_README = """
## Test description — data required to convert early signals into warranty-cycle prediction

### Objective
Create a labelled dataset that links **early-cycle measurable variables** to **actual cycle-life outcome**. The model should answer: *given formation and first 20–50 cycle behaviour, what is the probability that the battery survives 800 / 1000 / 1200 / 1500 cycles?*

### A. Manufacturing birth-record tests
Collect for every battery serial number or statistically valid sample per batch:
- Spine alloy: Sb%, Se dosage, heat number, casting temperature, porosity / dimensional checks.
- Paste and filling: paste peak temperature, density, lignin dosage, expander batch, tubular filling uniformity.
- Curing: temperature profile, humidity profile, duration, 3BS/4BS morphology proxy or lab measurement.
- Assembly and electrolyte: fill volume, SG, acid dilution record, cell weight spread.
- Formation: charger channel, current profile, Ah input, tank temperature profile, end voltage, end SG.

### B. Early-cycle electrical characterisation, 0–50 cycles
Recommended checkpoints: cycle 0, 5, 10, 25, 50.
- C10 or relevant capacity check: capacity at each checkpoint and early capacity-fade slope.
- Coulombic efficiency and charge acceptance.
- OCV recovery after rest.
- Voltage hysteresis between charge and discharge.
- SG recovery and SG drift per cycle.
- Temperature rise during charge/discharge.

### C. EIS / ECM tests
Recommended checkpoint: after formation, cycle 10, cycle 25, cycle 50.
- Frequency sweep from low frequency to high frequency as available from EIS equipment.
- Extract R0, Rct, diffusion/Warburg proxy, arc diameter and fitting residual.
- Compute early Rct growth percentage; this is a strong leading indicator for corrosion/PAM softening risk.

### D. Accelerated life and stress tests
Run structured ALT on representative variants and intentionally stressed process windows:
- Elevated-temperature cycling for corrosion acceleration.
- PSoC cycling for sulfation acceleration.
- Overcharge / float-voltage stress for water-loss acceleration.
- High DoD deep-cycle test for active material shedding.
- Periodic water-loss measurement, SG correction, capacity check and teardown at defined intervals.

### E. End-of-life labels
For model training, every tested battery needs a clear life label:
- Failure cycle number.
- Failure definition: e.g. capacity below accepted threshold, excessive water loss, internal short, high resistance or failed recharge acceptance.
- Dominant failure mode from teardown or diagnostic evidence: corrosion, sulfation, shedding, water loss, short or mixed mode.
- Censored samples: batteries that have not yet failed but have completed a known number of cycles.

### F. Minimum model-ready table
One row per battery should include: early-cycle features, process variables, EIS features, actual failure cycle, censoring flag and dominant failure mode. This enables Weibull AFT, Cox survival, gradient-boosted survival or Bayesian survival models.

### G. Decision output required for QC
The production dashboard should provide:
- P(survive >800 cycles)
- P(survive >1000 cycles)
- P(survive >1200 cycles)
- P(survive >1500 cycles)
- PASS / REVIEW / HOLD decision
- Top contributing early-life risk drivers
"""

# =============================================================================
# 7. Sidebar controls and data selection
# =============================================================================

st.sidebar.title("🔋 Battery Digital Twin")
st.sidebar.caption("Warranty life prediction from early-cycle measurements")

uploaded = st.sidebar.file_uploader("Upload plant CSV/Excel", type=["csv", "xlsx"])
df = load_data(uploaded)

variants = st.sidebar.multiselect("Variant", sorted(df["variant"].unique()), default=sorted(df["variant"].unique()))
batches = st.sidebar.multiselect("Batch", sorted(df["batch_id"].unique()), default=sorted(df["batch_id"].unique()))

filtered = df[df["variant"].isin(variants) & df["batch_id"].isin(batches)].copy()
if filtered.empty:
    st.warning("No records match the filters. Showing the full dataset instead.")
    filtered = df.copy()

sel_id = st.sidebar.selectbox("Battery ID", filtered["battery_id"].tolist())
row = filtered[filtered["battery_id"] == sel_id].iloc[0]

st.sidebar.markdown("---")
st.sidebar.markdown("**Warranty decision gates**")
p800_gate = st.sidebar.slider("Minimum P(>800 cycles)", 0.70, 0.99, 0.90, 0.01)
p1200_gate = st.sidebar.slider("Minimum P(>1200 cycles)", 0.50, 0.95, 0.75, 0.01)
p1500_gate = st.sidebar.slider("Minimum P(>1500 cycles)", 0.30, 0.90, 0.55, 0.01)


def custom_decision(r):
    if r.survival_800 < p800_gate or r.survival_1200 < p1200_gate or r.survival_1500 < p1500_gate:
        return "HOLD"
    if r.survival_800 < p800_gate + 0.05 or r.survival_1200 < p1200_gate + 0.10 or r.survival_1500 < p1500_gate + 0.10:
        return "REVIEW"
    return "PASS"


filtered["custom_warranty_decision"] = filtered.apply(custom_decision, axis=1)
row = filtered[filtered["battery_id"] == sel_id].iloc[0]

# =============================================================================
# 8. Header KPIs
# =============================================================================

st.title("Advanced Battery Digital Twin + Warranty Life Prediction")
st.markdown(
    "<div class='small-note'>Early-cycle measurements → Weibull survival curve → PASS / REVIEW / HOLD decision against 800–1500 cycle warranty horizons.</div>",
    unsafe_allow_html=True,
)

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Batteries", f"{len(filtered):,}")
m2.metric("Mean SOH", f"{filtered['soh_pct'].mean():.1f}%")
m3.metric("Mean P>800", f"{filtered['survival_800'].mean() * 100:.1f}%")
m4.metric("Mean P>1200", f"{filtered['survival_1200'].mean() * 100:.1f}%")
m5.metric("Mean P>1500", f"{filtered['survival_1500'].mean() * 100:.1f}%")
m6.metric("Hold count", f"{(filtered['custom_warranty_decision'] == 'HOLD').sum()}")

st.markdown("---")

# =============================================================================
# 9. Dashboard tabs
# =============================================================================

tabs = st.tabs([
    "01 Warranty Prediction",
    "02 SOC Monitor",
    "03 SOH Tracker",
    "04 RUL / Weibull",
    "05 Capacity Fade",
    "06 Water Loss",
    "07 Grid Corrosion",
    "08 Sulfation",
    "09 Shedding",
    "10 Failure Probability",
    "11 Fleet & Batch QC",
    "12 Test Description README",
])

# -----------------------------------------------------------------------------
# Tab 01: Warranty Prediction
# -----------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Warranty-cycle survival decision")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("P(>800 cycles)", f"{row.survival_800 * 100:.1f}%")
    c2.metric("P(>1200 cycles)", f"{row.survival_1200 * 100:.1f}%")
    c3.metric("P(>1500 cycles)", f"{row.survival_1500 * 100:.1f}%")

    decision_class = {"PASS": "pass", "REVIEW": "review", "HOLD": "hold"}[row.custom_warranty_decision]
    c4.markdown(
        f"### Decision<br><span class='{decision_class}'>{row.custom_warranty_decision}</span>",
        unsafe_allow_html=True,
    )

    show_chart(survival_curve(row), chart_key("survival_curve_warranty_tab", row))

    drivers = pd.DataFrame({
        "Driver": [
            "Early Rct growth %",
            "Capacity fade slope %/cycle",
            "SG drift / cycle",
            "Formation temperature °C",
            "Corrosion index",
            "Sulfation index",
            "Water-loss risk",
            "Shedding risk",
            "Voltage hysteresis mV",
        ],
        "Value": [
            row.rct_growth_pct_early,
            row.capacity_fade_slope_pct_per_cycle,
            row.sg_drift_per_cycle,
            row.formation_temp_c,
            row.corrosion_index,
            row.sulfation_index,
            row.water_loss_pct,
            row.shedding_risk,
            row.voltage_hysteresis_mv,
        ],
    })
    fig = px.bar(
        drivers,
        x="Value",
        y="Driver",
        orientation="h",
        title="Measured early-life risk drivers used for warranty prediction",
    )
    show_chart(style_layout(fig), chart_key("risk_driver_bar_warranty_tab", row))

    st.dataframe(
        filtered[[
            "battery_id",
            "variant",
            "cycles",
            "soh_pct",
            "early_life_stress_score",
            "weibull_eta_cycles",
            "weibull_beta",
            "survival_800",
            "survival_1000",
            "survival_1200",
            "survival_1500",
            "custom_warranty_decision",
            "dominant_failure_mode",
        ]].sort_values(["custom_warranty_decision", "survival_1500"]),
        use_container_width=True,
    )

# -----------------------------------------------------------------------------
# Tab 02: SOC Monitor
# -----------------------------------------------------------------------------
with tabs[1]:
    a, b, c = st.columns([1, 1, 2])
    show_column_chart(a, gauge(row.soc_pct, "SOC", low_good=False), chart_key("soc_gauge", row))
    show_column_chart(b, gauge(row.voltage_v, "Voltage", " V", 15, low_good=False), chart_key("voltage_gauge", row))

    fig = px.scatter(
        filtered,
        x="sg",
        y="soc_pct",
        color="custom_warranty_decision",
        hover_data=["battery_id", "variant"],
        title="Specific gravity to SOC mapping",
    )
    show_column_chart(c, style_layout(fig), chart_key("soc_scatter", row))

# -----------------------------------------------------------------------------
# Tab 03: SOH Tracker
# -----------------------------------------------------------------------------
with tabs[2]:
    a, b = st.columns([1, 2])
    show_column_chart(a, gauge(row.soh_pct, "SOH", low_good=False), chart_key("soh_gauge", row))

    fig = px.scatter(
        filtered,
        x="cycles",
        y="soh_pct",
        color="custom_warranty_decision",
        symbol="dominant_failure_mode",
        hover_data=["battery_id"],
        title="SOH versus early cycle count",
    )
    show_column_chart(b, style_layout(fig), chart_key("soh_scatter", row))

# -----------------------------------------------------------------------------
# Tab 04: RUL / Weibull
# -----------------------------------------------------------------------------
with tabs[3]:
    show_chart(survival_curve(row), chart_key("survival_curve_weibull_tab", row))

    fig = px.scatter(
        filtered,
        x="weibull_eta_cycles",
        y="weibull_beta",
        color="custom_warranty_decision",
        size="early_life_stress_score",
        hover_data=["battery_id", "survival_1500"],
        title="Weibull parameter map: characteristic life η and shape β",
    )
    show_chart(style_layout(fig), chart_key("weibull_parameter_map", row))

# -----------------------------------------------------------------------------
# Tab 05: Capacity Fade
# -----------------------------------------------------------------------------
with tabs[4]:
    show_chart(gauge(row.capacity_fade_pct, "Capacity fade"), chart_key("capacity_fade_gauge", row))

    fig = px.scatter(
        filtered,
        x="capacity_fade_slope_pct_per_cycle",
        y="survival_1500",
        color="custom_warranty_decision",
        hover_data=["battery_id"],
        title="Early capacity-fade slope versus P(>1500 cycles)",
    )
    show_chart(style_layout(fig), chart_key("capacity_fade_vs_survival", row))

# -----------------------------------------------------------------------------
# Tab 06: Water Loss
# -----------------------------------------------------------------------------
with tabs[5]:
    show_chart(gauge(row.water_loss_pct, "Water-loss risk"), chart_key("water_loss_gauge", row))

    fig = px.scatter(
        filtered,
        x="spine_sb_pct",
        y="water_loss_pct",
        color="custom_warranty_decision",
        hover_data=["battery_id"],
        title="Sb% and water-loss risk",
    )
    show_chart(style_layout(fig), chart_key("water_loss_scatter", row))

# -----------------------------------------------------------------------------
# Tab 07: Grid Corrosion
# -----------------------------------------------------------------------------
with tabs[6]:
    show_chart(gauge(row.corrosion_index, "Grid corrosion index", "", 100), chart_key("corrosion_gauge", row))

    fig = px.scatter(
        filtered,
        x="formation_temp_c",
        y="corrosion_index",
        color="custom_warranty_decision",
        hover_data=["battery_id"],
        title="Formation temperature linkage to corrosion index",
    )
    show_chart(style_layout(fig), chart_key("corrosion_scatter", row))

# -----------------------------------------------------------------------------
# Tab 08: Sulfation
# -----------------------------------------------------------------------------
with tabs[7]:
    show_chart(gauge(row.sulfation_index, "Sulfation index", "", 100), chart_key("sulfation_gauge", row))

    fig = px.scatter(
        filtered,
        x="psoc_hours",
        y="sulfation_index",
        color="custom_warranty_decision",
        hover_data=["battery_id"],
        title="PSoC hours versus sulfation index",
    )
    show_chart(style_layout(fig), chart_key("sulfation_scatter", row))

# -----------------------------------------------------------------------------
# Tab 09: Shedding
# -----------------------------------------------------------------------------
with tabs[8]:
    show_chart(gauge(row.shedding_risk, "Shedding risk", "", 100), chart_key("shedding_gauge", row))

    fig = px.scatter(
        filtered,
        x="packing_density_gcc",
        y="shedding_risk",
        color="custom_warranty_decision",
        hover_data=["battery_id"],
        title="Packing density deviation and shedding risk",
    )
    show_chart(style_layout(fig), chart_key("shedding_scatter", row))

# -----------------------------------------------------------------------------
# Tab 10: Failure Probability
# -----------------------------------------------------------------------------
with tabs[9]:
    show_chart(
        gauge(row.failure_prob_90d * 100, "Short-term failure probability"),
        chart_key("short_term_failure_gauge", row),
    )

    fig = px.histogram(
        filtered,
        x="failure_prob_90d",
        color="custom_warranty_decision",
        nbins=20,
        title="Short-term diagnostic risk distribution",
    )
    fig.update_xaxes(tickformat=".0%")
    show_chart(style_layout(fig), chart_key("short_term_failure_hist", row))

# -----------------------------------------------------------------------------
# Tab 11: Fleet & Batch QC
# -----------------------------------------------------------------------------
with tabs[10]:
    summary = filtered.groupby(["batch_id", "custom_warranty_decision"], as_index=False).agg(
        batteries=("battery_id", "count"),
        mean_p800=("survival_800", "mean"),
        mean_p1200=("survival_1200", "mean"),
        mean_p1500=("survival_1500", "mean"),
        mean_soh=("soh_pct", "mean"),
        mean_stress=("early_life_stress_score", "mean"),
    )

    fig = px.bar(
        summary,
        x="batch_id",
        y="batteries",
        color="custom_warranty_decision",
        title="Warranty decision distribution by batch",
    )
    show_chart(style_layout(fig), "fleet_batch_qc_decision_bar")

    st.dataframe(summary.round(3), use_container_width=True)
    st.download_button(
        "Download warranty QC summary",
        summary.to_csv(index=False).encode("utf-8"),
        "warranty_qc_summary.csv",
        "text/csv",
    )

# -----------------------------------------------------------------------------
# Tab 12: Test Description README
# -----------------------------------------------------------------------------
with tabs[11]:
    st.markdown(TEST_README)

st.caption(
    "Prototype model: transparent Weibull-survival surrogate. For production, train using early-cycle features and actual failure-cycle labels, including censored data."
)
