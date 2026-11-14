"""
Step 9: Interactive Streamlit Frontend Web App for Customer Lifetime Value (CLV) Prediction
Run locally via: streamlit run app.py
"""

import os
import json
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CLV Intelligence Studio | Real-Time Predictor",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# 2. FLUID RESPONSIVE CSS & KEYFRAME ANIMATIONS
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Keyframe Animations */
    @keyframes fadeInSlideUp {
        0% { opacity: 0; transform: translateY(14px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 18px rgba(16, 185, 129, 0.22); }
        50% { box-shadow: 0 0 32px rgba(16, 185, 129, 0.48); }
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes floatBadge {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-3px); }
    }

    /* Main Container Fluid Spacing */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1440px !important;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f3d3e 100%);
        background-size: 200% 200%;
        animation: gradientShift 10s ease infinite, fadeInSlideUp 0.6s ease-out;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 18px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.4rem;
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.28);
    }
    .hero-title {
        font-size: clamp(1.45rem, 2.5vw, 2.15rem);
        font-weight: 800;
        color: #f8fafc;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: clamp(0.88rem, 1.2vw, 1rem);
        color: #94a3b8;
        margin: 0;
    }
    .hero-badge-group {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        flex-wrap: wrap;
    }
    .status-pill {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.45);
        color: #34d399;
        padding: 0.4rem 0.9rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        animation: floatBadge 3.5s ease-in-out infinite;
    }
    .user-pill {
        background: rgba(59, 130, 246, 0.16);
        border: 1px solid rgba(96, 165, 250, 0.4);
        color: #93c5fd;
        padding: 0.4rem 0.95rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
    }

    /* Animated Glass Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.65);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.35rem 1.5rem;
        margin-bottom: 1rem;
        animation: fadeInSlideUp 0.5s ease-out;
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    }
    .glass-card:hover {
        transform: translateY(-3px);
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.25);
    }

    /* Real-Time Prediction Hero Card */
    .forecast-hero {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.85) 0%, rgba(15, 23, 42, 0.92) 100%);
        border: 1px solid rgba(52, 211, 153, 0.5);
        border-radius: 18px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.15rem;
        animation: pulseGlow 4s infinite ease-in-out, fadeInSlideUp 0.5s ease-out;
        position: relative;
        overflow: hidden;
    }
    .forecast-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6ee7b7;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .forecast-value {
        font-size: clamp(2rem, 3.5vw, 2.85rem);
        font-weight: 800;
        color: #ecfdf5;
        line-height: 1.15;
        margin-bottom: 0.5rem;
    }
    .forecast-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem;
        align-items: center;
        font-size: 0.85rem;
        color: #a7f3d0;
    }
    .tier-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.78rem;
        background: rgba(255, 255, 255, 0.14);
        color: #ffffff;
    }

    /* Responsive KPI Grid */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 0.85rem;
        margin-bottom: 1.15rem;
    }
    .kpi-box {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        transition: all 0.22s ease;
    }
    .kpi-box:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.45);
    }
    .kpi-title {
        font-size: 0.76rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }
    .kpi-num {
        font-size: 1.38rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .kpi-sub {
        font-size: 0.75rem;
        color: #38bdf8;
        margin-top: 0.2rem;
    }

    /* Insight Recommendation Banner */
    .insight-card {
        background: linear-gradient(90deg, rgba(30, 58, 138, 0.4) 0%, rgba(15, 23, 42, 0.7) 100%);
        border-left: 4px solid #38bdf8;
        border-radius: 10px;
        padding: 0.9rem 1.15rem;
        margin-top: 0.6rem;
        margin-bottom: 0.8rem;
        font-size: 0.9rem;
        color: #e2e8f0;
    }

    /* Profile Card in Sidebar */
    .sidebar-profile {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.95));
        border: 1px solid rgba(96, 165, 250, 0.3);
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    .avatar-circle {
        width: 54px;
        height: 54px;
        border-radius: 50%;
        margin: 0 auto 0.55rem auto;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.35rem;
        font-weight: 800;
        color: #ffffff;
        background: linear-gradient(135deg, #3b82f6, #10b981);
        box-shadow: 0 4px 14px rgba(59, 130, 246, 0.4);
    }

    /* Responsive adjustments for mobile screens */
    @media (max-width: 768px) {
        .hero-banner {
            padding: 1.15rem 1.25rem;
            flex-direction: column;
            align-items: flex-start;
        }
        .forecast-hero {
            padding: 1.15rem 1.25rem;
        }
        .kpi-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 3. DATA & ARTIFACTS LOADING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """Load trained sklearn artifacts if present, plus JSON metadata fallback."""
    meta = None
    if os.path.exists("model_artifacts.json"):
        with open("model_artifacts.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

    if os.path.exists("linear_clv_model.pkl") and os.path.exists("clv_scaler.pkl"):
        import joblib

        model = joblib.load("linear_clv_model.pkl")
        scaler = joblib.load("clv_scaler.pkl")
        return {"mode": "sklearn", "model": model, "scaler": scaler, "meta": meta}
    elif meta is not None:
        return {"mode": "json", "meta": meta}
    else:
        raise FileNotFoundError(
            "Run `python train_pipeline.py` first to generate model artifacts."
        )


@st.cache_data
def load_customer_dataset(meta_fallback):
    """Load ecommerce_customers.csv if present, or fall back to sample records."""
    if os.path.exists("ecommerce_customers.csv"):
        df = pd.read_csv("ecommerce_customers.csv")
        if "Time on App" in df.columns:
            df["Time on App"] = df["Time on App"].fillna(df["Time on App"].median())
        df["App_to_Web_Ratio"] = df["Time on App"] / df["Time on Website"].clip(lower=0.1)
        df["Engagement_Score"] = df["Avg. Session Length"] * df["Length of Membership"]
        return df
    elif meta_fallback and "sample_records" in meta_fallback:
        return pd.DataFrame(meta_fallback["sample_records"])
    return pd.DataFrame()


artifacts = load_artifacts()
meta_data = artifacts.get("meta") or {
    "intercept": 498.8146,
    "scaler_mean": [33.073963, 12.085704, 37.148821, 3.55494],
    "scaler_scale": [0.961222, 0.957648, 1.036059, 0.910954],
    "feature_cols": [
        "Avg. Session Length",
        "Time on App",
        "Time on Website",
        "Length of Membership",
    ],
    "scaled_coefficients": {
        "Avg. Session Length": 25.3039,
        "Time on App": 37.3365,
        "Time on Website": -0.4382,
        "Length of Membership": 55.1343,
    },
    "unscaled_coefficients": {
        "Avg. Session Length": 26.3248,
        "Time on App": 38.9877,
        "Time on Website": -0.4229,
        "Length of Membership": 60.5237,
    },
    "mae": 7.22,
    "rmse": 9.05,
    "r2": 0.9869,
    "clean_rows": 489,
}
customers_df = load_customer_dataset(meta_data)


# -----------------------------------------------------------------------------
# 4. SESSION STATE INITIALIZATION (AUTH, SLIDERS, SAVED SCENARIOS)
# -----------------------------------------------------------------------------
if "signed_in" not in st.session_state:
    st.session_state.signed_in = True
if "user_name" not in st.session_state:
    st.session_state.user_name = "Yash Saini"
if "user_email" not in st.session_state:
    st.session_state.user_email = "yash.saini@analytics.io"
if "user_role" not in st.session_state:
    st.session_state.user_role = "AI & Data Science Engineer"
if "login_time" not in st.session_state:
    st.session_state.login_time = datetime.now().strftime("%d %b %Y, %I:%M %p")

# Default slider states
if "session_length" not in st.session_state:
    st.session_state.session_length = 34.0
if "time_on_app" not in st.session_state:
    st.session_state.time_on_app = 12.0
if "time_on_web" not in st.session_state:
    st.session_state.time_on_web = 36.0
if "membership_length" not in st.session_state:
    st.session_state.membership_length = 3.5
if "active_preset_name" not in st.session_state:
    st.session_state.active_preset_name = "Custom Live Configuration"
if "saved_scenarios" not in st.session_state:
    st.session_state.saved_scenarios = []


def apply_preset(name, s_len, app_t, web_t, mem_l):
    """Callback to load a persona preset into the real-time sliders."""
    st.session_state.active_preset_name = name
    st.session_state.session_length = float(s_len)
    st.session_state.time_on_app = float(app_t)
    st.session_state.time_on_web = float(web_t)
    st.session_state.membership_length = float(mem_l)


def predict_clv(s_len, app_t, web_t, mem_l):
    """Predict yearly CLV for a single set of engagement inputs."""
    input_arr = np.array([[s_len, app_t, web_t, mem_l]])
    if artifacts["mode"] == "sklearn":
        scaled = artifacts["scaler"].transform(input_arr)
        pred = float(artifacts["model"].predict(scaled)[0])
    else:
        mean = np.array(meta_data["scaler_mean"])
        scale = np.array(meta_data["scaler_scale"])
        coefs = np.array(
            [meta_data["scaled_coefficients"][c] for c in meta_data["feature_cols"]]
        )
        scaled = (input_arr[0] - mean) / scale
        pred = float(meta_data["intercept"] + np.dot(scaled, coefs))
    return pred


def get_customer_tier(clv_value):
    """Classify predicted CLV into an actionable business tier."""
    if clv_value >= 600:
        return "💎 Platinum VIP", "#34d399"
    elif clv_value >= 520:
        return "🥇 Gold Growth", "#fbbf24"
    elif clv_value >= 440:
        return "🥈 Silver Core", "#38bdf8"
    else:
        return "🌱 Bronze Starter", "#f87171"


# -----------------------------------------------------------------------------
# 5. SIDEBAR: PROFILE SIGN-IN, PERSONA PRESETS & QUICK TOOLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 👤 Account & Workspace")

    if st.session_state.signed_in:
        initials = "".join(
            [part[0].upper() for part in st.session_state.user_name.split()[:2]]
        ) or "YS"
        st.markdown(
            f"""
            <div class="sidebar-profile">
                <div class="avatar-circle">{initials}</div>
                <div style="font-weight:700; font-size:1.05rem; color:#f8fafc;">
                    {st.session_state.user_name}
                </div>
                <div style="font-size:0.8rem; color:#38bdf8; margin-top:2px;">
                    {st.session_state.user_role}
                </div>
                <div style="font-size:0.74rem; color:#94a3b8; margin-top:4px;">
                    {st.session_state.user_email}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("⚙️ Edit Profile / Switch Account", expanded=False):
            new_name = st.text_input("Full Name", value=st.session_state.user_name)
            new_email = st.text_input("Email Address", value=st.session_state.user_email)
            role_options = [
                "AI & Data Science Engineer",
                "Growth Product Manager",
                "Customer Retention Analyst",
                "VP of E-Commerce Strategy",
            ]
            current_idx = (
                role_options.index(st.session_state.user_role)
                if st.session_state.user_role in role_options
                else 0
            )
            new_role = st.selectbox("Workspace Role", role_options, index=current_idx)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Save Profile", use_container_width=True, type="primary"):
                    st.session_state.user_name = new_name.strip() or "User"
                    st.session_state.user_email = new_email.strip() or "user@company.com"
                    st.session_state.user_role = new_role
                    st.rerun()
            with c2:
                if st.button("Sign Out", use_container_width=True):
                    st.session_state.signed_in = False
                    st.rerun()
    else:
        st.info("Sign in to unlock scenario bookmarking and personalized reports.")
        with st.form("sidebar_signin_form"):
            login_name = st.text_input("Name", placeholder="e.g., Yash Saini")
            login_email = st.text_input("Email", placeholder="name@domain.com")
            login_role = st.selectbox(
                "Role",
                [
                    "AI & Data Science Engineer",
                    "Growth Product Manager",
                    "Customer Retention Analyst",
                    "VP of E-Commerce Strategy",
                ],
            )
            submitted = st.form_submit_button(
                "🔐 Sign In to Studio", use_container_width=True, type="primary"
            )
            if submitted:
                st.session_state.signed_in = True
                st.session_state.user_name = login_name.strip() or "Yash Saini"
                st.session_state.user_email = (
                    login_email.strip() or "yash.saini@analytics.io"
                )
                st.session_state.user_role = login_role
                st.session_state.login_time = datetime.now().strftime(
                    "%d %b %Y, %I:%M %p"
                )
                st.rerun()

    st.markdown("---")
    st.markdown("### ⚡ Instant Persona Presets")
    st.caption("Click any customer archetype to animate sliders in real time:")

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.button(
            "💎 VIP Mobile",
            use_container_width=True,
            on_click=apply_preset,
            args=("💎 Platinum Mobile VIP", 36.0, 15.0, 35.0, 5.5),
        )
        st.button(
            "🌱 New Visitor",
            use_container_width=True,
            on_click=apply_preset,
            args=("🌱 New Web Visitor", 31.0, 9.5, 39.5, 1.2),
        )
    with p_col2:
        st.button(
            "🚀 Loyal Omni",
            use_container_width=True,
            on_click=apply_preset,
            args=("🚀 Loyal Omnichannel", 34.5, 13.5, 37.5, 4.5),
        )
        st.button(
            "🔄 Baseline",
            use_container_width=True,
            on_click=apply_preset,
            args=("🔄 Dataset Baseline", 33.0, 12.0, 37.0, 3.5),
        )

    st.markdown("---")
    st.markdown("### 📊 Financial Assumptions")
    discount_rate = st.slider(
        "Annual Retention / Discount Factor (%)",
        min_value=50,
        max_value=100,
        value=85,
        step=5,
        help="Used to compute 3-Year Cumulative Customer Lifetime Value.",
    )
    st.caption(
        f"Model Status: **OLS Linear Regression** (`R² = {meta_data.get('r2', 0.9869)*100:.2f}%`)"
    )


# -----------------------------------------------------------------------------
# 6. LIVE PREDICTION COMPUTATION (REAL-TIME)
# -----------------------------------------------------------------------------
session_length = st.session_state.session_length
time_on_app = st.session_state.time_on_app
time_on_web = st.session_state.time_on_web
membership_length = st.session_state.membership_length

prediction = predict_clv(session_length, time_on_app, time_on_web, membership_length)
baseline_spend = float(meta_data.get("intercept", 498.8146))
spend_delta = prediction - baseline_spend
spend_delta_pct = (spend_delta / baseline_spend) * 100.0

app_web_ratio = time_on_app / max(time_on_web, 0.1)
engagement_score = session_length * membership_length
retention_factor = discount_rate / 100.0
three_year_ltv = prediction * (1.0 + retention_factor + (retention_factor**2))
tier_label, tier_color = get_customer_tier(prediction)

if not customers_df.empty and "Yearly Amount Spent" in customers_df.columns:
    percentile_rank = float(
        (customers_df["Yearly Amount Spent"] < prediction).mean() * 100.0
    )
else:
    percentile_rank = float(np.clip(50.0 + spend_delta * 0.35, 1.0, 99.9))


# -----------------------------------------------------------------------------
# 7. TOP HERO BANNER
# -----------------------------------------------------------------------------
active_user_display = (
    f"👤 {st.session_state.user_name}" if st.session_state.signed_in else "👤 Guest Mode"
)
st.markdown(
    f"""
    <div class="hero-banner">
        <div>
            <div class="hero-title">💸 Customer Lifetime Value (CLV) Intelligence Studio</div>
            <div class="hero-subtitle">
                Real-time predictive revenue modeling, cohort ROI simulation, and behavioral driver analytics
            </div>
        </div>
        <div class="hero-badge-group">
            <span class="status-pill">● Real-Time Engine Active</span>
            <span class="user-pill">{active_user_display}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 8. MULTI-SECTION INTERACTIVE TABS
# -----------------------------------------------------------------------------
tab_sim, tab_roi, tab_directory, tab_diagnostics, tab_profile = st.tabs(
    [
        "🎯 Real-Time Simulator",
        "📈 Campaign & ROI Lab",
        "👥 Customer Directory",
        "🔬 Model Diagnostics & EDA",
        f"🔖 Saved Scenarios ({len(st.session_state.saved_scenarios)})",
    ]
)

# =============================================================================
# TAB 1: REAL-TIME CLV SIMULATOR (NO REDUNDANT CALCULATE BUTTON)
# =============================================================================
with tab_sim:
    col_form, col_result = st.columns([1.05, 1.25], gap="large")

    with col_form:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🎛️ Customer Engagement Controls")
        st.caption(
            f"Active Profile: **{st.session_state.active_preset_name}** — Drag any slider to update predictions immediately."
        )

        session_length = st.slider(
            "⏱️ Average Session Length (Minutes)",
            min_value=20.0,
            max_value=50.0,
            key="session_length",
            step=0.5,
        )
        time_on_app = st.slider(
            "📱 Time spent on Mobile App (Minutes)",
            min_value=5.0,
            max_value=20.0,
            key="time_on_app",
            step=0.25,
        )
        time_on_web = st.slider(
            "💻 Time spent on Desktop Website (Minutes)",
            min_value=10.0,
            max_value=50.0,
            key="time_on_web",
            step=0.5,
        )
        membership_length = st.slider(
            "🏅 Length of Membership (Years)",
            min_value=0.0,
            max_value=10.0,
            key="membership_length",
            step=0.1,
        )

        # Recompute synchronously with latest slider values
        prediction = predict_clv(
            session_length, time_on_app, time_on_web, membership_length
        )
        spend_delta = prediction - baseline_spend
        spend_delta_pct = (spend_delta / baseline_spend) * 100.0
        app_web_ratio = time_on_app / max(time_on_web, 0.1)
        engagement_score = session_length * membership_length
        three_year_ltv = prediction * (1.0 + retention_factor + (retention_factor**2))
        tier_label, tier_color = get_customer_tier(prediction)
        if not customers_df.empty and "Yearly Amount Spent" in customers_df.columns:
            percentile_rank = float(
                (customers_df["Yearly Amount Spent"] < prediction).mean() * 100.0
            )

        # Action bar for saving scenarios (replaces the old Calculate button with useful bookmarking)
        b_col1, b_col2 = st.columns([1.4, 1])
        with b_col1:
            scenario_label = st.text_input(
                "Scenario Tag",
                value=f"Scenario #{len(st.session_state.saved_scenarios) + 1}",
                label_visibility="collapsed",
                placeholder="Name this scenario...",
            )
        with b_col2:
            if st.button("📌 Bookmark Scenario", use_container_width=True):
                st.session_state.saved_scenarios.append(
                    {
                        "Timestamp": datetime.now().strftime("%H:%M:%S"),
                        "Scenario": scenario_label,
                        "User": st.session_state.user_name
                        if st.session_state.signed_in
                        else "Guest",
                        "Tier": tier_label,
                        "Avg Session (m)": session_length,
                        "Mobile App (m)": time_on_app,
                        "Website (m)": time_on_web,
                        "Membership (yr)": membership_length,
                        "Predicted Yearly CLV ($)": round(prediction, 2),
                        "3-Yr Projected LTV ($)": round(three_year_ltv, 2),
                    }
                )
                st.toast(f"Saved '{scenario_label}' to your bookmarks!", icon="✅")

        st.markdown("</div>", unsafe_allow_html=True)

        # Dynamic AI Strategic Recommendation Card
        if time_on_app < 11.5 and time_on_web > 35.0:
            rec_title = "📲 High-Priority Mobile App Migration Opportunity"
            rec_body = (
                "This user spends heavily on Desktop Web (**-$0.43/min** impact) but under-utilizes the Mobile App "
                "(**+$38.99/min** impact). Offering an app-exclusive checkout reward that shifts just **+3 minutes** to "
                f"the mobile app will boost yearly CLV by **+${3 * 38.99:,.2f}**."
            )
        elif membership_length < 2.5:
            rec_title = "🔒 Early-Tenure Loyalty Retention Trigger"
            rec_body = (
                "Membership duration is the #1 revenue driver (**+$60.52/year**). Enrolling this customer in an annual "
                "VIP loyalty tier to extend retention by **+2 years** projects an additional "
                f"**+${2 * 60.52:,.2f}/year** in spending."
            )
        else:
            rec_title = "🌟 High-Value Omnichannel Champion"
            rec_body = (
                "Strong mobile app usage and membership tenure place this profile in the upper spending distribution. "
                "Prioritize early-access product drops and referral rewards to maximize 3-year LTV."
            )

        st.markdown(
            f"""
            <div class="insight-card">
                <div style="font-weight:700; color:#38bdf8; margin-bottom:4px;">{rec_title}</div>
                <div>{rec_body}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_result:
        delta_sign = "+" if spend_delta >= 0 else ""
        st.markdown(
            f"""
            <div class="forecast-hero">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
                    <div class="forecast-label">⚡ Live Predicted Yearly Spending</div>
                    <span class="tier-badge" style="border:1px solid {tier_color};">{tier_label}</span>
                </div>
                <div class="forecast-value">${prediction:,.2f} <span style="font-size:1rem; font-weight:500; color:#a7f3d0;">/ year</span></div>
                <div class="forecast-meta">
                    <span>📊 <b>{delta_sign}${spend_delta:,.2f} ({delta_sign}{spend_delta_pct:.1f}%)</b> vs. Dataset Mean ($498.81)</span>
                    <span>•</span>
                    <span>🏆 Outperforms <b>{percentile_rank:.1f}%</b> of customer cohort</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Responsive 4-Card KPI Grid
        st.markdown(
            f"""
            <div class="kpi-grid">
                <div class="kpi-box">
                    <div class="kpi-title">Monthly Equivalent</div>
                    <div class="kpi-num">${prediction / 12.0:,.2f}</div>
                    <div class="kpi-sub">Recurring run-rate</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-title">3-Year Projected LTV</div>
                    <div class="kpi-num">${three_year_ltv:,.2f}</div>
                    <div class="kpi-sub">@ {discount_rate}% annual retention</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-title">App-to-Web Ratio</div>
                    <div class="kpi-num">{app_web_ratio:.2f}x</div>
                    <div class="kpi-sub">{time_on_app:.1f}m App vs {time_on_web:.1f}m Web</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-title">Engagement Score</div>
                    <div class="kpi-num">{engagement_score:.1f}</div>
                    <div class="kpi-sub">Session × Tenure index</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Live Feature Contribution Breakdown Chart (Dollar Impact vs Mean Baseline)
        st.markdown("#### 🧩 Live Revenue Contribution by Feature (vs. $498.81 Mean)")
        means = meta_data["scaler_mean"]
        scales = meta_data["scaler_scale"]
        s_coefs = [
            meta_data["scaled_coefficients"][c] for c in meta_data["feature_cols"]
        ]
        user_vals = [session_length, time_on_app, time_on_web, membership_length]
        contributions = [
            ((u - m) / s) * c for u, m, s, c in zip(user_vals, means, scales, s_coefs)
        ]
        labels = [
            f"Avg. Session ({session_length:.1f}m)",
            f"Mobile App ({time_on_app:.2f}m)",
            f"Website ({time_on_web:.1f}m)",
            f"Membership ({membership_length:.1f}y)",
        ]

        fig_contrib, ax_contrib = plt.subplots(figsize=(7.2, 2.75), dpi=120)
        fig_contrib.patch.set_facecolor("#0f172a")
        ax_contrib.set_facecolor("#0f172a")

        bar_colors = ["#10b981" if val >= 0 else "#f43f5e" for val in contributions]
        y_pos = np.arange(len(labels))
        bars = ax_contrib.barh(
            y_pos, contributions, color=bar_colors, height=0.58, edgecolor="none"
        )
        ax_contrib.axvline(0, color="#64748b", linewidth=1, linestyle="--")
        ax_contrib.set_yticks(y_pos)
        ax_contrib.set_yticklabels(labels, color="#e2e8f0", fontsize=9.5)
        ax_contrib.set_xlabel(
            "Dollar Contribution Relative to Average Customer ($)",
            color="#94a3b8",
            fontsize=8.5,
        )
        ax_contrib.tick_params(axis="x", colors="#94a3b8", labelsize=8.5)
        for spine in ["top", "right", "left"]:
            ax_contrib.spines[spine].set_visible(False)
        ax_contrib.spines["bottom"].set_color("#334155")

        max_abs = max([abs(v) for v in contributions] + [25.0])
        ax_contrib.set_xlim(-max_abs * 1.32, max_abs * 1.32)

        for bar, val in zip(bars, contributions):
            offset = max_abs * 0.04 if val >= 0 else -max_abs * 0.04
            align = "left" if val >= 0 else "right"
            sign_str = "+" if val >= 0 else ""
            ax_contrib.text(
                val + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{sign_str}${val:,.2f}",
                va="center",
                ha=align,
                color="#f8fafc",
                fontsize=8.8,
                fontweight="bold",
            )

        plt.tight_layout()
        st.pyplot(fig_contrib, use_container_width=True)
        plt.close(fig_contrib)

        with st.expander("📋 View Standardized Model Weights Table", expanded=False):
            drivers_df = pd.DataFrame(
                {
                    "Feature": [
                        "Length of Membership",
                        "Time on App",
                        "Avg. Session Length",
                        "Time on Website",
                    ],
                    "Scaled Coefficient ($ / SD)": [55.13, 37.34, 25.30, -0.44],
                    "Unscaled Impact ($ / unit)": [
                        "+$60.52 / yr",
                        "+$38.99 / min",
                        "+$26.32 / min",
                        "-$0.42 / min",
                    ],
                }
            )
            st.dataframe(drivers_df, hide_index=True, use_container_width=True)


# =============================================================================
# TAB 2: CAMPAIGN & ROI WHAT-IF LAB
# =============================================================================
with tab_roi:
    st.subheader("📈 Cohort Marketing Campaign & ROI Simulator")
    st.write(
        "Simulate the financial return of product or marketing interventions (such as mobile app onboarding nudges or annual loyalty programs) across a target customer cohort."
    )

    c_sim1, c_sim2, c_sim3 = st.columns(3)
    with c_sim1:
        cohort_size = st.number_input(
            "Target Customer Cohort Size",
            min_value=50,
            max_value=100000,
            value=1000,
            step=50,
        )
        campaign_cost_per_user = st.slider(
            "Campaign Spend per Customer ($)", 0.0, 150.0, 25.0, 2.5
        )
    with c_sim2:
        lift_app_mins = st.slider(
            "Projected Lift in Mobile App Time (+Minutes)", 0.0, 8.0, 2.0, 0.25
        )
        lift_session_mins = st.slider(
            "Projected Lift in Session Length (+Minutes)", 0.0, 10.0, 1.5, 0.5
        )
    with c_sim3:
        lift_membership_yrs = st.slider(
            "Projected Lift in Membership Retention (+Years)", 0.0, 5.0, 0.5, 0.1
        )
        web_shift_mins = st.slider(
            "Desktop Web Time Adjustment (Minutes)", -10.0, 10.0, -2.0, 0.5
        )

    new_pred_single = predict_clv(
        np.clip(session_length + lift_session_mins, 10.0, 60.0),
        np.clip(time_on_app + lift_app_mins, 1.0, 30.0),
        np.clip(time_on_web + web_shift_mins, 5.0, 60.0),
        np.clip(membership_length + lift_membership_yrs, 0.0, 15.0),
    )

    per_user_lift = new_pred_single - prediction
    baseline_cohort_rev = prediction * cohort_size
    new_cohort_rev = new_pred_single * cohort_size
    gross_rev_lift = per_user_lift * cohort_size
    total_campaign_cost = campaign_cost_per_user * cohort_size
    net_profit_lift = gross_rev_lift - total_campaign_cost
    roi_pct = (
        (net_profit_lift / total_campaign_cost) * 100.0
        if total_campaign_cost > 0
        else 0.0
    )

    r1, r2, r3, r4 = st.columns(4)
    r1.metric(
        "New CLV per Customer",
        f"${new_pred_single:,.2f}",
        f"{per_user_lift:+,.2f} / user",
    )
    r2.metric(
        "Gross Cohort Revenue Lift",
        f"${gross_rev_lift:,.0f}",
        f"From ${baseline_cohort_rev:,.0f} base",
    )
    r3.metric(
        "Net Profit After Campaign",
        f"${net_profit_lift:,.0f}",
        f"Cost: ${total_campaign_cost:,.0f}",
    )
    r4.metric(
        "Simulated Campaign ROI",
        f"{roi_pct:,.1f}%" if total_campaign_cost > 0 else "Infinite (Zero Cost)",
        "Positive Return" if net_profit_lift >= 0 else "Negative Return",
    )

    # Sensitivity Curve Plot
    st.markdown("#### 📈 CLV Sensitivity Across Mobile App Engagement & Tenure")
    app_grid = np.linspace(6.0, 18.0, 40)
    fig_sens, ax_sens = plt.subplots(figsize=(8.5, 3.2), dpi=120)
    fig_sens.patch.set_facecolor("#0f172a")
    ax_sens.set_facecolor("#0f172a")

    for tenure_val, line_col in [
        (2.0, "#38bdf8"),
        (membership_length, "#10b981"),
        (6.0, "#f59e0b"),
    ]:
        curve_preds = [
            predict_clv(session_length, a_val, time_on_web, tenure_val)
            for a_val in app_grid
        ]
        ax_sens.plot(
            app_grid,
            curve_preds,
            label=f"Tenure = {tenure_val:.1f} yrs",
            color=line_col,
            linewidth=2.3,
        )

    ax_sens.scatter(
        [time_on_app],
        [prediction],
        color="#ffffff",
        s=70,
        zorder=5,
        label="Current Baseline",
    )
    ax_sens.scatter(
        [np.clip(time_on_app + lift_app_mins, 6.0, 18.0)],
        [new_pred_single],
        color="#f43f5e",
        s=85,
        zorder=6,
        label="Post-Campaign Target",
    )
    ax_sens.set_xlabel("Time Spent on Mobile App (Minutes)", color="#94a3b8")
    ax_sens.set_ylabel("Predicted Yearly CLV ($)", color="#94a3b8")
    ax_sens.tick_params(colors="#94a3b8")
    ax_sens.grid(color="#1e293b", linestyle="--", linewidth=0.8)
    for spine in ["top", "right"]:
        ax_sens.spines[spine].set_visible(False)
    ax_sens.spines["bottom"].set_color("#334155")
    ax_sens.spines["left"].set_color("#334155")
    ax_sens.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc")
    plt.tight_layout()
    st.pyplot(fig_sens, use_container_width=True)
    plt.close(fig_sens)


# =============================================================================
# TAB 3: CUSTOMER DIRECTORY & RECORD INSPECTOR
# =============================================================================
with tab_directory:
    st.subheader("👥 Customer Directory & Profile Loader")
    if customers_df.empty:
        st.warning("Customer dataset not found.")
    else:
        f_col1, f_col2, f_col3 = st.columns([1.4, 1, 1])
        with f_col1:
            search_query = st.text_input(
                "🔍 Search by Email or Address",
                placeholder="e.g., customer101@gmail.com or CA",
            )
        with f_col2:
            min_spend_filter = st.slider(
                "Minimum Yearly Spend ($)",
                int(customers_df["Yearly Amount Spent"].min()),
                int(customers_df["Yearly Amount Spent"].max()),
                int(customers_df["Yearly Amount Spent"].min()),
            )
        with f_col3:
            sort_col = st.selectbox(
                "Sort Customers By",
                [
                    "Yearly Amount Spent",
                    "Length of Membership",
                    "Time on App",
                    "Avg. Session Length",
                ],
            )

        filtered_df = customers_df[
            customers_df["Yearly Amount Spent"] >= min_spend_filter
        ].copy()
        if search_query.strip():
            q = search_query.strip().lower()
            mask = filtered_df["Email"].str.lower().str.contains(q, na=False)
            if "Address" in filtered_df.columns:
                mask = mask | filtered_df["Address"].str.lower().str.contains(
                    q, na=False
                )
            filtered_df = filtered_df[mask]

        filtered_df = filtered_df.sort_values(by=sort_col, ascending=False)
        st.caption(
            f"Showing **{len(filtered_df)}** of **{len(customers_df)}** customer records."
        )
        st.dataframe(filtered_df.head(50), use_container_width=True, hide_index=True)

        st.markdown("#### 🎯 Load a Specific Customer into the Real-Time Simulator")
        selected_email = st.selectbox(
            "Select Customer Email to Inspect & Load",
            options=filtered_df["Email"].head(100).tolist()
            if not filtered_df.empty
            else customers_df["Email"].head(50).tolist(),
        )
        if selected_email:
            cust_row = customers_df[customers_df["Email"] == selected_email].iloc[0]
            cust_pred = predict_clv(
                float(cust_row["Avg. Session Length"]),
                float(cust_row["Time on App"]),
                float(cust_row["Time on Website"]),
                float(cust_row["Length of Membership"]),
            )
            cust_actual = float(cust_row["Yearly Amount Spent"])
            cust_err = cust_actual - cust_pred

            ic1, ic2, ic3, ic4 = st.columns(4)
            ic1.metric("Actual Yearly Spend", f"${cust_actual:,.2f}")
            ic2.metric("Model Predicted CLV", f"${cust_pred:,.2f}")
            ic3.metric("Residual Error", f"${cust_err:+,.2f}")
            with ic4:
                st.write("")
                st.button(
                    "🚀 Load into Simulator Sliders",
                    type="primary",
                    use_container_width=True,
                    on_click=apply_preset,
                    args=(
                        f"Customer: {selected_email}",
                        round(float(cust_row["Avg. Session Length"]), 1),
                        round(float(cust_row["Time on App"]), 2),
                        round(float(cust_row["Time on Website"]), 1),
                        round(float(cust_row["Length of Membership"]), 1),
                    ),
                )


# =============================================================================
# TAB 4: MODEL DIAGNOSTICS & EDA INSIGHTS
# =============================================================================
with tab_diagnostics:
    st.subheader("🔬 OLS Regression Diagnostics & Holdout Test Evaluation")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("R² Variance Explained", f"{meta_data.get('r2', 0.9869)*100:.2f}%")
    d2.metric("Mean Absolute Error (MAE)", f"${meta_data.get('mae', 7.22):.2f}")
    d3.metric("Root Mean Squared Error", f"${meta_data.get('rmse', 9.05):.2f}")
    d4.metric("Cleaned Training Cohort", f"{meta_data.get('clean_rows', 489)} records")

    if meta_data and "test_actual_vs_pred" in meta_data:
        test_df = pd.DataFrame(meta_data["test_actual_vs_pred"])
        diag_col1, diag_col2 = st.columns(2)

        with diag_col1:
            st.markdown("##### 🎯 Holdout Test Set: Actual vs. Predicted CLV")
            fig_ap, ax_ap = plt.subplots(figsize=(5.5, 3.6), dpi=120)
            fig_ap.patch.set_facecolor("#0f172a")
            ax_ap.set_facecolor("#0f172a")
            ax_ap.scatter(
                test_df["actual"],
                test_df["predicted"],
                color="#38bdf8",
                alpha=0.8,
                edgecolors="#0284c7",
                s=45,
            )
            lims = [
                min(test_df["actual"].min(), test_df["predicted"].min()) - 15,
                max(test_df["actual"].max(), test_df["predicted"].max()) + 15,
            ]
            ax_ap.plot(lims, lims, "--", color="#10b981", linewidth=1.8, label="Ideal Fit (y = x)")
            ax_ap.set_xlabel("Actual Yearly Spend ($)", color="#94a3b8")
            ax_ap.set_ylabel("Predicted Yearly Spend ($)", color="#94a3b8")
            ax_ap.tick_params(colors="#94a3b8")
            ax_ap.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc")
            for spine in ["top", "right"]:
                ax_ap.spines[spine].set_visible(False)
            ax_ap.spines["bottom"].set_color("#334155")
            ax_ap.spines["left"].set_color("#334155")
            plt.tight_layout()
            st.pyplot(fig_ap, use_container_width=True)
            plt.close(fig_ap)

        with diag_col2:
            st.markdown("##### 📊 Residual Error Distribution (Actual − Predicted)")
            fig_res, ax_res = plt.subplots(figsize=(5.5, 3.6), dpi=120)
            fig_res.patch.set_facecolor("#0f172a")
            ax_res.set_facecolor("#0f172a")
            ax_res.hist(
                test_df["residual"],
                bins=16,
                color="#10b981",
                edgecolor="#064e3b",
                alpha=0.85,
            )
            ax_res.axvline(0, color="#f43f5e", linestyle="--", linewidth=1.8)
            ax_res.set_xlabel("Prediction Residual ($)", color="#94a3b8")
            ax_res.set_ylabel("Customer Count", color="#94a3b8")
            ax_res.tick_params(colors="#94a3b8")
            for spine in ["top", "right"]:
                ax_res.spines[spine].set_visible(False)
            ax_res.spines["bottom"].set_color("#334155")
            ax_res.spines["left"].set_color("#334155")
            plt.tight_layout()
            st.pyplot(fig_res, use_container_width=True)
            plt.close(fig_res)

    with st.expander("🖼️ View Exported EDA Pipeline Charts", expanded=False):
        img_cols = st.columns(2)
        if os.path.exists("eda_target_distribution.png"):
            img_cols[0].image(
                "eda_target_distribution.png",
                caption="Target Spend Distribution (Before & After IQR Filtering)",
                use_container_width=True,
            )
        if os.path.exists("eda_correlation_heatmap.png"):
            img_cols[1].image(
                "eda_correlation_heatmap.png",
                caption="Feature Correlation Matrix Heatmap",
                use_container_width=True,
            )


# =============================================================================
# TAB 5: SAVED SCENARIOS & USER PROFILE WORKSPACE
# =============================================================================
with tab_profile:
    st.subheader("🔖 Saved Simulation Scenarios & Analyst Workspace")
    st.write(
        f"**Active Analyst:** {st.session_state.user_name} ({st.session_state.user_role}) • **Session Started:** {st.session_state.login_time}"
    )

    if not st.session_state.saved_scenarios:
        st.info(
            "No scenarios bookmarked yet. Use the **📌 Bookmark Scenario** button in the **Real-Time Simulator** tab to save and compare customer configurations here."
        )
    else:
        saved_df = pd.DataFrame(st.session_state.saved_scenarios)
        st.dataframe(saved_df, use_container_width=True, hide_index=True)

        exp_col1, exp_col2 = st.columns([1, 1])
        with exp_col1:
            csv_bytes = saved_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Bookmarked Scenarios (CSV)",
                data=csv_bytes,
                file_name="clv_bookmarked_scenarios.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary",
            )
        with exp_col2:
            if st.button("🗑️ Clear All Saved Scenarios", use_container_width=True):
                st.session_state.saved_scenarios = []
                st.rerun()
