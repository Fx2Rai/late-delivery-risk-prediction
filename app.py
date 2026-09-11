"""
==============================================================================
Streamlit Dashboard — Late Delivery Risk Prediction
APL Logistics (KWE Group)
==============================================================================
Run with:  streamlit run app.py
==============================================================================
"""

import os
import io
import json
import joblib
import requests
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from PIL import Image
import shap
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="APL Logistics — Delivery Risk Dashboard",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — premium dark theme
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e8eaf6;
  }

  /* ── Sidebar base ────────────────────────────────── */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b27 100%);
    border-right: 1px solid rgba(124,58,237,0.25);
  }
  section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

  /* Brand strip */
  .sidebar-brand {
    background: linear-gradient(135deg, rgba(124,58,237,0.2), rgba(37,99,235,0.12));
    border: 1px solid rgba(124,58,237,0.3);
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 4px;
    text-align: center;
  }
  .sidebar-brand-title {
    font-size: 1.05rem; font-weight: 700;
    color: #a78bfa !important; letter-spacing: 0.03em;
  }
  .sidebar-brand-sub {
    font-size: 0.72rem; color: #64748b !important; margin-top: 2px;
  }

  /* Filter section headers */
  .filter-section-title {
    font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.12em; color: #64748b !important;
    margin: 18px 0 6px 2px;
  }

  /* Filter label rows */
  .filter-label-row {
    display: flex; align-items: center;
    justify-content: space-between; margin-bottom: 4px;
  }
  .filter-label {
    font-size: 0.80rem; font-weight: 600;
    color: #94a3b8 !important; display: flex; align-items: center; gap: 5px;
  }
  .filter-count-pill {
    font-size: 0.65rem; font-weight: 700;
    background: rgba(124,58,237,0.25); color: #a78bfa !important;
    border: 1px solid rgba(124,58,237,0.4);
    border-radius: 20px; padding: 1px 8px; min-width: 24px; text-align: center;
  }

  /* Multiselect tags — purple theme */
  section[data-testid="stSidebar"] [data-baseweb="tag"] {
    background: rgba(124,58,237,0.35) !important;
    border: 1px solid rgba(124,58,237,0.55) !important;
    border-radius: 8px !important;
  }
  section[data-testid="stSidebar"] [data-baseweb="tag"] span {
    color: #ddd6fe !important; font-size: 0.73rem !important; font-weight: 500 !important;
  }
  section[data-testid="stSidebar"] [data-baseweb="tag"] [role="button"] svg {
    fill: #a78bfa !important;
  }
  section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
  }
  section[data-testid="stSidebar"] [data-baseweb="select"] > div:focus-within {
    border-color: rgba(124,58,237,0.6) !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.15) !important;
  }

  /* Slider section card */
  .slider-section {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 12px 14px; margin-top: 4px;
  }

  /* Risk zone legend */
  .risk-zone-legend { display: flex; gap: 5px; margin-top: 8px; flex-wrap: wrap; }
  .rzl-low    { font-size:0.63rem; padding:2px 8px; border-radius:20px; background:rgba(34,197,94,0.15);  border:1px solid rgba(34,197,94,0.35);  color:#4ade80 !important; font-weight:600; }
  .rzl-medium { font-size:0.63rem; padding:2px 8px; border-radius:20px; background:rgba(245,158,11,0.15); border:1px solid rgba(245,158,11,0.35); color:#fbbf24 !important; font-weight:600; }
  .rzl-high   { font-size:0.63rem; padding:2px 8px; border-radius:20px; background:rgba(239,68,68,0.15);  border:1px solid rgba(239,68,68,0.35);  color:#f87171 !important; font-weight:600; }

  /* Active filter summary card */
  .active-filter-card {
    background: linear-gradient(135deg, rgba(124,58,237,0.1), rgba(37,99,235,0.06));
    border: 1px solid rgba(124,58,237,0.22);
    border-radius: 12px; padding: 10px 12px; margin-top: 14px;
  }
  .active-filter-card-title {
    font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; color: #7c3aed !important; margin-bottom: 7px;
  }
  .afc-row {
    display: flex; justify-content: space-between;
    font-size: 0.73rem; padding: 3px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
  }
  .afc-row:last-child { border-bottom: none; }
  .afc-key { color: #64748b !important; }
  .afc-val { color: #a78bfa !important; font-weight: 600; }

  /* About card */
  .about-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px; padding: 10px 12px;
    font-size: 0.75rem; color: #64748b !important;
    line-height: 1.6; margin-top: 4px;
  }

  /* KPI metric cards */
  .kpi-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px; padding: 22px 20px; text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }
  .kpi-card:hover { transform: translateY(-4px); box-shadow: 0 12px 32px rgba(0,0,0,0.4); }
  .kpi-value {
    font-size: 2.4rem; font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1.2;
  }
  .kpi-label {
    font-size: 0.82rem; color: #94a3b8; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.08em; margin-top: 6px;
  }

  .section-header {
    font-size: 1.4rem; font-weight: 700; color: #e2e8f0;
    margin: 1.5rem 0 1rem 0; padding-bottom: 0.5rem;
    border-bottom: 2px solid rgba(167,139,250,0.4);
  }

  .badge-high   { background:#ef4444; color:#fff; border-radius:8px; padding:3px 10px; font-size:0.78rem; font-weight:600; }
  .badge-medium { background:#f59e0b; color:#fff; border-radius:8px; padding:3px 10px; font-size:0.78rem; font-weight:600; }
  .badge-low    { background:#22c55e; color:#fff; border-radius:8px; padding:3px 10px; font-size:0.78rem; font-weight:600; }

  .result-box {
    background: rgba(255,255,255,0.06); border-radius: 16px;
    padding: 28px; border: 1px solid rgba(255,255,255,0.1); margin-top: 12px;
  }

  .dataframe { background: transparent !important; }

  .stTabs [data-baseweb="tab-list"] {
    gap: 8px; background: rgba(255,255,255,0.04); border-radius: 12px; padding: 6px;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px; color: #94a3b8; font-weight: 500; padding: 8px 20px;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    color: #fff !important;
  }

  .js-plotly-plot { border-radius: 12px; }
  hr { border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH  = "APL_Logistics.csv"
MODEL_DIR  = "models"
FIG_DIR    = "figures"

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.03)",
    font=dict(family="Inter", color="#e2e8f0"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.07)", showline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.07)", showline=False),
    colorway=["#7c3aed","#2563eb","#06b6d4","#10b981","#f59e0b","#ef4444"],
)

RISK_COLORS = {"Low Risk": "#22c55e", "Medium Risk": "#f59e0b", "High Risk": "#ef4444"}


# ──────────────────────────────────────────────────────────────────────────────
# DATA & MODEL LOADING  (cached)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="latin-1")
    df["Customer Lname"].fillna("Unknown", inplace=True)
    df["Customer Zipcode"].fillna(df["Customer Zipcode"].median(), inplace=True)
    return df

@st.cache_data(show_spinner=False)
def get_order_probabilities(df_raw: pd.DataFrame, _pipeline, _region_map):
    """Vectorized scoring of entire dataset (cached for speed)."""
    df_fe = add_features(df_raw, _region_map)
    return _pipeline.predict_proba(df_fe)[:, 1]

@st.cache_resource(show_spinner=False)
def load_models():
    """Load the Group 2 unified pipeline artifacts."""
    # Try Group 2 pipeline first, fall back to Group 1 model
    if os.path.exists(f"{MODEL_DIR}/best_pipeline.pkl"):
        pipeline = joblib.load(f"{MODEL_DIR}/best_pipeline.pkl")
        region_map = joblib.load(f"{MODEL_DIR}/region_risk_map.pkl")
    else:
        pipeline   = joblib.load(f"{MODEL_DIR}/best_model.pkl")
        region_map = {}
    with open(f"{MODEL_DIR}/metrics_summary.json") as f:
        metrics = json.load(f)
    return pipeline, region_map, metrics

def models_available() -> bool:
    return (
        os.path.exists(f"{MODEL_DIR}/best_pipeline.pkl") or
        os.path.exists(f"{MODEL_DIR}/best_model.pkl")
    )

# ──────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  (mirrors group2_ml.py exactly)
# ──────────────────────────────────────────────────────────────────────────────
def add_features(X: pd.DataFrame, region_risk_map: dict, global_mean: float = 0.5483) -> pd.DataFrame:
    """Add all engineered features. Must exactly mirror group2_ml.py."""
    X = X.copy()
    X["shipping_pressure_idx"]   = X["Days for shipment (scheduled)"] / (X["Order Item Quantity"] + 1)
    express = {"First Class", "Same Day"}
    X["mode_risk_flag"]          = X["Shipping Mode"].apply(lambda x: 0 if x in express else 1)
    X["regional_congestion_idx"] = X["Order Region"].map(region_risk_map).fillna(global_mean)
    qty_norm   = X["Order Item Quantity"] / 5.0
    disc_norm  = X["Order Item Discount Rate"]
    sched_norm = X["Days for shipment (scheduled)"] / 4.0
    X["order_complexity_score"]  = (qty_norm + disc_norm + sched_norm) / 3.0
    X["order_value_per_unit"]    = X["Sales"] / (X["Order Item Quantity"] + 1)
    X["profit_margin"]           = X["Order Profit Per Order"] / (X["Sales"] + 1)
    X["high_discount_flag"]      = (X["Order Item Discount Rate"] > 0.10).astype(int)
    X["benefit_per_unit"]        = X["Benefit per order"] / (X["Order Item Quantity"] + 1)
    return X


# ──────────────────────────────────────────────────────────────────────────────
# RISK HELPER
# ──────────────────────────────────────────────────────────────────────────────
def risk_category(prob: float, low=0.40, high=0.70) -> str:
    if prob < low:  return "Low Risk"
    if prob < high: return "Medium Risk"
    return "High Risk"

def badge_html(cat: str) -> str:
    cls = {"Low Risk": "low", "Medium Risk": "medium", "High Risk": "high"}[cat]
    return f'<span class="badge-{cls}">{cat}</span>'


# ──────────────────────────────────────────────────────────────────────────────
# PREDICT  (single order or bulk) using unified Group 2 pipeline
# ──────────────────────────────────────────────────────────────────────────────
def predict_order(order_dict: dict, pipeline, region_map: dict,
                  low_thresh: float, high_thresh: float):
    """Predict late delivery risk for a single order dict."""
    row = pd.DataFrame([order_dict])
    row = add_features(row, region_map)
    prob = pipeline.predict_proba(row)[0][1]
    cat  = risk_category(prob, low_thresh, high_thresh)
    return prob, cat


def predict_orders_bulk(df_sample: pd.DataFrame, pipeline, region_map: dict,
                        low_thresh: float, high_thresh: float):
    """Predict late delivery risk for a batch DataFrame (fast vectorised)."""
    df = add_features(df_sample.copy(), region_map)
    probs = pipeline.predict_proba(df)[:, 1]
    cats  = [risk_category(p, low_thresh, high_thresh) for p in probs]
    return probs, cats


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
def render_sidebar(df: pd.DataFrame):
    # ── Brand ─────────────────────────────────────────────────────────────────
    st.sidebar.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-title">🚢 APL Logistics</div>
        <div class="sidebar-brand-sub">Delivery Risk Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Section header ─────────────────────────────────────────────────────────
    st.sidebar.markdown(
        '<div class="filter-section-title">⚙&nbsp;&nbsp;Global Filters</div>',
        unsafe_allow_html=True
    )

    # ── Shipping Mode ──────────────────────────────────────────────────────────
    mode_options = sorted(df["Shipping Mode"].unique().tolist())
    st.sidebar.markdown(
        f'<div class="filter-label-row">'
        f'<span class="filter-label">🚚 Shipping Mode</span>'
        f'<span class="filter-count-pill">{len(mode_options)}</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    sel_mode = st.sidebar.multiselect(
        "", options=mode_options, default=mode_options, key="mode_filter",
        label_visibility="collapsed"
    )

    # ── Market ─────────────────────────────────────────────────────────────────
    markets = sorted(df["Market"].unique().tolist())
    st.sidebar.markdown(
        f'<div class="filter-label-row">'
        f'<span class="filter-label">🌍 Market</span>'
        f'<span class="filter-count-pill">{len(markets)}</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    sel_market = st.sidebar.multiselect(
        "", options=markets, default=markets, key="market_filter",
        label_visibility="collapsed"
    )

    # ── Customer Segment ───────────────────────────────────────────────────────
    segments = sorted(df["Customer Segment"].unique().tolist())
    st.sidebar.markdown(
        f'<div class="filter-label-row">'
        f'<span class="filter-label">👤 Customer Segment</span>'
        f'<span class="filter-count-pill">{len(segments)}</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    sel_segment = st.sidebar.multiselect(
        "", options=segments, default=segments, key="seg_filter",
        label_visibility="collapsed"
    )

    # ── Risk Threshold ─────────────────────────────────────────────────────────
    st.sidebar.markdown(
        '<div class="filter-section-title">🎯&nbsp;&nbsp;Risk Threshold</div>',
        unsafe_allow_html=True
    )
    st.sidebar.markdown('<div class="slider-section">', unsafe_allow_html=True)
    risk_thresh = st.sidebar.slider(
        "High-Risk cutoff (≥)", 0.40, 1.0, 0.70, 0.05, key="risk_thresh",
        help="Orders at or above this probability are flagged High Risk. Low Risk is always fixed at < 40%."
    )
    med_upper = int(risk_thresh * 100)
    st.sidebar.markdown(
        f'<div class="risk-zone-legend">'
        f'<span class="rzl-low">🟢 Low &lt;40%</span>'
        f'<span class="rzl-medium">🟡 Med 40–{med_upper}%</span>'
        f'<span class="rzl-high">🔴 High ≥{med_upper}%</span>'
        f'</div></div>',
        unsafe_allow_html=True
    )

    # ── Active Filter Summary Card ─────────────────────────────────────────────
    def fmt(n, total):
        return f"All ({total})" if n >= total else f"{n} / {total}"

    st.sidebar.markdown(
        f"""
        <div class="active-filter-card">
            <div class="active-filter-card-title">📋 Active Filters</div>
            <div class="afc-row">
                <span class="afc-key">Shipping Modes</span>
                <span class="afc-val">{fmt(len(sel_mode), len(mode_options))}</span>
            </div>
            <div class="afc-row">
                <span class="afc-key">Markets</span>
                <span class="afc-val">{fmt(len(sel_market), len(markets))}</span>
            </div>
            <div class="afc-row">
                <span class="afc-key">Segments</span>
                <span class="afc-val">{fmt(len(sel_segment), len(segments))}</span>
            </div>
            <div class="afc-row">
                <span class="afc-key">High-Risk Cutoff</span>
                <span class="afc-val">&ge; {risk_thresh:.0%}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ── About ──────────────────────────────────────────────────────────────────
    st.sidebar.markdown(
        '<div class="filter-section-title">ℹ️&nbsp;&nbsp;About</div>',
        unsafe_allow_html=True
    )
    st.sidebar.markdown(
        '<div class="about-card">'
        'Predictive ML system identifying delivery risk <b>before</b> shipment '
        'using order features, shipping mode, region, and product data. '
        'Powered by XGBoost + SHAP explainability.'
        '</div>',
        unsafe_allow_html=True
    )

    return sel_mode, sel_market, sel_segment, risk_thresh



# ──────────────────────────────────────────────────────────────────────────────
# MODULE 1 — DELAY RISK OVERVIEW
# ──────────────────────────────────────────────────────────────────────────────
def module_overview(df: pd.DataFrame, risk_thresh: float):
    st.markdown('<div class="section-header">📊 Module 1 — Delay Risk Overview</div>',
                unsafe_allow_html=True)

    # ── 1.1 Predicted Risk Intelligence (ML Model Output) ─────────────────────
    if "Late_Probability" in df.columns:
        probs = df["Late_Probability"].values
        n_total = len(probs)
        
        low_cnt  = int((probs < 0.40).sum())
        med_cnt  = int(((probs >= 0.40) & (probs < risk_thresh)).sum())
        high_cnt = int((probs >= risk_thresh).sum())
        
        low_pct  = (low_cnt / n_total * 100) if n_total > 0 else 0
        med_pct  = (med_cnt / n_total * 100) if n_total > 0 else 0
        high_pct = (high_cnt / n_total * 100) if n_total > 0 else 0
        mean_prob = float(probs.mean()) if n_total > 0 else 0

        st.markdown(f"#### 🤖 Machine Learning Predicted Risk (Threshold: High Risk ≥ {risk_thresh:.0%})")
        
        pk1, pk2, pk3, pk4 = st.columns(4)
        pk1.markdown(
            f'<div class="kpi-card"><div class="kpi-value" style="color:#ef4444;">{high_cnt:,}</div>'
            f'<div class="kpi-label">🔴 High Risk (≥ {risk_thresh:.0%})</div>'
            f'<div style="font-size:0.8rem; color:#94a3b8; margin-top:4px;">{high_pct:.1f}% of filtered orders</div></div>',
            unsafe_allow_html=True
        )
        pk2.markdown(
            f'<div class="kpi-card"><div class="kpi-value" style="color:#f59e0b;">{med_cnt:,}</div>'
            f'<div class="kpi-label">🟡 Medium Risk (40%–{risk_thresh:.0%})</div>'
            f'<div style="font-size:0.8rem; color:#94a3b8; margin-top:4px;">{med_pct:.1f}% of filtered orders</div></div>',
            unsafe_allow_html=True
        )
        pk3.markdown(
            f'<div class="kpi-card"><div class="kpi-value" style="color:#22c55e;">{low_cnt:,}</div>'
            f'<div class="kpi-label">🟢 Low Risk (&lt; 40%)</div>'
            f'<div style="font-size:0.8rem; color:#94a3b8; margin-top:4px;">{low_pct:.1f}% of filtered orders</div></div>',
            unsafe_allow_html=True
        )
        pk4.markdown(
            f'<div class="kpi-card"><div class="kpi-value">{mean_prob:.1%}</div>'
            f'<div class="kpi-label">🎯 Average Risk Probability</div>'
            f'<div style="font-size:0.8rem; color:#94a3b8; margin-top:4px;">Across {n_total:,} orders</div></div>',
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Risk Distribution Charts
        pcol1, pcol2 = st.columns([1, 2])
        with pcol1:
            risk_cat_data = pd.DataFrame({
                "Category": ["Low Risk", "Medium Risk", "High Risk"],
                "Count": [low_cnt, med_cnt, high_cnt]
            })
            fig_pie = px.pie(
                risk_cat_data, names="Category", values="Count",
                title="Predicted Risk Tier Distribution",
                color="Category",
                color_discrete_map=RISK_COLORS,
                hole=0.55
            )
            fig_pie.update_layout(
                **PLOTLY_THEME, height=320,
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
                margin=dict(t=50, b=50, l=20, r=20)
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with pcol2:
            fig_dist = px.histogram(
                df, x="Late_Probability", nbins=40,
                title="Predicted Late Delivery Probability Distribution",
                color_discrete_sequence=["#7c3aed"],
                labels={"Late_Probability": "Predicted Late Delivery Probability"}
            )
            fig_dist.add_vline(x=0.40, line_dash="dash", line_color="#f59e0b",
                               annotation_text="Medium Risk ≥ 40%", annotation_position="top right")
            fig_dist.add_vline(x=risk_thresh, line_dash="dash", line_color="#ef4444",
                               annotation_text=f"High Risk ≥ {risk_thresh:.0%}", annotation_position="top right")
            fig_dist.update_layout(**PLOTLY_THEME, height=320, margin=dict(t=50, b=30, l=40, r=20))
            st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown("---")

    # ── 1.2 Historical Delivery Baseline Performance ─────────────────────────
    st.markdown("#### 📦 Historical Delivery Performance (Actual Outcomes)")
    total_orders   = len(df)
    late_orders    = int(df["Late_delivery_risk"].sum())
    pct_late       = late_orders / total_orders * 100
    pct_on_time    = 100 - pct_late
    cancel_orders  = int((df["Order Status"] == "CANCELED").sum()) if "Order Status" in df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [
        (c1, f"{total_orders:,}", "Total Orders"),
        (c2, f"{late_orders:,}", "Late Delivery Orders"),
        (c3, f"{pct_late:.1f}%", "Late Delivery Rate"),
        (c4, f"{pct_on_time:.1f}%", "On-Time Rate"),
    ]:
        col.markdown(
            f'<div class="kpi-card"><div class="kpi-value">{val}</div>'
            f'<div class="kpi-label">{lbl}</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])

    with col1:
        # Donut chart — target distribution
        counts = df["Late_delivery_risk"].value_counts()
        fig = go.Figure(go.Pie(
            labels=["On-Time", "Late Delivery"],
            values=[counts.get(0, 0), counts.get(1, 0)],
            hole=0.55,
            marker_colors=["#22c55e", "#ef4444"],
            textfont=dict(size=13, color="white"),
            hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>"
        ))
        fig.update_layout(
            title="Delivery Outcome Distribution",
            **PLOTLY_THEME,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            margin=dict(t=50, b=50, l=20, r=20), height=340
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Delivery Status breakdown
        if "Delivery Status" in df.columns:
            status_counts = df["Delivery Status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            colors = {"Late delivery": "#ef4444", "Advance shipping": "#22c55e",
                      "Shipping on time": "#06b6d4", "Shipping canceled": "#f59e0b"}
            status_counts["Color"] = status_counts["Status"].map(colors)
            fig2 = px.bar(
                status_counts, x="Status", y="Count",
                color="Status",
                color_discrete_map=colors,
                title="Delivery Status Breakdown",
                text_auto=True
            )
            fig2.update_layout(**PLOTLY_THEME, showlegend=False,
                               margin=dict(t=50, b=20, l=20, r=20), height=340)
            fig2.update_traces(textfont_size=12, textangle=0)
            st.plotly_chart(fig2, use_container_width=True)

    # Order status distribution
    if "Order Status" in df.columns:
        os_counts = df["Order Status"].value_counts().reset_index()
        os_counts.columns = ["Order Status", "Count"]
        fig3 = px.bar(os_counts, x="Count", y="Order Status", orientation="h",
                      title="Order Status Distribution",
                      color="Count", color_continuous_scale="Viridis",
                      text_auto=True)
        fig3.update_layout(**PLOTLY_THEME, coloraxis_showscale=False,
                           height=350, margin=dict(t=50, b=20, l=20, r=20))
        st.plotly_chart(fig3, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# MODULE 2 — ORDER-LEVEL RISK PREDICTION
# ──────────────────────────────────────────────────────────────────────────────
def module_prediction(df: pd.DataFrame, pipeline, region_map: dict, metrics,
                      low_thresh: float, high_thresh: float):
    st.markdown('<div class="section-header">🔮 Module 2 — Order-Level Risk Prediction</div>',
                unsafe_allow_html=True)

    if not models_available():
        st.warning("⚠️ Models not yet trained. Please run `python ml_pipeline.py` first.")
        return

    # Model performance summary
    best_name = metrics.get("best_model", "XGBoost")
    test_m    = metrics.get("test_metrics", {}).get(best_name, {})
    st.markdown(f"**Active Model:** `{best_name}`")
    mc1, mc2, mc3, mc4 = st.columns(4)
    for col, key, icon in [
        (mc1, "ROC-AUC",  "🎯"),
        (mc2, "Precision","🔬"),
        (mc3, "Recall",   "📡"),
        (mc4, "F1 Score", "⚖️"),
    ]:
        val = test_m.get(key, "N/A")
        col.metric(f"{icon} {key}", f"{val:.4f}" if isinstance(val, float) else val)

    st.markdown("---")
    st.markdown("#### 📝 Enter Order Details to Predict Risk")

    # Input form
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        sched_days = st.number_input("Scheduled Shipping Days", 1, 14, 4)
        quantity   = st.number_input("Order Item Quantity",     1, 50,  3)
        sales      = st.number_input("Sales Amount (USD)",     10.0, 10000.0, 300.0, step=10.0)
        discount   = st.number_input("Item Discount Amount",    0.0,  500.0,  20.0)
    with col_b:
        disc_rate  = st.number_input("Discount Rate (0–1)",   0.0, 1.0, 0.06, step=0.01)
        prod_price = st.number_input("Product Price (USD)",   5.0, 2000.0, 99.99)
        profit     = st.number_input("Order Profit (USD)",  -200.0, 2000.0, 80.0)
        profit_ratio = st.number_input("Profit Ratio (0–1)", -1.0, 1.0, 0.30, step=0.01)
    with col_c:
        ship_mode   = st.selectbox("Shipping Mode",   sorted(df["Shipping Mode"].unique()))
        market      = st.selectbox("Market",          sorted(df["Market"].unique()))
        region      = st.selectbox("Order Region",    sorted(df["Order Region"].unique()))
        order_type  = st.selectbox("Payment Type",    sorted(df["Type"].unique()))
        customer_seg= st.selectbox("Customer Segment",sorted(df["Customer Segment"].unique()))
        dept        = st.selectbox("Department",      sorted(df["Department Name"].unique()))
        category    = st.selectbox("Category Name",   sorted(df["Category Name"].unique()))

    predict_btn = st.button("🚀 Predict Delivery Risk", type="primary", use_container_width=True)

    if predict_btn:
        order_dict = {
            "Type": order_type,
            "Days for shipment (scheduled)": sched_days,
            "Benefit per order": profit,
            "Sales per customer": sales,
            "Category Id": 1,
            "Category Name": category,
            "Customer Segment": customer_seg,
            "Customer State": "TX",
            "Department Id": 1,
            "Department Name": dept,
            "Latitude": 25.0,
            "Longitude": -97.0,
            "Market": market,
            "Order Item Discount": discount,
            "Order Item Discount Rate": disc_rate,
            "Order Item Product Price": prod_price,
            "Order Item Profit Ratio": profit_ratio,
            "Order Item Quantity": quantity,
            "Sales": sales,
            "Order Item Total": sales - discount,
            "Order Profit Per Order": profit,
            "Order Region": region,
            "Order Status": "COMPLETE",
            "Product Price": prod_price,
            "Shipping Mode": ship_mode,
        }

        prob, cat = predict_order(order_dict, pipeline, region_map,
                                  low_thresh, high_thresh)
        color = RISK_COLORS[cat]

        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        r1, r2 = st.columns([1, 2])
        with r1:
            st.markdown(f"### Risk Score: `{prob:.1%}`")
            st.markdown(
                f"**Category:** {badge_html(cat)}",
                unsafe_allow_html=True
            )
            if cat == "High Risk":
                st.error("⚠️ Immediate attention required!")
            elif cat == "Medium Risk":
                st.warning("🔶 Monitor this order closely.")
            else:
                st.success("✅ Delivery expected on time.")
        with r2:
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Late Delivery Probability (%)", "font": {"size": 14, "color": "#e2e8f0"}},
                gauge={
                    "axis": {"range": [0, 100], "tickfont": {"color": "#94a3b8"}},
                    "bar": {"color": color, "thickness": 0.3},
                    "steps": [
                        {"range": [0,  low_thresh * 100], "color": "rgba(34,197,94,0.15)"},
                        {"range": [low_thresh * 100, high_thresh * 100], "color": "rgba(245,158,11,0.15)"},
                        {"range": [high_thresh * 100, 100],"color": "rgba(239,68,68,0.15)"},
                    ],
                    "threshold": {
                        "line": {"color": color, "width": 4},
                        "thickness": 0.8,
                        "value": prob * 100,
                    },
                },
                number={"suffix": "%", "font": {"size": 40, "color": color}},
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
                height=240, margin=dict(t=30, b=10, l=20, r=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── 14. Individual Explainability (SHAP) ─────────────────────────────
        st.markdown("#### 🔍 Why was this prediction made? (SHAP Analysis)")
        try:
            # Reconstruct the exact feature row
            row = pd.DataFrame([order_dict])
            row_fe = add_features(row, region_map)
            
            # Extract components
            preprocessor = pipeline.named_steps["preprocessor"]
            model = pipeline.named_steps["model"]
            
            # Transform
            X_transformed = preprocessor.transform(row_fe)
            
            # Compute SHAP
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_transformed)
            
            # We want an Explanation object for the waterfall plot
            # Since xgboost returns raw values, we construct it
            base_value = explainer.expected_value
            
            # SHAP Waterfall Plot
            fig_shap, ax = plt.subplots(figsize=(8, 4))
            # We use the legacy or the new shap decision plot/waterfall
            # shap.waterfall requires Explanation object which is tricky to construct manually for some versions,
            # Let's use a standard horizontal bar chart of local impacts for robustness in Streamlit
            
            impacts = pd.DataFrame({
                "Feature": metrics.get("feature_names", []),
                "Impact": shap_values[0]
            })
            impacts["Abs_Impact"] = impacts["Impact"].abs()
            impacts = impacts.sort_values("Abs_Impact", ascending=False).head(10)
            
            # Color based on pushing risk Up (Red) or Down (Green)
            impacts["Color"] = impacts["Impact"].apply(lambda x: "#ef4444" if x > 0 else "#22c55e")
            
            fig_bar = px.bar(
                impacts, x="Impact", y="Feature", orientation="h",
                title="Top Risk Drivers for this Specific Order",
                color="Color", color_discrete_map={"#ef4444": "#ef4444", "#22c55e": "#22c55e"},
                labels={"Impact": "Impact on Risk Probability (Log-Odds)"}
            )
            fig_bar.update_layout(**PLOTLY_THEME, showlegend=False, 
                                  height=380, margin=dict(t=50, b=20, l=20, r=20))
            st.plotly_chart(fig_bar, use_container_width=True)
            
        except Exception as e:
            st.error(f"Could not generate SHAP explanation: {e}")
            
        # ──────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# MODULE 3 — REGION & MODE RISK ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
def module_region_mode(df: pd.DataFrame):
    st.markdown('<div class="section-header">🌍 Module 3 — Region & Mode Risk Analysis</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # By Market
        market_risk = (
            df.groupby("Market")["Late_delivery_risk"].mean()
            .sort_values(ascending=True)
            .reset_index()
        )
        market_risk.columns = ["Market", "Late Rate"]
        market_risk["Pct"] = (market_risk["Late Rate"] * 100).round(1)
        fig = px.bar(
            market_risk, x="Pct", y="Market", orientation="h",
            title="Late Delivery Rate by Market",
            color="Pct",
            color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"],
            text="Pct", labels={"Pct": "Late Rate (%)"}
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                          textfont_color="white")
        fig.update_layout(**PLOTLY_THEME, coloraxis_showscale=False,
                          height=360, margin=dict(t=50, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # By Shipping Mode
        mode_risk = (
            df.groupby("Shipping Mode")["Late_delivery_risk"].mean()
            .sort_values(ascending=True)
            .reset_index()
        )
        mode_risk.columns = ["Shipping Mode", "Late Rate"]
        mode_risk["Pct"] = (mode_risk["Late Rate"] * 100).round(1)
        fig2 = px.bar(
            mode_risk, x="Pct", y="Shipping Mode", orientation="h",
            title="Late Delivery Rate by Shipping Mode",
            color="Pct",
            color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"],
            text="Pct", labels={"Pct": "Late Rate (%)"}
        )
        fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                           textfont_color="white")
        fig2.update_layout(**PLOTLY_THEME, coloraxis_showscale=False,
                           height=360, margin=dict(t=50, b=20, l=20, r=20))
        st.plotly_chart(fig2, use_container_width=True)

    # By Order Region (top 15)
    region_risk = (
        df.groupby("Order Region")["Late_delivery_risk"]
        .agg(["mean", "count"])
        .reset_index()
    )
    region_risk.columns = ["Region", "Late Rate", "Order Count"]
    region_risk["Late Rate %"] = (region_risk["Late Rate"] * 100).round(1)
    region_risk = region_risk.sort_values("Late Rate", ascending=False).head(20)

    fig3 = px.scatter(
        region_risk, x="Late Rate %", y="Order Count",
        size="Order Count", color="Late Rate %",
        color_continuous_scale=["#22c55e","#f59e0b","#ef4444"],
        hover_name="Region", text="Region",
        title="Regions: Late Delivery Rate vs Order Volume",
        labels={"Late Rate %": "Late Rate (%)", "Order Count": "Number of Orders"}
    )
    fig3.update_traces(textposition="top center", textfont_size=10)
    fig3.update_layout(**PLOTLY_THEME, coloraxis_showscale=True,
                       height=480, margin=dict(t=60, b=40, l=60, r=20))
    st.plotly_chart(fig3, use_container_width=True)

    # Customer Segment risk
    col3, col4 = st.columns(2)
    with col3:
        seg = df.groupby("Customer Segment")["Late_delivery_risk"].mean().reset_index()
        seg.columns = ["Segment", "Late Rate"]
        seg["Pct"] = (seg["Late Rate"] * 100).round(1)
        fig4 = px.pie(
            seg, names="Segment", values="Pct",
            title="Late Rate by Customer Segment",
            color_discrete_sequence=["#7c3aed","#2563eb","#06b6d4"],
            hole=0.45
        )
        fig4.update_layout(**PLOTLY_THEME, height=320,
                           margin=dict(t=50, b=20, l=20, r=20))
        st.plotly_chart(fig4, use_container_width=True)

    with col4:
        # Department risk
        dept_risk = (
            df.groupby("Department Name")["Late_delivery_risk"].mean()
            .sort_values(ascending=False).head(10).reset_index()
        )
        dept_risk.columns = ["Department", "Late Rate"]
        dept_risk["Pct"] = (dept_risk["Late Rate"] * 100).round(1)
        fig5 = px.bar(
            dept_risk, x="Department", y="Pct",
            title="Late Rate by Department (Top 10)",
            color="Pct",
            color_continuous_scale=["#22c55e","#f59e0b","#ef4444"],
            labels={"Pct": "Late Rate (%)"}
        )
        fig5.update_layout(**PLOTLY_THEME, coloraxis_showscale=False,
                           xaxis_tickangle=-35, height=320,
                           margin=dict(t=50, b=60, l=40, r=20))
        st.plotly_chart(fig5, use_container_width=True)

    # Regional Risk Heatmap (Part 18 spec requirement)
    st.markdown("#### 🗺️ Regional Risk Heatmap")
    # Heatmap of Region vs Shipping Mode
    heatmap_data = df.groupby(["Order Region", "Shipping Mode"])["Late_delivery_risk"].mean().reset_index()
    heatmap_data["Late_delivery_risk"] = heatmap_data["Late_delivery_risk"] * 100
    
    fig_heat = px.density_heatmap(
        heatmap_data, x="Shipping Mode", y="Order Region", z="Late_delivery_risk", 
        histfunc="avg", text_auto=".1f",
        title="Average Late Rate (%) by Region and Shipping Mode",
        color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"]
    )
    fig_heat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#e2e8f0"),
        height=550, margin=dict(t=50, b=20, l=20, r=20)
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# MODULE 4 — OPERATIONS ACTION PANEL
# ──────────────────────────────────────────────────────────────────────────────
def module_action_panel(df: pd.DataFrame, sel_mode, sel_market, sel_segment,
                        risk_thresh: float, pipeline, region_map: dict):
    st.markdown('<div class="section-header">⚡ Module 4 — Operations Action Panel</div>',
                unsafe_allow_html=True)

    if not models_available():
        st.warning("⚠️ Models not yet trained. Run `python ml_pipeline.py` first.")
        return

    # Compute risk scores on the full operational queue
    queue = df.copy()
    if queue.empty:
        st.warning("No orders match the selected filters.")
        return

    low_thresh = 0.40
    high_thresh = risk_thresh

    if "Late_Probability" in queue.columns:
        probs = queue["Late_Probability"].values
        cats = [risk_category(p, low_thresh, high_thresh) for p in probs]
    else:
        with st.spinner(f"Scoring full operational queue ({len(queue):,} orders)..."):
            try:
                probs, cats = predict_orders_bulk(queue, pipeline, region_map,
                                                  low_thresh, high_thresh)
            except Exception as e:
                probs = [0.5] * len(queue)
                cats = ["Medium Risk"] * len(queue)

    queue["Risk Probability"] = probs
    queue["Risk Category"]    = cats

    high_risk = queue[queue["Risk Category"] == "High Risk"].copy()
    high_risk = high_risk.sort_values("Risk Probability", ascending=False)

    # Summary KPIs
    k1, k2, k3, k4 = st.columns(4)
    for col, val, lbl in [
        (k1, f"{len(queue):,}", "Orders Analyzed (Full Queue)"),
        (k2, f"{len(high_risk):,}", "🔴 High-Risk Orders"),
        (k3, f"{len(queue[queue['Risk Category']=='Medium Risk']):,}", "🟡 Medium-Risk"),
        (k4, f"{len(queue[queue['Risk Category']=='Low Risk']):,}", "🟢 Low-Risk"),
    ]:
        col.markdown(
            f'<div class="kpi-card"><div class="kpi-value">{val}</div>'
            f'<div class="kpi-label">{lbl}</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Risk category distribution
    col_pie, col_bar = st.columns([1, 2])
    with col_pie:
        cat_counts = queue["Risk Category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig = px.pie(
            cat_counts, names="Category", values="Count",
            title="Risk Distribution (Operational Queue)",
            color="Category",
            color_discrete_map=RISK_COLORS,
            hole=0.5
        )
        fig.update_layout(**PLOTLY_THEME, height=320,
                          margin=dict(t=50, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_bar:
        # Risk by shipping mode in full queue
        mode_cat = (
            queue.groupby(["Shipping Mode","Risk Category"]).size()
            .reset_index(name="Count")
        )
        fig2 = px.bar(
            mode_cat, x="Shipping Mode", y="Count", color="Risk Category",
            title="Risk Categories by Shipping Mode (Queue)",
            color_discrete_map=RISK_COLORS, barmode="stack"
        )
        fig2.update_layout(**PLOTLY_THEME, height=320,
                           margin=dict(t=50, b=30, l=40, r=20))
        st.plotly_chart(fig2, use_container_width=True)

    # High-risk order table
    st.markdown(f"#### 🔴 High-Risk Orders Requiring Immediate Attention ({len(high_risk):,} orders)")

    display_cols = [c for c in [
        "Order Id", "Risk Probability", "Risk Category", "Order Region", 
        "Shipping Mode", "Market", "Sales"
    ] if c in high_risk.columns]
    
    # If Order Id isn't present, use the index
    if "Order Id" not in display_cols:
        high_risk["Order"] = "ORD-" + high_risk.index.astype(str)
        display_cols.insert(0, "Order")

    # Add Action column
    high_risk["Action"] = "⚠️ Expedite Shipping"

    # Match spec exactly: Order, Risk Probability, Category, Region, Shipping Mode, Action
    final_cols = ["Order", "Risk Probability", "Risk Category", "Order Region", "Shipping Mode", "Action"]
    # Fallback to available columns if renaming fails
    final_cols = [c for c in final_cols if c in high_risk.columns]

    display_df = high_risk[final_cols].copy()
    display_df["Risk Probability"] = display_df["Risk Probability"].map(lambda x: f"{x:.1%}")

    st.dataframe(
        display_df.head(50),
        use_container_width=True,
        height=400,
    )

    # CSV Download
    csv_data = high_risk[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download High-Risk Orders (CSV)",
        data=csv_data,
        file_name="high_risk_orders.csv",
        mime="text/csv",
        type="primary"
    )

    # Risk probability histogram
    fig3 = px.histogram(
        queue, x="Risk Probability",
        nbins=40,
        title="Risk Probability Distribution Across Operational Queue",
        color_discrete_sequence=["#7c3aed"],
        labels={"Risk Probability": "Predicted Late Delivery Probability"}
    )
    fig3.add_vline(x=0.40, line_dash="dash", line_color="#f59e0b",
                   annotation_text="Medium Risk ≥ 40%", annotation_position="top right")
    fig3.add_vline(x=0.70, line_dash="dash", line_color="#ef4444",
                   annotation_text="High Risk ≥ 70%", annotation_position="top right")
    fig3.update_layout(**PLOTLY_THEME, height=360,
                       margin=dict(t=60, b=40, l=60, r=20))
    st.plotly_chart(fig3, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 2rem 0 1.5rem 0;">
      <h1 style="font-size:2.2rem; font-weight:800;
                 background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
                 -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                 margin-bottom: 0.3rem;">
        🚢 APL Logistics — Delivery Risk Intelligence
      </h1>
      <p style="color:#94a3b8; font-size:1rem; margin:0;">
        Machine Learning–based Late Delivery Risk Prediction in Global Supply Chain Operations
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("Loading logistics data..."):
        df = load_data()

    # ── Load models ───────────────────────────────────────────────────────────
    model_loaded = models_available()
    if model_loaded:
        pipeline, region_map, metrics = load_models()
        if "Late_Probability" not in df.columns:
            with st.spinner("Computing predictive risk scores across supply chain..."):
                df["Late_Probability"] = get_order_probabilities(df, pipeline, region_map)
    else:
        pipeline = region_map = metrics = None
        st.warning(
            "⚠️ **Models not trained yet.** Run `python ml_pipeline.py` to train models, "
            "then refresh this page. Modules 2 & 4 will be limited until models are available."
        )

    # ── Sidebar filters ───────────────────────────────────────────────────────
    sel_mode, sel_market, sel_segment, risk_thresh = render_sidebar(df)

    # Apply filters for downstream modules
    df_filtered = df.copy()
    if sel_mode:
        df_filtered = df_filtered[df_filtered["Shipping Mode"].isin(sel_mode)]
    if sel_market:
        df_filtered = df_filtered[df_filtered["Market"].isin(sel_market)]
    if sel_segment:
        df_filtered = df_filtered[df_filtered["Customer Segment"].isin(sel_segment)]

    if df_filtered.empty:
        st.error("⚠️ No data matches the current filters. Please adjust the sidebar selections.")
        return

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📊 Risk Overview",
        "🔮 Order Prediction",
        "🌍 Region & Mode Analysis",
        "⚡ Action Panel",
    ])

    with tabs[0]:
        module_overview(df_filtered, risk_thresh)

    with tabs[1]:
        if model_loaded:
            module_prediction(df_filtered, pipeline, region_map, metrics,
                              0.40, risk_thresh)
        else:
            st.warning("Train models first: `python ml_pipeline.py`")

    with tabs[2]:
        module_region_mode(df_filtered)

    with tabs[3]:
        if model_loaded:
            module_action_panel(df_filtered, sel_mode, sel_market, sel_segment,
                                risk_thresh, pipeline, region_map)
        else:
            st.warning("Train models first: `python ml_pipeline.py`")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center; color:#475569; font-size:0.8rem;">'
        'APL Logistics (KWE Group) · Delivery Risk Intelligence Platform · '
        'Built with Streamlit + Scikit-Learn + XGBoost'
        '</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
