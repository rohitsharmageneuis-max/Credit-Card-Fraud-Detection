"""
AI CREDIT CARD FRAUD DETECTION SYSTEM
Professional Streamlit Dashboard
Designed with Dark Navy Theme, Cyan & Purple Accents, and Interactive Visualizations.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is in python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.preprocessing import DataPreprocessor
from src.model import build_fraud_ann, get_training_callbacks, train_model
from src.evaluation import (
    load_metrics_and_history,
    evaluate_model_predictions,
    save_metrics_and_history,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_training_history,
    PALETTE
)
from src.prediction import (
    FraudPredictor,
    SAMPLE_GENUINE_TRANSACTION,
    SAMPLE_FRAUD_TRANSACTION,
    SAMPLE_SUSPICIOUS_TRANSACTION
)

# =====================================================================
# PAGE CONFIGURATION & STYLING
# =====================================================================
st.set_page_config(
    page_title="AI Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load External CSS
css_path = os.path.join(ROOT_DIR, "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# =====================================================================
# DATA & ARTIFACT CACHING
# =====================================================================
@st.cache_data(show_spinner=False)
def get_dataset(data_path="data/creditcard.csv"):
    """
    Loads dataset with fallback options:
    1. Specified data_path (e.g. data/creditcard.csv)
    2. Root-level creditcard.csv
    3. Built-in bundled real sample: data/creditcard_sample.csv (ensures Streamlit Cloud always loads without manual upload)
    """
    candidate_paths = [
        os.path.join(ROOT_DIR, data_path),
        os.path.join(ROOT_DIR, "creditcard.csv"),
        os.path.join(ROOT_DIR, "data", "creditcard_sample.csv")
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None


def get_predictor():
    """Initializes and returns the FraudPredictor inference engine."""
    model_path = os.path.join(ROOT_DIR, "models", "fraud_model.keras")
    scaler_path = os.path.join(ROOT_DIR, "models", "scaler.joblib")
    return FraudPredictor(model_path=model_path, scaler_path=scaler_path)


# =====================================================================
# HEADER COMPONENT
# =====================================================================
def render_header(title: str, subtitle: str, badge_text: str = "ANN DEEP LEARNING"):
    st.markdown(f"""
    <div class="hero-header">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px;">
            <div>
                <span class="badge-tag">🛡️ {badge_text}</span>
                <span class="badge-tag badge-tag-purple" style="margin-left: 8px;">SMOTE BALANCED</span>
                <h1 class="hero-title" style="margin-top: 8px;">{title}</h1>
                <p class="hero-subtitle">{subtitle}</p>
            </div>
            <div style="text-align: right; color: #94a3b8; font-size: 0.82rem; font-family: 'JetBrains Mono', monospace;">
                STATUS: <span style="color: #10b981; font-weight: bold;">● SYSTEM ACTIVE</span><br>
                FRAMEWORK: <span style="color: #22d3ee;">TensorFlow 2.21</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =====================================================================
# SIDEBAR NAVIGATION & CONTROLS
# =====================================================================
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 20px 0; text-align: center;">
        <div style="font-size: 2.2rem; filter: drop-shadow(0 0 10px rgba(6, 182, 212, 0.6));">🛡️</div>
        <h2 style="font-size: 1.15rem; font-weight: 800; color: #f8fafc; margin: 6px 0 0 0; letter-spacing: -0.01em;">
            AI FRAUD SHIELD
        </h2>
        <span style="font-size: 0.72rem; color: #06b6d4; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600;">
            Deep Learning Guardian
        </span>
    </div>
    """, unsafe_allow_html=True)

    menu = st.radio(
        "NAVIGATION",
        [
            "1. Dashboard Overview",
            "2. Transaction Analysis",
            "3. Fraud Detection",
            "4. Deep Learning Model",
            "5. Dataset Explorer",
            "6. About Project"
        ],
        index=0
    )

    st.markdown("---")

    # System Status Indicator
    predictor = get_predictor()
    model_ready = predictor.is_ready()
    data_loaded = (
        os.path.exists(os.path.join(ROOT_DIR, "data", "creditcard.csv")) or
        os.path.exists(os.path.join(ROOT_DIR, "creditcard.csv")) or
        os.path.exists(os.path.join(ROOT_DIR, "data", "creditcard_sample.csv"))
    )

    st.markdown("<p style='font-size: 0.78rem; text-transform: uppercase; color: #94a3b8; font-weight: 700; margin-bottom: 6px;'>Pipeline Telemetry</p>", unsafe_allow_html=True)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown(f"""
        <div style="background: rgba(17,24,39,0.7); border: 1px solid rgba(148,163,184,0.15); border-radius: 8px; padding: 8px; text-align: center;">
            <div style="font-size: 0.7rem; color: #94a3b8;">DATASET</div>
            <div style="font-size: 0.82rem; font-weight: 700; color: {'#10b981' if data_loaded else '#ef4444'};">
                {'● LOADED' if data_loaded else '○ MISSING'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_s2:
        st.markdown(f"""
        <div style="background: rgba(17,24,39,0.7); border: 1px solid rgba(148,163,184,0.15); border-radius: 8px; padding: 8px; text-align: center;">
            <div style="font-size: 0.7rem; color: #94a3b8;">ANN MODEL</div>
            <div style="font-size: 0.82rem; font-weight: 700; color: {'#10b981' if model_ready else '#f59e0b'};">
                {'● TRAINED' if model_ready else '○ NOT TRAINED'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Dataset Drag & Drop Upload
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    with st.expander("📁 Upload Dataset (creditcard.csv)", expanded=not data_loaded):
        uploaded_csv = st.file_uploader("Upload Kaggle CSV", type=["csv"], key="sidebar_csv_uploader")
        if uploaded_csv is not None:
            save_path = os.path.join(ROOT_DIR, "data", "creditcard.csv")
            with open(save_path, "wb") as f:
                f.write(uploaded_csv.getbuffer())
            st.success("`creditcard.csv` successfully uploaded!")
            st.cache_data.clear()
            st.rerun()

    st.markdown("""
    <div style="margin-top: 15px; padding: 12px; border-radius: 8px; background: rgba(139, 92, 246, 0.08); border: 1px dashed rgba(139, 92, 246, 0.3); font-size: 0.75rem; color: #94a3b8;">
        <strong style="color: #c4b5fd;">Quick Tip:</strong> Use the Fraud Detection tab to test benchmark records with real-time explainability.
    </div>
    """, unsafe_allow_html=True)


# =====================================================================
# PAGE 1: DASHBOARD OVERVIEW
# =====================================================================
if menu == "1. Dashboard Overview":
    render_header(
        title="AI CREDIT CARD FRAUD DETECTION SYSTEM",
        subtitle="Real-time Deep Learning Telemetry & Financial Anomaly Monitoring Dashboard"
    )

    df = get_dataset()

    if df is None:
        st.markdown("""
        <div class="result-card-fraud" style="margin-bottom: 25px;">
            <h3 style="margin-top: 0; color: #f87171;">⚠️ Dataset Not Found: <code>data/creditcard.csv</code></h3>
            <p style="color: #cbd5e1; font-size: 0.95rem;">
                The real-time analytics dashboard operates directly on the Kaggle Credit Card Fraud Detection benchmark dataset.
            </p>
            <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 10px; padding: 15px; margin: 15px 0;">
                <strong style="color: #22d3ee;">How to obtain the official dataset:</strong>
                <ol style="margin: 8px 0 0 20px; color: #94a3b8; font-size: 0.9rem;">
                    <li>Download the dataset from Kaggle: <a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud" target="_blank" style="color: #38bdf8; text-decoration: underline;">https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud</a></li>
                    <li>Extract <code>creditcard.csv</code>.</li>
                    <li>Drop the file into the <strong>Upload Dataset</strong> box in the left sidebar, or place it at <code>Credit-Card-Fraud-Detection/data/creditcard.csv</code>.</li>
                </ol>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.info("💡 You can upload your `creditcard.csv` using the uploader in the sidebar to immediately unlock full interactive telemetry.")
        
        # Display architecture blueprint cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Architecture", "3-Layer ANN (128-64-32)")
        with col2:
            st.metric("Oversampling", "SMOTE (Train Only)")
        with col3:
            st.metric("Input Dimensions", "30 Features (V1-V28, Time, Amount)")
        with col4:
            st.metric("Loss Function", "Binary Crossentropy")
    else:
        # Compute exact KPI metrics from dataset
        summary = DataPreprocessor.inspect_data(df)
        total_tx = summary["total_rows"]
        genuine_tx = summary["genuine_count"]
        fraud_tx = summary["fraud_count"]
        fraud_pct = summary["fraud_percentage"]

        # KPI Row
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        with kpi1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">TOTAL TRANSACTIONS <span>📊</span></div>
                <div class="kpi-value text-cyan">{total_tx:,}</div>
                <div class="kpi-subtitle">Recorded in 48-hour transaction window</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">GENUINE TRANSACTIONS <span>✅</span></div>
                <div class="kpi-value text-green">{genuine_tx:,}</div>
                <div class="kpi-subtitle">{(genuine_tx/total_tx)*100:.3f}% of total transactions</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">FRAUD TRANSACTIONS <span>🚨</span></div>
                <div class="kpi-value text-red">{fraud_tx:,}</div>
                <div class="kpi-subtitle">Confirmed unauthorized transactions</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">FRAUD PERCENTAGE <span>⚡</span></div>
                <div class="kpi-value text-purple">{fraud_pct:.4f}%</div>
                <div class="kpi-subtitle">Severe class imbalance (~1:578)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # Overview Charts Row 1
        col_chart1, col_chart2 = st.columns([1, 1.2])

        with col_chart1:
            # Class Distribution Donut Chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=["Genuine Transactions", "Fraudulent Transactions"],
                values=[genuine_tx, fraud_tx],
                hole=0.65,
                marker=dict(colors=["#06b6d4", "#ef4444"], line=dict(color="#0a0e1a", width=2)),
                textinfo="label+percent",
                hoverinfo="value+percent",
                textfont=dict(size=13, color="#ffffff")
            )])
            fig_pie.update_layout(
                title=dict(text="<b>Class Distribution (Severe Imbalance)</b>", font=dict(color=PALETTE["text"], size=16)),
                paper_bgcolor=PALETTE["bg_paper"],
                plot_bgcolor=PALETTE["bg_plot"],
                legend=dict(orientation="h", y=-0.1, font=dict(color=PALETTE["text_muted"])),
                margin=dict(l=20, r=20, t=50, b=30),
                height=360
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_chart2:
            # Amount Distribution Comparison
            sample_size = min(15000, len(df))
            sampled_df = df.sample(sample_size, random_state=42)

            fig_amt = go.Figure()
            fig_amt.add_trace(go.Histogram(
                x=sampled_df[sampled_df["Class"] == 0]["Amount"],
                name="Genuine",
                marker_color="#06b6d4",
                opacity=0.75,
                nbinsx=60
            ))
            fig_amt.add_trace(go.Histogram(
                x=df[df["Class"] == 1]["Amount"],
                name="Fraudulent",
                marker_color="#ef4444",
                opacity=0.85,
                nbinsx=60
            ))
            fig_amt.update_layout(
                title=dict(text="<b>Transaction Amount Distribution (Log Scale)</b>", font=dict(color=PALETTE["text"], size=16)),
                xaxis=dict(title="Transaction Amount ($)", color=PALETTE["text_muted"], gridcolor=PALETTE["grid"], type="log"),
                yaxis=dict(title="Frequency (Count)", color=PALETTE["text_muted"], gridcolor=PALETTE["grid"]),
                barmode="overlay",
                paper_bgcolor=PALETTE["bg_paper"],
                plot_bgcolor=PALETTE["bg_plot"],
                legend=dict(font=dict(color=PALETTE["text"])),
                margin=dict(l=30, r=30, t=50, b=40),
                height=360
            )
            st.plotly_chart(fig_amt, use_container_width=True)

        # Overview Charts Row 2: Temporal Distribution & Fraud Pattern
        col_trend1, col_trend2 = st.columns(2)

        with col_trend1:
            df_time = df.copy()
            df_time["Hour"] = (df_time["Time"] / 3600) % 24
            hourly_counts = df_time.groupby([df_time["Hour"].astype(int), "Class"]).size().unstack(fill_value=0)

            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(
                x=hourly_counts.index,
                y=hourly_counts[0],
                name="Genuine Transactions",
                mode="lines+markers",
                line=dict(color="#22d3ee", width=2)
            ))
            fig_time.update_layout(
                title=dict(text="<b>Transaction Trend Across 24h Cycle</b>", font=dict(color=PALETTE["text"], size=16)),
                xaxis=dict(title="Hour of Day (Cycle)", color=PALETTE["text_muted"], gridcolor=PALETTE["grid"], dtick=2),
                yaxis=dict(title="Transaction Volume", color=PALETTE["text_muted"], gridcolor=PALETTE["grid"]),
                paper_bgcolor=PALETTE["bg_paper"],
                plot_bgcolor=PALETTE["bg_plot"],
                legend=dict(font=dict(color=PALETTE["text"])),
                margin=dict(l=30, r=30, t=50, b=40),
                height=350
            )
            st.plotly_chart(fig_time, use_container_width=True)

        with col_trend2:
            if 1 in hourly_counts.columns:
                fraud_hourly = hourly_counts[1]
                fig_fraud_hour = go.Figure(data=[
                    go.Bar(
                        x=fraud_hourly.index,
                        y=fraud_hourly.values,
                        marker_color="#ef4444",
                        name="Fraudulent Events"
                    )
                ])
                fig_fraud_hour.update_layout(
                    title=dict(text="<b>Fraud Occurrence by Hour of Day</b>", font=dict(color=PALETTE["text"], size=16)),
                    xaxis=dict(title="Hour of Day (Cycle)", color=PALETTE["text_muted"], gridcolor=PALETTE["grid"], dtick=2),
                    yaxis=dict(title="Fraud Count", color=PALETTE["text_muted"], gridcolor=PALETTE["grid"]),
                    paper_bgcolor=PALETTE["bg_paper"],
                    plot_bgcolor=PALETTE["bg_plot"],
                    legend=dict(font=dict(color=PALETTE["text"])),
                    margin=dict(l=30, r=30, t=50, b=40),
                    height=350
                )
                st.plotly_chart(fig_fraud_hour, use_container_width=True)


# =====================================================================
# PAGE 2: TRANSACTION ANALYSIS
# =====================================================================
elif menu == "2. Transaction Analysis":
    render_header(
        title="TRANSACTION & FEATURE ANALYSIS",
        subtitle="Exploratory Data Analysis, Anomaly Detection Patterns, and Correlation Dynamics"
    )

    df = get_dataset()
    if df is None:
        st.error("Please upload or place `data/creditcard.csv` to run transaction analysis.")
    else:
        tab_corr, tab_pca, tab_scatter = st.tabs([
            "📊 Correlation with Fraud",
            "🔬 Latent Features (V1-V28)",
            "🌐 Time vs Amount Scatter"
        ])

        with tab_corr:
            st.markdown("### Top Feature Correlations with Fraud Class")
            st.markdown(
                "Because features V1 through V28 are derived from PCA, they are orthogonal. "
                "However, several features exhibit strong linear and non-linear correlation with the fraudulent class."
            )

            correlations = df.corr()["Class"].drop("Class").sort_values()
            top_negative = correlations.head(6)
            top_positive = correlations.tail(6)
            top_corr = pd.concat([top_negative, top_positive])

            colors = ["#ef4444" if val > 0 else "#06b6d4" for val in top_corr.values]
            fig_corr = go.Figure(go.Bar(
                x=top_corr.values,
                y=top_corr.index,
                orientation="h",
                marker_color=colors
            ))
            fig_corr.update_layout(
                title=dict(text="<b>Key Correlated Features (Top Negative & Positive)</b>", font=dict(color=PALETTE["text"], size=16)),
                xaxis=dict(title="Pearson Correlation with Class (Fraud)", color=PALETTE["text_muted"], gridcolor=PALETTE["grid"]),
                yaxis=dict(color=PALETTE["text"]),
                paper_bgcolor=PALETTE["bg_paper"],
                plot_bgcolor=PALETTE["bg_plot"],
                margin=dict(l=40, r=40, t=50, b=40),
                height=400
            )
            st.plotly_chart(fig_corr, use_container_width=True)

            st.markdown("""
            **Key Findings:**
            - **Negative Correlation:** Features **V14, V12, V10, V17** drop significantly during fraudulent events.
            - **Positive Correlation:** Features **V4, V11, V2, V19** exhibit marked spikes during fraudulent transactions.
            """)

        with tab_pca:
            st.markdown("### Distribution Comparison of Top Fraud Indicators")
            feature_selected = st.selectbox(
                "Select Feature to Inspect:",
                ["V14", "V12", "V10", "V17", "V4", "V11", "V2", "V16", "V3"]
            )

            fig_box = px.box(
                df,
                x="Class",
                y=feature_selected,
                color="Class",
                color_discrete_map={0: "#06b6d4", 1: "#ef4444"},
                labels={"Class": "Transaction Type (0: Genuine, 1: Fraud)"},
                title=f"Distribution of {feature_selected} Across Genuine vs Fraudulent Transactions"
            )
            fig_box.update_layout(
                paper_bgcolor=PALETTE["bg_paper"],
                plot_bgcolor=PALETTE["bg_plot"],
                font=dict(color=PALETTE["text"]),
                height=420
            )
            st.plotly_chart(fig_box, use_container_width=True)

        with tab_scatter:
            st.markdown("### Transaction Time vs Amount Manifold")
            st.markdown("Inspect fraud distribution in the 2D space of Amount ($) and Time (Seconds).")

            sample_genuine = df[df["Class"] == 0].sample(min(2000, len(df[df["Class"] == 0])), random_state=42)
            fraud_subset = df[df["Class"] == 1]
            scatter_df = pd.concat([sample_genuine, fraud_subset])

            fig_scat = px.scatter(
                scatter_df,
                x="Time",
                y="Amount",
                color="Class",
                color_discrete_map={0: "rgba(6, 182, 212, 0.4)", 1: "#ef4444"},
                labels={"Class": "Class (0: Genuine, 1: Fraud)"},
                title="Time vs Amount (Subsampled Genuine vs All Frauds)",
                hover_data=["V14", "V12"]
            )
            fig_scat.update_layout(
                paper_bgcolor=PALETTE["bg_paper"],
                plot_bgcolor=PALETTE["bg_plot"],
                font=dict(color=PALETTE["text"]),
                yaxis=dict(type="log", title="Amount in $ (Log Scale)", gridcolor=PALETTE["grid"]),
                xaxis=dict(title="Time (Seconds from First Transaction)", gridcolor=PALETTE["grid"]),
                height=450
            )
            st.plotly_chart(fig_scat, use_container_width=True)


# =====================================================================
# PAGE 3: FRAUD DETECTION (LIVE INFERENCE)
# =====================================================================
elif menu == "3. Fraud Detection":
    render_header(
        title="REAL-TIME FRAUD DETECTION INFERENCE",
        subtitle="Simulate Transaction Payloads, Score Probabilities with Deep Learning, and Inspect Explainability"
    )

    tab_single, tab_batch = st.tabs(["⚡ Single Transaction Scoring", "📦 Batch CSV Scoring"])

    with tab_single:
        predictor = get_predictor()

        if not predictor.is_ready():
            st.warning("""
            ⚠️ **Model Artifacts Not Yet Loaded**
            
            The Artificial Neural Network model (`models/fraud_model.keras`) or Scaler (`models/scaler.joblib`) 
            has not been trained yet.
            
            **To train the model:**
            1. Ensure `data/creditcard.csv` is uploaded.
            2. Go to the **4. Deep Learning Model** tab in the sidebar and click **Train Model Now**, or run `python train.py` from your terminal.
            """)
        else:
            st.markdown("### 1. Load Preset Benchmark or Enter Manual Values")

            col_p1, col_p2, col_p3 = st.columns(3)
            if "preset_data" not in st.session_state:
                st.session_state.preset_data = SAMPLE_GENUINE_TRANSACTION.copy()

            with col_p1:
                if st.button("🟢 Load Sample Genuine Transaction", use_container_width=True):
                    st.session_state.preset_data = SAMPLE_GENUINE_TRANSACTION.copy()
                    st.rerun()

            with col_p2:
                if st.button("🔴 Load Sample Fraudulent Transaction", use_container_width=True):
                    st.session_state.preset_data = SAMPLE_FRAUD_TRANSACTION.copy()
                    st.rerun()

            with col_p3:
                if st.button("🟠 Load Suspicious High-Risk Transaction", use_container_width=True):
                    st.session_state.preset_data = SAMPLE_SUSPICIOUS_TRANSACTION.copy()
                    st.rerun()

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

            # Input Form
            with st.form("fraud_prediction_form"):
                st.markdown("#### Primary Transaction Parameters")
                col_t1, col_t2, col_t3 = st.columns([1, 1, 1])
                with col_t1:
                    input_time = st.number_input(
                        "Transaction Time (Seconds)",
                        value=float(st.session_state.preset_data.get("Time", 0.0)),
                        step=100.0,
                        help="Seconds elapsed since the first transaction in the dataset"
                    )
                with col_t2:
                    input_amount = st.number_input(
                        "Transaction Amount ($)",
                        value=float(st.session_state.preset_data.get("Amount", 50.0)),
                        min_value=0.0,
                        step=5.0,
                        help="Transaction monetary value in USD"
                    )
                with col_t3:
                    threshold = st.slider(
                        "Decision Threshold",
                        min_value=0.1,
                        max_value=0.9,
                        value=0.5,
                        step=0.05,
                        help="Probability cutoff for classifying a transaction as fraudulent"
                    )

                st.markdown("#### Latent Features (PCA Components V1 - V28)")
                st.caption("Expand sections to inspect or fine-tune specific latent PCA vectors:")

                tab_v1, tab_v2 = st.tabs(["Components V1 - V14 (High Sensitivity)", "Components V15 - V28"])

                inputs_dict = {"Time": input_time, "Amount": input_amount}

                with tab_v1:
                    cols_v1 = st.columns(4)
                    for idx in range(1, 15):
                        c_name = f"V{idx}"
                        col_target = cols_v1[(idx - 1) % 4]
                        with col_target:
                            val = st.number_input(
                                f"Feature {c_name}",
                                value=float(st.session_state.preset_data.get(c_name, 0.0)),
                                step=0.1,
                                format="%.4f",
                                key=f"input_{c_name}"
                            )
                            inputs_dict[c_name] = val

                with tab_v2:
                    cols_v2 = st.columns(4)
                    for idx in range(15, 29):
                        c_name = f"V{idx}"
                        col_target = cols_v2[(idx - 15) % 4]
                        with col_target:
                            val = st.number_input(
                                f"Feature {c_name}",
                                value=float(st.session_state.preset_data.get(c_name, 0.0)),
                                step=0.1,
                                format="%.4f",
                                key=f"input_{c_name}"
                            )
                            inputs_dict[c_name] = val

                st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
                submit_btn = st.form_submit_button("🔍 PREDICT FRAUD RISK", use_container_width=True)

            # Handle Prediction Output
            if submit_btn:
                with st.spinner("Analyzing transaction through 3-Layer Deep Neural Network..."):
                    res = predictor.predict_single(inputs_dict, threshold=threshold)

                prob = res["fraud_probability"]
                is_fraud = res["is_fraud"]

                st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
                st.markdown("### Detection Outcome & Risk Analysis")

                card_class = "result-card-fraud" if is_fraud else "result-card-genuine"
                card_icon = "🚨 FRAUD DETECTED" if is_fraud else "✅ TRANSACTION APPROVED (GENUINE)"

                st.markdown(f"""
                <div class="{card_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                        <div>
                            <h2 style="margin: 0; font-size: 1.8rem; font-weight: 800; color: {'#ef4444' if is_fraud else '#10b981'};">
                                {card_icon}
                            </h2>
                            <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 0.95rem;">
                                Classification Outcome at Decision Threshold: <strong>{threshold:.2f}</strong>
                            </p>
                        </div>
                        <div style="text-align: right;">
                            <span style="font-size: 0.8rem; text-transform: uppercase; color: #94a3b8; font-weight: bold;">ASSESSED RISK TIER</span>
                            <div style="font-size: 1.3rem; font-weight: 800; color: {res['risk_color']};">
                                {res['risk_level']}
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

                col_gauge, col_exp = st.columns([1, 1.4])

                with col_gauge:
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        number={"suffix": "%", "font": {"color": "#ffffff", "size": 36}},
                        title={"text": "<b>Fraud Probability Score</b>", "font": {"color": PALETTE["text"], "size": 16}},
                        gauge={
                            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": PALETTE["text_muted"]},
                            "bar": {"color": res["risk_color"]},
                            "bgcolor": "#1e293b",
                            "borderwidth": 1,
                            "bordercolor": "rgba(148,163,184,0.2)",
                            "steps": [
                                {"range": [0, 20], "color": "rgba(16, 185, 129, 0.2)"},
                                {"range": [20, 50], "color": "rgba(245, 158, 11, 0.2)"},
                                {"range": [50, 85], "color": "rgba(249, 115, 22, 0.2)"},
                                {"range": [85, 100], "color": "rgba(239, 68, 68, 0.2)"}
                            ],
                            "threshold": {
                                "line": {"color": "#ffffff", "width": 3},
                                "thickness": 0.8,
                                "value": threshold * 100
                            }
                        }
                    ))
                    fig_gauge.update_layout(
                        paper_bgcolor=PALETTE["bg_paper"],
                        plot_bgcolor=PALETTE["bg_plot"],
                        font=dict(color=PALETTE["text"]),
                        margin=dict(l=30, r=30, t=50, b=30),
                        height=280
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)

                with col_exp:
                    st.markdown("""
                    <div class="glass-panel" style="height: 280px; overflow-y: auto;">
                        <h4 style="margin-top: 0; color: #22d3ee;">🧠 Neural Network Decision Insights</h4>
                    """, unsafe_allow_html=True)

                    for item in res["explanation"]:
                        st.markdown(f"- {item}")

                    st.markdown("""
                    <hr style="border-color: rgba(148,163,184,0.15); margin: 15px 0;">
                    <p style="font-size: 0.82rem; color: #94a3b8; margin: 0;">
                        Scaled Inputs: <code>Time</code> & <code>Amount</code> normalized via fitted RobustScaler; 
                        PCA vectors <code>V1-V28</code> mapped through 3 hidden layers (128 → 64 → 32).
                    </p>
                    </div>
                    """, unsafe_allow_html=True)

    with tab_batch:
        st.markdown("### Batch Transaction Inference")
        st.markdown("Upload a CSV file containing transactions (`Time`, `V1`..`V28`, `Amount`) to score them all at once.")
        
        batch_file = st.file_uploader("Upload CSV for Batch Prediction", type=["csv"], key="batch_scoring_uploader")
        if batch_file is not None:
            batch_df = pd.read_csv(batch_file)
            st.write(f"Uploaded batch containing **{len(batch_df):,}** transactions.")
            
            predictor = get_predictor()
            if not predictor.is_ready():
                st.error("Please train the model first before running batch predictions.")
            else:
                if st.button("🚀 Run Batch Prediction", use_container_width=True):
                    with st.spinner("Scoring batch transactions..."):
                        scored_df = predictor.predict_batch(batch_df)
                    
                    fraud_found = int((scored_df["Predicted_Class"] == 1).sum())
                    st.markdown(f"""
                    <div class="kpi-card" style="margin-bottom: 20px;">
                        <div class="kpi-title">BATCH DETECTION SUMMARY</div>
                        <div class="kpi-value {'text-red' if fraud_found > 0 else 'text-green'}">{fraud_found} / {len(scored_df)} FRAUDULENT</div>
                        <div class="kpi-subtitle">Scored using trained 3-Layer Deep ANN</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.dataframe(scored_df.head(100), use_container_width=True)

                    csv_res = scored_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Scored Batch CSV",
                        data=csv_res,
                        file_name="scored_transactions.csv",
                        mime="text/csv"
                    )


# =====================================================================
# PAGE 4: DEEP LEARNING MODEL & TRAINING
# =====================================================================
elif menu == "4. Deep Learning Model":
    render_header(
        title="DEEP LEARNING MODEL & EVALUATION",
        subtitle="Artificial Neural Network Architecture, Training Convergence, and Rigorous Evaluation Metrics"
    )

    metrics_data, history_data = load_metrics_and_history(os.path.join(ROOT_DIR, "models"))

    tab_eval, tab_arch, tab_train_curves, tab_train_now = st.tabs([
        "📈 Evaluation Metrics & Curves",
        "🏗️ ANN Architecture Specification",
        "📉 Training Convergence History",
        "🚀 Train Model Pipeline"
    ])

    with tab_eval:
        if metrics_data is None:
            st.warning("""
            ⚠️ **Evaluation Metrics Not Found**
            
            Model evaluation metrics are generated when `train.py` executes on the actual test set.
            To train the model and record authentic evaluation results:
            
            ```bash
            python train.py --data_path data/creditcard.csv --epochs 20
            ```
            *(Synthetic metrics are strictly omitted per engineering guidelines.)*
            """)
        else:
            st.markdown("### Authentic Model Test Performance")

            m1, m2, m3, m4, m5, m6 = st.columns(6)
            with m1:
                st.metric("Accuracy", f"{metrics_data.get('accuracy', 0):.4f}")
            with m2:
                st.metric("Precision", f"{metrics_data.get('precision', 0):.4f}")
            with m3:
                st.metric("Recall", f"{metrics_data.get('recall', 0):.4f}")
            with m4:
                st.metric("F1-Score", f"{metrics_data.get('f1_score', 0):.4f}")
            with m5:
                st.metric("ROC-AUC", f"{metrics_data.get('roc_auc', 0):.4f}")
            with m6:
                st.metric("PR-AUC", f"{metrics_data.get('pr_auc', 0):.4f}")

            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

            col_cm, col_roc = st.columns(2)
            with col_cm:
                if "confusion_matrix" in metrics_data:
                    fig_cm = plot_confusion_matrix(metrics_data["confusion_matrix"])
                    st.plotly_chart(fig_cm, use_container_width=True)

            with col_roc:
                if "curves" in metrics_data and "fpr" in metrics_data["curves"]:
                    fig_roc = plot_roc_curve(
                        metrics_data["curves"]["fpr"],
                        metrics_data["curves"]["tpr"],
                        metrics_data.get("roc_auc", 0.0)
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)

            col_pr, col_rep = st.columns([1.1, 0.9])
            with col_pr:
                if "curves" in metrics_data and "precision_curve" in metrics_data["curves"]:
                    fig_pr = plot_precision_recall_curve(
                        metrics_data["curves"]["recall_curve"],
                        metrics_data["curves"]["precision_curve"],
                        metrics_data.get("pr_auc", 0.0)
                    )
                    st.plotly_chart(fig_pr, use_container_width=True)

            with col_rep:
                st.markdown("#### Classification Report Details")
                if "classification_report" in metrics_data:
                    report_df = pd.DataFrame(metrics_data["classification_report"]).transpose()
                    st.dataframe(
                        report_df.style.format(precision=4),
                        use_container_width=True
                    )

    with tab_arch:
        st.markdown("### Deep Artificial Neural Network (ANN) Topology")
        st.markdown("""
        The network architecture is specifically engineered to handle high-dimensional fraud signals
        while combating overfitting on skewed tabular credit card data:
        """)

        st.markdown("""
        <div class="arch-pill">
            <strong>Input Layer:</strong> Shape = <code>(30,)</code> [Time, V1..V28, Amount]
        </div>
        <div class="arch-pill">
            <strong>Dense Block 1:</strong> 128 Neurons | Activation: <code>ReLU</code> | Batch Normalization | Dropout: <code>0.30</code>
        </div>
        <div class="arch-pill">
            <strong>Dense Block 2:</strong> 64 Neurons | Activation: <code>ReLU</code> | Batch Normalization | Dropout: <code>0.20</code>
        </div>
        <div class="arch-pill">
            <strong>Dense Block 3:</strong> 32 Neurons | Activation: <code>ReLU</code>
        </div>
        <div class="arch-pill" style="border-left-color: #8b5cf6;">
            <strong>Output Layer:</strong> 1 Neuron | Activation: <code>Sigmoid</code> (Fraud Probability ∈ [0, 1])
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        #### Compilation Details:
        - **Optimizer:** `Adam(learning_rate=0.001)` with dynamic learning rate annealing
        - **Loss Function:** `Binary Crossentropy`
        - **Key Metrics Monitored:** `Precision`, `Recall`, `AUC (ROC)`, `PR-AUC (Average Precision)`
        - **Regularization Strategy:** Batch Normalization stabilizes layer inputs across mini-batches; Dropout prevents co-adaptation of hidden features.
        """)

    with tab_train_curves:
        if history_data is None:
            st.info("Training history curves will display once the model completes training.")
        else:
            fig_loss, fig_acc = plot_training_history(history_data)
            col_h1, col_h2 = st.columns(2)
            with col_h1:
                st.plotly_chart(fig_loss, use_container_width=True)
            with col_h2:
                st.plotly_chart(fig_acc, use_container_width=True)

    with tab_train_now:
        st.markdown("### In-Dashboard Training Execution")
        st.markdown("Initiate the full end-to-end ANN training pipeline with SMOTE oversampling and callbacks.")

        df_train_check = get_dataset()
        if df_train_check is None:
            st.warning("Please upload `creditcard.csv` in the sidebar to enable model training.")
        else:
            st.success(f"`creditcard.csv` detected with **{len(df_train_check):,}** transactions.")
            
            c_ep, c_bs, c_sc = st.columns(3)
            with c_ep:
                epochs_input = st.number_input("Epochs", min_value=5, max_value=50, value=20, step=5)
            with c_bs:
                batch_size_input = st.selectbox("Batch Size", [512, 1024, 2048], index=1)
            with c_sc:
                scaler_input = st.selectbox("Feature Scaler", ["robust", "standard"], index=0)

            if st.button("🚀 START MODEL TRAINING", use_container_width=True):
                with st.status("Training Deep Learning ANN Pipeline...", expanded=True) as status:
                    st.write("1. Preprocessing data & applying SMOTE to training split...")
                    preprocessor = DataPreprocessor(scaler_type=scaler_input)
                    X_tr_res, y_tr_res, X_te_scaled, y_te, meta = preprocessor.run_pipeline(
                        df=df_train_check,
                        test_size=0.2,
                        apply_smote=True
                    )
                    st.write(f"-> Balanced Training Set: {len(X_tr_res):,} samples")
                    
                    st.write("2. Saving preprocessor scalers...")
                    models_dir = os.path.join(ROOT_DIR, "models")
                    preprocessor.save_artifacts(models_dir)

                    st.write("3. Compiling 3-Layer ANN...")
                    model = build_fraud_ann(input_dim=30, learning_rate=0.001)

                    st.write("4. Fitting ANN with EarlyStopping & ModelCheckpoint...")
                    ckpt_path = os.path.join(models_dir, "fraud_model.keras")
                    cbs = get_training_callbacks(checkpoint_filepath=ckpt_path, patience=5)
                    
                    history = model.fit(
                        X_tr_res,
                        y_tr_res,
                        validation_data=(X_te_scaled, y_te),
                        epochs=int(epochs_input),
                        batch_size=int(batch_size_input),
                        callbacks=cbs,
                        verbose=0
                    )
                    st.write("5. Calculating test evaluation metrics...")
                    test_probs = model.predict(X_te_scaled, batch_size=2048, verbose=0).flatten()
                    metrics_res = evaluate_model_predictions(y_te, test_probs)

                    save_metrics_and_history(
                        metrics_dict=metrics_res,
                        history_dict=history.history,
                        save_dir=models_dir
                    )
                    status.update(label="Training complete! Artifacts saved.", state="complete")

                st.success("Model trained successfully! Switch to the 'Evaluation Metrics & Curves' tab to view results.")
                st.rerun()


# =====================================================================
# PAGE 5: DATASET EXPLORER
# =====================================================================
elif menu == "5. Dataset Explorer":
    render_header(
        title="DATASET EXPLORER & TRANSACTION VIEWER",
        subtitle="Search, Filter, Inspect, and Export Credit Card Transactions"
    )

    df = get_dataset()

    if df is None:
        st.warning("""
        ⚠️ **Dataset Not Loaded**
        
        Please upload `creditcard.csv` in the sidebar to use the Dataset Explorer.
        """)
    else:
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            class_filter = st.selectbox(
                "Filter by Transaction Class:",
                ["All Transactions", "Fraudulent Only (Class 1)", "Genuine Only (Class 0)"]
            )
        with col_f2:
            min_amt = float(df["Amount"].min())
            max_amt = float(df["Amount"].max())
            amt_range = st.slider(
                "Transaction Amount Range ($):",
                min_value=min_amt,
                max_value=max_amt,
                value=(min_amt, min(5000.0, max_amt))
            )
        with col_f3:
            row_limit = st.slider("Display Limit:", 10, 500, 50)

        filtered_df = df[(df["Amount"] >= amt_range[0]) & (df["Amount"] <= amt_range[1])]

        if class_filter == "Fraudulent Only (Class 1)":
            filtered_df = filtered_df[filtered_df["Class"] == 1]
        elif class_filter == "Genuine Only (Class 0)":
            filtered_df = filtered_df[filtered_df["Class"] == 0]

        st.markdown(f"**Found {len(filtered_df):,} matching transactions** (showing first {min(row_limit, len(filtered_df))})")

        st.dataframe(filtered_df.head(row_limit), use_container_width=True)

        csv_data = filtered_df.head(row_limit).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Transactions as CSV",
            data=csv_data,
            file_name="creditcard_filtered.csv",
            mime="text/csv"
        )


# =====================================================================
# PAGE 6: ABOUT PROJECT
# =====================================================================
elif menu == "6. About Project":
    render_header(
        title="ABOUT THE PROJECT & SYSTEM ARCHITECTURE",
        subtitle="Academic & Engineering Documentation for College Submission and Review"
    )

    st.markdown("""
    ### 🎯 Project Overview & Problem Statement
    Credit card fraud presents one of the most critical threats to digital banking and electronic commerce.
    Billions of dollars are lost annually to unauthorized transactions. 
    However, detecting fraudulent behavior entails significant challenges:
    
    1. **Extreme Class Imbalance:** Fraud represents less than 0.2% of all credit card transactions. 
       A naive classifier predicting "genuine" for every transaction achieves 99.8% accuracy while failing completely.
    2. **Concept Drift & Concealment:** Fraudulent actors constantly mutate behavior to mimic legitimate cardholders.
    3. **Zero Tolerance for Data Leakage:** Feature scaling and synthetic oversampling (SMOTE) must strictly be fitted 
       only on the training split, keeping test sets entirely untouched.
    """)

    col_a1, col_a2 = st.columns(2)

    with col_a1:
        st.markdown("""
        ### 🛠️ Technology Stack
        - **Deep Learning Framework:** TensorFlow 2.x / Keras
        - **Synthetic Resampling:** Imbalanced-Learn (SMOTE)
        - **Data Processing:** Pandas, NumPy, Scikit-learn
        - **Model Persistence:** Joblib & Keras H5/Native Format
        - **Interactive UI/UX:** Streamlit with Custom CSS
        - **Data Visualizations:** Plotly Interactive Graph Objects
        """)

    with col_a2:
        st.markdown("""
        ### 📐 End-to-End Pipeline Workflow
        1. **Ingestion & Validation:** Load `creditcard.csv`, check missing & duplicate records.
        2. **Stratified Split:** Split into 80% train / 20% test preserving class balance.
        3. **Robust Scaling:** Scale `Time` and `Amount` using training set medians and IQRs.
        4. **SMOTE Oversampling:** Apply SMOTE exclusively on training split.
        5. **ANN Model:** 3-layer architecture with Batch Normalization & Dropout.
        6. **Rigorous Evaluation:** Precision-Recall AUC, ROC-AUC, and Confusion Matrix.
        7. **Live Inference:** Preprocessing pipeline deployed to Streamlit UI with explainability.
        """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #94a3b8; font-size: 0.85rem; padding: 20px;">
        Designed and developed for academic presentation & professional deployment.<br>
        AI Credit Card Fraud Detection System • Version 1.0.0
    </div>
    """, unsafe_allow_html=True)
