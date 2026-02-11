"""
Credit Risk Classification
Streamlit App (Single-Screen Layout with caching)
"""

import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------
# Ensure src/ is on Python path
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# -------------------------------------------------
# Backend imports
# -------------------------------------------------
from src.data_utils import get_dataset_info
from src.pipeline import load_models
from src.metrics import (
    evaluate_model,
    comparison_as_df,
    load_results,
)

# -------------------------------------------------
# Cached loaders
# -------------------------------------------------
@st.cache_resource
def load_models_cached():
    """Load all trained models."""
    return load_models()


@st.cache_data
def load_results_cached():
    """Load precomputed evaluation results."""
    try:
        return load_results()
    except Exception:
        return None


@st.cache_data
def load_dataset_info():
    """Load dataset metadata."""
    return get_dataset_info()

# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Classification",
    page_icon="💳",
    layout="wide"
)

# -------------------------------------------------
# Title + description
# -------------------------------------------------
st.title("💳 Credit Risk Classification")
st.markdown("### 📋 Problem Statement")
st.markdown("""
        <div class="info-box">
        <b>Objective:</b> Classify individuals as good or bad credit risks based on their financial 
        and personal attributes using multiple machine learning models.
        <br>
        <b>Business Impact:</b> This classification helps financial institutions make informed 
        decisions about loan approvals, reducing default risk and improving lending efficiency.
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# -------------------------------------------------
# Dataset info
# -------------------------------------------------
st.markdown('<h2 class="sub-header">📊 Dataset Information</h2>', unsafe_allow_html=True)
    
dataset_info = load_dataset_info()

col1, col2, col3, col4 = st.columns(4)
    
with col1:
    st.metric("📁 Instances", dataset_info['instances'])
with col2:
    st.metric("🔢 Features", dataset_info['features'])
with col3:
    st.metric("🎯 Task", dataset_info['task'])
with col4:
    st.metric("❌ Missing Values", "No" if not dataset_info['missing_values'] else "Yes")

st.markdown("### Dataset Details")
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    **Name:** {dataset_info['name']}  
    **Source:** {dataset_info['source']}  
    **URL:** [{dataset_info['url']}]({dataset_info['url']})
    """)

with col2:
    st.markdown(f"""
    **Target Variable:** {dataset_info['target']}  
    **Classes:** Good Credit (0), Bad Credit (1)  
    **Distribution:** Imbalanced (70% Good, 30% Bad)
    """)

st.markdown("---")
    # Technical Implementation
st.markdown('<h2 class="sub-header">🔧 Technical Implementation</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    - ✅ Pipeline implementation
    - ✅ ColumnTransformer
    - ✅ OneHotEncoder for categorical features
    - ✅ StandardScaler for numerical features
    """)

with col2:
    st.markdown("""
    - ✅ Model & other Serialization -joblib
    - ✅ Package structure with data, pipeline and metrics modules - SoC
    - ✅ No data leakage
    - ✅ Streamlit application
    """)

# -------------------------------------------------
# Model Comparison Table – all models
# -------------------------------------------------
st.header("📋 Model Comparison (All Models)")

results_cached = load_results_cached()

if results_cached is None:
    st.warning(
        "⚠️ Precomputed results not found.\n\n"
        "Run the evaluation script to generate and save results."
    )
else:
    try:
        comparison_df = comparison_as_df(results_cached)
        st.dataframe(comparison_df, width='stretch', hide_index=True)
        st.caption("Confusion matrix is raveled as: TN, FP, FN, TP. Click on the column header to sort.")
    except Exception as e:
        st.error("❌ Failed to display model comparison")
        st.error(str(e))
st.markdown(
    """
    **Confusion Matrix Interpretation (Credit Risk)**  
    For credit risk classification, the **positive class (1) is Bad Credit**.
    - **Negative class (0):** Good Credit  
    - **Positive class (1):** Bad Credit  
    **Confusion Matrix Terms:**
    - **TN (True Negative):** Good Credit correctly predicted as Good Credit  
    - **FP (False Positive):** Good Credit incorrectly predicted as Bad Credit  
    - **FN (False Negative):** Bad Credit incorrectly predicted as Good Credit  
    - **TP (True Positive):** Bad Credit correctly predicted as Bad Credit
    """
)

# Quick Stats
if results_cached:
    st.markdown("---")
    st.markdown('<h2 class="sub-header">📈 Quick Statistics</h2>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
with col1:
    best_accuracy = comparison_df['Accuracy'].max()
    best_acc_model = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model Name']
    st.metric("🎯 Best Accuracy", f"{best_accuracy:.4f}", f"{best_acc_model}")

with col2:
    best_auc = comparison_df['AUC'].max()
    best_auc_model = comparison_df.loc[comparison_df['AUC'].idxmax(), 'Model Name']
    st.metric("📊 Best AUC", f"{best_auc:.4f}", f"{best_auc_model}")

with col3:
    best_f1 = comparison_df['F1'].max()
    best_f1_model = comparison_df.loc[comparison_df['F1'].idxmax(), 'Model Name']
    st.metric("⚖️ Best F1 Score", f"{best_f1:.4f}", f"{best_f1_model}")

st.markdown("---")
# -------------------------------------------------
# Model Comparison page
# -------------------------------------------------
#st.markdown('<h2 class="sub-header">📊 Model Performance Comparison</h2>', unsafe_allow_html=True)
st.markdown('<h2 class="sub-header">📈 Performance Visualizations</h2>', unsafe_allow_html=True)
# Visualizations
#st.markdown("### 📈 Performance Visualizations")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1']
    x = np.arange(len(comparison_df))
    width = 0.15
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, metric in enumerate(metrics):
        values = comparison_df[metric].values
        ax.bar(x + i*width, values, width, label=metric, color=colors[i])
    
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(comparison_df['Model Name'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(10, 6))
    mcc_scores = comparison_df['MCC'].values
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(mcc_scores)))
    bars = ax.barh(comparison_df['Model Name'], mcc_scores, color=colors)
    ax.set_xlabel('MCC Score', fontsize=12)
    ax.set_title('Matthews Correlation Coefficient', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    for i, (bar, score) in enumerate(zip(bars, mcc_scores)):
        ax.text(score + 0.01, i, f'{score:.4f}', va='center')
    
    plt.tight_layout()
    st.pyplot(fig)

# -------------------------------------------------
# CSV Upload
# -------------------------------------------------
st.header("📁 Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"],
    label_visibility="collapsed"
)

df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {df.shape[0]} rows")
        st.dataframe(df.head(5), width='stretch')
        #st.dataframe(df.head(5))
        if "Target" not in df.columns:
            st.warning("⚠️ Target column missing (evaluation disabled)")
    except Exception as e:
        st.error("❌ Failed to read CSV")
        st.error(str(e))
else:
    st.info("Upload test dataset to continue")

st.markdown("---")

# -------------------------------------------------
# Model Selection
# -------------------------------------------------
st.header("🤖 Model Selection")

models = {}
try:
    models = load_models_cached()
except Exception:
    pass

selected_model = None
selected_model_name = None

if not models:
    st.error("❌ Trained models not found")
elif df is None:
    st.info("Upload data first to select a model")
else:
    selected_model_name = st.selectbox(
        "Select model",
        list(models.keys())
    )
    selected_model = models[selected_model_name]
    st.success(f"Selected model: **{selected_model_name}**")

st.markdown("---")

# -------------------------------------------------
# Evaluation Metrics – 1 row
# -------------------------------------------------
st.header("📊 Evaluation Metrics")

results_single = None

if df is None or selected_model is None:
    st.info("Upload data and select a model to view metrics")
elif "Target" not in df.columns:
    st.warning("Target column missing — metrics unavailable")
else:
    X_test = df.drop("Target", axis=1)
    y_test = df["Target"].values

    try:
        results_single = evaluate_model(selected_model, X_test, y_test)

        cols = st.columns(6)
        cols[0].metric("Accuracy", f"{results_single['Accuracy']:.4f}")
        cols[1].metric("Precision", f"{results_single['Precision']:.4f}")
        cols[2].metric("Recall", f"{results_single['Recall']:.4f}")
        cols[3].metric("F1", f"{results_single['F1']:.4f}")
        cols[4].metric("AUC", f"{results_single['AUC']:.4f}")
        cols[5].metric("MCC", f"{results_single['MCC']:.4f}")

    except Exception as e:
        st.error("❌ Evaluation failed")
        st.error(str(e))

st.markdown("---")

# -------------------------------------------------
# Confusion Matrix – compact
# -------------------------------------------------
st.header("📌 Confusion Matrix")

if results_single is None:
    st.info("Confusion matrix will appear after evaluation")
else:
    _, col_cm, _ = st.columns([1, 2, 1])
    with col_cm:
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(
            results_single["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            annot_kws={"size": 10},
            xticklabels=["Good (0)", "Bad (1)"],
            yticklabels=["Good (0)", "Bad (1)"],
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(selected_model_name)
        plt.tight_layout()
        st.pyplot(fig)

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption("Credit Risk Classification - Multiple ML classifiier models end-to-end")
