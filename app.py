"""
Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.pipeline import load_models, get_model_names
from src.metrics import (
    evaluate_model,
    comparison_as_df,
    load_results,
    get_metric_descriptions
)
from src.data_utils import get_dataset_info
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Statlog German Credit Risk Classification",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2196f3;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">💳 Statlog German Credit Risk Classification</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/bank-building.png", width=100)
    st.title("🎯 Navigation")
    st.markdown("---")
    
    page = st.radio(
        "Select Page:",
        ["🏠 Home", "📊 Model Comparison", "🔍 Model Details", "🔮 Predictions"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### 📚 About")
    st.info("This app classifies credit risk using 6 ML models with ColumnTransformer preprocessing and joblib serialization.")
    
    st.markdown("### 👨‍💻 Models")
    st.markdown("""
    - Logistic Regression
    - Decision Tree
    - K-Nearest Neighbors
    - Naive Bayes
    - Random Forest
    - XGBoost
    """)

# Cache functions
@st.cache_resource
def load_models_cached():
    """Load all models"""
    return load_models()

@st.cache_data
def load_results_cached():
    """Load evaluation results"""
    try:
        return load_results()
    except:
        return None

@st.cache_data
def load_dataset_info():
    """Load dataset information"""
    return get_dataset_info()

# Load data
models = load_models_cached()
results = load_results_cached()
dataset_info = load_dataset_info()

# HOME PAGE
if page == "🏠 Home":
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 📋 Problem Statement")
        st.markdown("""
        <div class="info-box">
        <b>Objective:</b> Classify individuals as good or bad credit risks based on their financial 
        and personal attributes using multiple machine learning models.
        <br><br>
        <b>Business Impact:</b> This classification helps financial institutions make informed 
        decisions about loan approvals, reducing default risk and improving lending efficiency.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Dataset Information
    st.markdown('<h2 class="sub-header">📊 Dataset Information</h2>', unsafe_allow_html=True)
    
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
        **URL:** [View Dataset]({dataset_info['url']})
        """)
    
    with col2:
        st.markdown(f"""
        **Target Variable:** {dataset_info['target']}  
        **Classes:** Good Credit (0), Bad Credit (1)  
        **Distribution:** Imbalanced (70% Good, 30% Bad)
        """)
    
    st.markdown("---")
 
  
    # Quick Stats
    if results:
        st.markdown("---")
        st.markdown('<h2 class="sub-header">📈 Quick Statistics</h2>', unsafe_allow_html=True)
        
        comparison_df = comparison_as_df(results)
        
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

# MODEL COMPARISON PAGE
elif page == "📊 Model Comparison":
    st.markdown('<h2 class="sub-header">📊 Model Performance Comparison</h2>', unsafe_allow_html=True)
    
    if results:
        # Comparison Table
        comparison_df = comparison_as_df(results)
        
        # Format for display
        formatted_df = comparison_df.copy()
        for col in ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.4f}")
        
        st.dataframe(formatted_df, use_container_width=True, hide_index=True)
        
        # Visualizations
        st.markdown("### 📈 Performance Visualizations")
        
        #tab1, tab2, tab3 = st.tabs(["Bar Charts", "Radar Chart", "Heatmap"])
        tab1 = st.tabs(["Bar Charts"])[0]
        
        with tab1:
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
       
    
    else:
        st.error("⚠️ Results not found. Please run the training script first: `python train.py`")

# MODEL DETAILS PAGE
elif page == "🔍 Model Details":
    st.markdown('<h2 class="sub-header">🔍 Detailed Model Analysis</h2>', unsafe_allow_html=True)
    
    if results and models:
        # Model selection
        model_names = list(results.keys())
        selected_model = st.selectbox("Select a model to view details:", model_names)
        
        st.markdown(f"### {selected_model}")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🎯 Accuracy", f"{results[selected_model]['Accuracy']:.4f}")
            st.metric("📊 AUC Score", f"{results[selected_model]['AUC']:.4f}")
        
        with col2:
            st.metric("✅ Precision", f"{results[selected_model]['Precision']:.4f}")
            st.metric("🔄 Recall", f"{results[selected_model]['Recall']:.4f}")
        
        with col3:
            st.metric("⚖️ F1 Score", f"{results[selected_model]['F1']:.4f}")
            st.metric("📈 MCC Score", f"{results[selected_model]['MCC']:.4f}")
        
        st.markdown("---")
        
        # Confusion Matrix and Report
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📊 Confusion Matrix")
            cm = results[selected_model]['confusion_matrix']
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Good Credit', 'Bad Credit'],
                       yticklabels=['Good Credit', 'Bad Credit'],
                       cbar_kws={'label': 'Count'})
            ax.set_xlabel('Predicted Label', fontsize=12)
            ax.set_ylabel('True Label', fontsize=12)
            ax.set_title(f'{selected_model} - Confusion Matrix', fontsize=14, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
    
    else:
        st.error("⚠️ Models or results not found. Please run the training script first.")

# PREDICTIONS PAGE
elif page == "🔮 Predictions":
    st.markdown('<h2 class="sub-header">🔮 Make Predictions</h2>', unsafe_allow_html=True)
    
    if models and results:
        # Model selection
        model_names = list(models.keys())
        selected_model_name = st.selectbox("Select a model for predictions:", model_names)
        
        st.markdown("---")
        
        # File upload
        st.markdown("### 📁 Upload Test Data")
        st.info("Upload a CSV file with the same features as the training data. The pipeline will handle preprocessing automatically!")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
                
                # Show preview
                with st.expander("👀 Preview Data"):
                    st.dataframe(df.head(10))
                
                # Check if target exists
                has_target = 'Target' in df.columns
                
                if has_target:
                    y_true = df['Target'].values
                    X_test = df.drop('Target', axis=1)
                else:
                    X_test = df
                
                # Make predictions (pipeline handles preprocessing!)
                model = models[selected_model_name]
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Display results
                st.markdown("### 📊 Prediction Results")
                
                results_df = pd.DataFrame({
                    'Sample': range(1, len(y_pred) + 1),
                    'Predicted Class': ['Bad Credit' if p == 1 else 'Good Credit' for p in y_pred],
                    'Confidence (Bad)': [f"{prob:.2%}" for prob in y_pred_proba],
                    'Prediction': y_pred
                })
                
                if has_target:
                    results_df['Actual Class'] = ['Bad Credit' if t == 1 else 'Good Credit' for t in y_true]
                    results_df['Correct'] = ['✅' if p == t else '❌' for p, t in zip(y_pred, y_true)]
                
                # Show results
                st.dataframe(results_df.drop('Prediction', axis=1).head(20), use_container_width=True, hide_index=True)
                
                # Summary
                st.markdown("### 📈 Prediction Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📊 Total Samples", len(y_pred))
                
                with col2:
                    good_count = (y_pred == 0).sum()
                    st.metric("✅ Good Credit", good_count)
                
                with col3:
                    bad_count = (y_pred == 1).sum()
                    st.metric("❌ Bad Credit", bad_count)
                
                with col4:
                    avg_confidence = np.mean(y_pred_proba)
                    st.metric("🎯 Avg Confidence", f"{avg_confidence:.2%}")
                
                # Performance metrics if target is available
                if has_target:
                    st.markdown("---")
                    st.markdown("### 🎯 Performance on Uploaded Data")
                    
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        acc = accuracy_score(y_true, y_pred)
                        st.metric("Accuracy", f"{acc:.4f}")
                    
                    with col2:
                        prec = precision_score(y_true, y_pred)
                        st.metric("Precision", f"{prec:.4f}")
                    
                    with col3:
                        rec = recall_score(y_true, y_pred)
                        st.metric("Recall", f"{rec:.4f}")
                    
                    with col4:
                        f1 = f1_score(y_true, y_pred)
                        st.metric("F1 Score", f"{f1:.4f}")
                    
                    # Confusion matrix
                    st.markdown("### 📊 Confusion Matrix on Uploaded Data")
                    from sklearn.metrics import confusion_matrix
                    cm = confusion_matrix(y_true, y_pred)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax,
                               xticklabels=['Good Credit', 'Bad Credit'],
                               yticklabels=['Good Credit', 'Bad Credit'])
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title('Confusion Matrix - Uploaded Data')
                    st.pyplot(fig)
                
                # Download predictions
                st.markdown("---")
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name=f"predictions_{selected_model_name.replace(' ', '_').lower()}.csv",
                    mime="text/csv"
                )
            
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.info("Please ensure your CSV has the correct format (20 features, optional 'Target' column)")
        
        else:
            st.info("👆 Please upload a CSV file to make predictions")
            
            # Show example format
            with st.expander("📋 Example CSV Format"):
                st.markdown("""
                Your CSV should have the following columns (with meaningful names):
                
                **Numerical Features (7):**
                - Duration, CreditAmount, InstallmentRate, ResidenceDuration, Age, ExistingCredits, NumDependents
                
                **Categorical Features (13):**
                - Status, CreditHistory, Purpose, Savings, Employment, PersonalStatusSex, 
                  OtherDebtors, Property, OtherInstallmentPlans, Housing, Job, Telephone, ForeignWorker
                
                **Optional:**
                - Target (0=Good Credit, 1=Bad Credit)
                
                You can use the `data/test_data.csv` file generated during training.
                """)
    
    else:
        st.error("⚠️ Models not loaded. Please run the training script first.")

# Footer
st.markdown("---")