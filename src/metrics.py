from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    precision_score,
    recall_score, 
    f1_score, 
    matthews_corrcoef,
    confusion_matrix, 
    classification_report
)

import numpy as np
import pandas as pd
import joblib
import os


def compute_metrics(y_true, y_pred, y_pred_proba):
    metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='binary'),
            'Recall': recall_score(y_true, y_pred, average='binary'),
            'F1': f1_score(y_true, y_pred, average='binary'),
            'AUC': roc_auc_score(y_true, y_pred_proba),
            'MCC': matthews_corrcoef(y_true, y_pred)
        }
        
    return metrics    

def get_classification_report(y_true, y_pred)
    return classification_report(y_true, y_pred, target_names=['Good Credit', 'Bad Credit'])

def evaluate_model(model, X_test, y_test):
    # make predictions
    y_pred = model.predict(X_test)
    
    """
    model.predict_proba returns class probabilities:
    [[0.12, 0.88],
    [0.73, 0.27],
    [0.45, 0.55]]
    [:,1] - selects the probability of positive class bad credit (class index 1)
    [0.88, 0.27, 0.55]
    pos_class_index = list(model.classes_).index(1)
    y_pred_proba = model.predict_proba(X_test)[:, pos_class_index]
    """
    y_pred_proba = model.predict_proba(X_test)[:,1] # identifying bad credit as positive class

    metrics = compute_metrics(y_test, y_pred, y_pred_proba)

    # confusion matrix and report
    cm = confusion_matrix(y_test, y_pred)
    report  = get_classification_report(y_test, y_pred)
    
    results = {
        'Accuracy': metrics['Accuracy'],
        'Precision': metrics['Precision'],
        'Recall': metrics['Recall'],
        'F1': metrics['F1'],
        'AUC': metrics['AUC'],
        'MCC': metrics['MCC'],
        'confusion_matrix': cm,
        'classification_report': report    
        }
    
    return results

def evaluate_models(models, X_test, y_test):
    all_results={}
    
    for name, model in models.items():
        results = evaluate_model(model, X_test, y_test)
        all_results[name] = results
    
    return all_results





"""
Evaluation metrics for classification models
FINAL VERSION - Uses joblib
"""

from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    precision_score,
    recall_score, 
    f1_score, 
    matthews_corrcoef,
    confusion_matrix, 
    classification_report
)
import numpy as np
import pandas as pd
import joblib
import os


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """
    Calculate all evaluation metrics
    
    Args:
        y_true (np.array): True labels
        y_pred (np.array): Predicted labels
        y_pred_proba (np.array): Prediction probabilities
        
    Returns:
        dict: Dictionary containing all metrics
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_pred_proba),
        'Precision': precision_score(y_true, y_pred, average='binary'),
        'Recall': recall_score(y_true, y_pred, average='binary'),
        'F1': f1_score(y_true, y_pred, average='binary'),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    
    return metrics


def get_confusion_matrix(y_true, y_pred):
    """
    Calculate confusion matrix
    
    Args:
        y_true (np.array): True labels
        y_pred (np.array): Predicted labels
        
    Returns:
        np.array: Confusion matrix
    """
    return confusion_matrix(y_true, y_pred)


def get_classification_report(y_true, y_pred):
    """
    Generate classification report
    
    Args:
        y_true (np.array): True labels
        y_pred (np.array): Predicted labels
        
    Returns:
        str: Classification report
    """
    return classification_report(y_true, y_pred, target_names=['Good Credit', 'Bad Credit'])


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a model and return all metrics
    
    Args:
        model: Trained model or Pipeline
        X_test (pd.DataFrame): Test features
        y_test (np.array): Test labels
        
    Returns:
        dict: Dictionary containing all evaluation results
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    
    # Get confusion matrix and report
    cm = get_confusion_matrix(y_test, y_pred)
    report = get_classification_report(y_test, y_pred)
    
    results = {
        'Accuracy': metrics['Accuracy'],
        'AUC': metrics['AUC'],
        'Precision': metrics['Precision'],
        'Recall': metrics['Recall'],
        'F1': metrics['F1'],
        'MCC': metrics['MCC'],
        'confusion_matrix': cm,
        'classification_report': report
    }
    
    return results


def evaluate_all_models(models, X_test, y_test):
    """
    Evaluate all models and return results
    
    Args:
        models (dict): Dictionary of trained models/pipelines
        X_test (pd.DataFrame): Test features
        y_test (np.array): Test labels
        
    Returns:
        dict: Dictionary of evaluation results for all models
    """
    all_results = {}
    
    print("\nEvaluating all models...")
    for name, model in models.items():
        print(f"Evaluating {name}...")
        results = evaluate_model(model, X_test, y_test)
        all_results[name] = results
        print(f"  Accuracy: {results['Accuracy']:.4f} | AUC: {results['AUC']:.4f} | F1: {results['F1']:.4f}")
    
    return all_results


def comparison_as_df(results):
    comparison_data = []
    
    for model_name, metrics in results.items():
        comparison_data.append({
            'Model Name': model_name,
            'Accuracy': metrics['Accuracy'],
            'Precision': metrics['Precision'],
            'Recall': metrics['Recall'],
            'F1': metrics['F1'],
            'AUC': metrics['AUC'],
            'MCC': metrics['MCC']
        })
    
    df = pd.DataFrame(comparison_data)
    return df

def save_results(results, filepath='./model/model_results.pkl'):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(results, filepath, compress=3)
    print('results saved to:{filepath}')

def load_results(filepath='models/model_results.pkl'):
    results = joblib.load(filepath)
    print('loaded results from:{filepath}')
    return results