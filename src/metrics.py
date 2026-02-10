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

def get_classification_report(y_true, y_pred):
    return classification_report(y_true, y_pred, target_names=['Good Credit', 'Bad Credit'])

def evaluate_model(model, X_test, y_test):
    # make predictions
    y_pred = model.predict(X_test)
    
    """
    For credit risk, the positive class should be Bad Credit (1) because:
    - We care about identifying risky customers
    - Recall = "of all bad credits, how many did we catch?"
    - Precision = "of those we flagged as bad, how many were actually bad?"
    This is the standard in risk assessment
    So 1 = bad, treating Bad Credit as the positive class.
    """
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

def comparison_as_df(results):
    comparison_data = []

    for model_name, metrics in results.items():
        cm = metrics['confusion_matrix']
        """
        For credit risk, the positive class is Bad Credit(1)
        -ve class - Good Credit(0)
        +ve class - Bad Credit(1)
        TN : Good Credit correctly predicted as Good Credit
        FP : Good Credit predicted as Bad Credit
        FN : Bad Credit predicted as Good Credit
        TP : Bad Credit correctly predicted as Bad Credit
        """
        TN, FP, FN, TP = cm.ravel()
        comparison_data.append({
            'Model Name': model_name,
            'Accuracy': metrics['Accuracy'],
            'Precision': metrics['Precision'],
            'Recall': metrics['Recall'],
            'F1': metrics['F1'],
            'AUC': metrics['AUC'],
            'MCC': metrics['MCC'],
            'TN': TN,
            'FP': FP,
            'FN': FN,
            'TP': TP
        })
    
    df = pd.DataFrame(comparison_data)
    return df

def save_results(results, filepath='./model/model_results.pkl'):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(results, filepath, compress=3)
    print(f'results saved to:{filepath}')

def load_results(filepath='./model/model_results.pkl'):
    results = joblib.load(filepath)
    print(f'loaded results from:{filepath}')
    return results

def pretty_confusion_matrix(cm):
    df = pd.DataFrame(
            cm,
            index=["Actual:Good(0)", "Actual:Bad(1)"],
            columns=['Predicted:Good(0)', 'Predicted:Bad(1)']
    )

    return df

def print_model_summary(name, results):
    print(f'{name} - Detailed Results')
    
    print(f'Performance Metrics:')
    print(f'  Accuracy:  {results["Accuracy"]:.4f}')
    print(f'  Precision: {results["Precision"]:.4f}')
    print(f'  Recall:    {results["Recall"]:.4f}')
    print(f'  F1 Score:  {results["F1"]:.4f}')
    print(f'  AUC Score: {results["AUC"]:.4f}')
    print(f'  MCC Score: {results["MCC"]:.4f}')
    
    print(f"\nConfusion Matrix:")
    print(pretty_confusion_matrix(results['confusion_matrix']))
    
    #print(f"\nClassification Report:")
    #print(results['classification_report'])
    