#  💳 Credit Risk Classification

## 📋 Problem Statement
**Objective**: Classify individuals as **good or bad credit risks** based on their financial and personal attributes using multiple machine learning models. This project implements multiple machine learning classification models to predict credit risk based on the **Statlog (German Credit Data)**.  
This binary classification problem has significant real-world applications in banking and financial services for assessing loan applications and managing credit risk.

## 🐙 Github Repository
**URL:** https://github.com/2rajeevmv/klsfi  

## 🧠 Live Streamlit App
**URL:** https://2rajeevmvklsfi.streamlit.app/  

## 📊 Dataset Description
**Dataset Name:** Statlog (German Credit Data)  
**Source:** UCI Machine Learning Repository  
**Dataset ID:** 144  
**URL:** https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data  
**Local File:** `./data/german.data`

### Dataset Characteristics:
- **Number of Instances:** 1000
- **Number of Features:** 20
- **Feature Types:** Categorical (13) and Numerical (7)
- **Missing Values:** None
- **Class Distribution:** Imbalanced (70% Good, 30% Bad)
- **Task:** Binary Classification
- **Target Variable:** Binary (1 = Good Credit, 2 = Bad Credit → converted to 0 = Good Credit, 1 = Bad Credit)  
    For credit risk classification, the positive class (1) is Bad Credit because:
    * We care about identifying risky customers
    * Recall = "of all bad credits, how many did we catch?"
    * Precision = "of those we flagged as bad, how many were actually bad?"  



### Features (Meaningful Names):

| Feature | Type | Description |
|---------|------|-------------|
| Status | Categorical | Status of existing checking account |
| Duration | Numerical | Duration in months |
| CreditHistory | Categorical | Credit history |
| Purpose | Categorical | Purpose of credit |
| CreditAmount | Numerical | Credit amount |
| Savings | Categorical | Savings account/bonds |
| Employment | Categorical | Present employment since |
| InstallmentRate | Numerical | Installment rate % of disposable income |
| PersonalStatusSex | Categorical | Personal status and sex |
| OtherDebtors | Categorical | Other debtors/guarantors |
| ResidenceDuration | Numerical | Present residence since |
| Property | Categorical | Property |
| Age | Numerical | Age in years |
| OtherInstallmentPlans | Categorical | Other installment plans |
| Housing | Categorical | Housing |
| ExistingCredits | Numerical | Number of existing credits |
| Job | Categorical | Job |
| NumDependents | Numerical | Number of dependents |
| Telephone | Categorical | Telephone |
| ForeignWorker | Categorical | Foreign worker |


### Data Preprocessing:
- **ColumnTransformer** used for proper preprocessing
- **OneHotEncoder** for categorical variables (13 features)
- **StandardScaler** for numerical features (7 features)
- Train-test split: **70-30** ratio with stratified
- Target variable converted from (1=Good, 2=Bad) to (0=Good, 1=Bad)

## 🎯 Models Used

Six classification models implemented as complete scikit-learn Pipelines:

1. **Logistic Regression** - Linear classification baseline
2. **Decision Tree Classifier** - Non-linear decision boundary
3. **K-Nearest Neighbors (KNN)** - Instance-based learning
4. **Naive Bayes (Gaussian)** - Probabilistic classifier
5. **Random Forest** - Ensemble of decision trees
6. **XGBoost** - Gradient boosting ensemble

| Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC | TN | FP | FN | TP |
|--------------|----------|-----|-----------|--------|----|----|------|-----|---|----|
| Logistic Regression | 0.783333 |   0.676056  |0.533333 | 0.596273 |  0.801058 |  0.456935 |  187 |  23 |  42 |  48 |
| Decision Tree | 0.670000 | 0.453608 | 0.488889 | 0.470588 |  0.609074 |  0.231709 |  157 | 53 |  46 |  44 |
| K-Nearest Neighbors | 0.753333 |  0.642857 |  0.400000 |  0.493151 |  0.752328  | 0.358429 |  190 |  20  | 54 |  36 |
| Naive Bayes |  0.696667 |  0.495935 |  0.677778 |  0.572770  | 0.726138  | 0.356425  |148 |  62 |  29|  61|
| Random Forest (Ensemble) | 0.760000 |   0.687500 | 0.366667 |  0.478261 |  0.802063 |  0.369048  |195 |  15 |  57 |  33|
| XGBoost (Ensemble) | 0.730000 |   0.567164 |  0.422222 |  0.484076 |  0.779788 |  0.312628 |  181 | 29 |  52 |  38 |

Confusion matrix is raveled as: TN, FP, FN, TP   
**Confusion Matrix Terms:**
* **TN** (True Negative): Good Credit correctly predicted as Good Credit  
* **FP** (False Positive): Good Credit incorrectly predicted as Bad Credit  
* **FN** (False Negative): Bad Credit incorrectly predicted as Good Credit  
* **TP** (True Positive): Bad Credit correctly predicted as Bad Credit  

| Metric | Formula | Interpretation | Range |
|--------|---------|----------------|-------|
| **Accuracy** | (TP + TN) / Total | Overall correctness | [0, 1] |
| **AUC** | Area under ROC curve | Discrimination ability | [0, 1] |
| **Precision** | TP / (TP + FP) | Positive prediction accuracy | [0, 1] |
| **Recall** | TP / (TP + FN) | True positive coverage | [0, 1] |
| **F1 Score** | 2 × (Prec × Rec) / (Prec + Rec) | Harmonic mean | [0, 1] |
| **MCC** | Correlation coefficient | Balanced measure | [-1, 1] |

### Model Performance Observations
#### Key Summary Points:
* **Best Overall: Logistic Regression** (78.33% accuracy - highest, 0.457 MCC - highest)  
* Worst Performer: Decision Tree (67% accuracy, 0.232 MCC - lowest)  
* Highest Precision: Naive Bayes (0.678) - but very conservative 
* Best Balance: Logistic Regression - highest MCC indicates most reliable predictions
* Best at detecting Bad Credit: Naive Bayes (highest TP = 61)
* Best at detecting Good Credit: Random Forest (highest TN = 195)
* Surprising Result: XGBoost underperforms, Logistic Regression wins  
<br>
<br> 

| Model Name | Observation about model performance |
|--------------|-------------------------------------|
| Logistic Regression | Logistic Regression achieves the highest accuracy of 78.33% with a strong AUC of 0.676. The MCC of 0.457 (highest among all models) indicates the most reliable and balanced predictions across both classes. The model is the __most robust linear classifier for this credit risk problem__.|
| Decision Tree | Decision Tree shows the poorest performance with only 67% accuracy and lowest AUC of 0.454. The lowest MCC 0.2317 suggest weaker generalization and class discrimination. |
| K-Nearest Neighbors | KNN achieves 75.33%, good overall accuracy with moderate AUC of 0.643. But it shows extremely low precision (0.400) for Bad Credit. The MCC of 0.358 suggests moderate but inconsistent performance. The high-dimensional space created by OneHotEncoding (13 categorical features) may be affecting KNN's distance calculations, causing it to struggle with the curse of dimensionality. |
| Naive Bayes | Naive Bayes achieves the highest precision (0.678) among all models and highest TP count(61), meaning good at identifying Bad Credit. The high FP (62) reduces overall reliability. The confusion matrix reveals it's risk-averse, which might be acceptable in conservative lending scenarios.|
| Random Forest | Random Forest achieves 76% accuracy with good AUC of 0.688, demonstrating strong ensemble benefits at identifying Good Credit. However, it exhibits very low precision (0.367) for Bad Credit which limits its effectiveness. |
| XGBoost | XGBoost achieves 73% accuracy with moderate AUC of 0.567. Better than Decision Tree but still underperforming for an ensemble model. Simpler model like Logistic Regression outperforming it might mean that credit risk patterns may be more linea than expected, or the dataset may be too small for it to show its advantages. |

## 🏗️ Project Structure

```
klsfi/
│
├── app.py                          # Streamlit web application
├── main.py                         # Training script (run on BITS Virtual Lab)
├── requirements.txt                # Python dependencies
├── environment.yml                 # conda environment yaml
├── README.md                       # Project documentation
├── .gitignore                      # Git ignore rules
│
├── src/                            # Source code modules
│   ├── __init__.py                 # Package initializer
│   ├── data_utils.py               # Data loading with meaningful column names
│   ├── pipeline.py                 # build_preprocessor() & build_model()
│   └── metrics.py                  # Evaluation metrics
│
├── model/                          # Saved model files (generated after training)
│   ├── logistic_regression_model.pkl
│   ├── decision_tree_model.pkl
│   ├── k_nearest_neighbors_model.pkl
│   ├── naive_bayes_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── feature_names.pkl           # Categorical & numerical feature lists
│   └── model_results.pkl           # Evaluation results
│
└── data/                           # Data files
    ├── german.data                 # Original UCI data file
    └── test_data.csv               # Generated test data for Streamlit upload
```
### Prerequisites
python==3.12.12  
streamlit==1.54.0  
numpy  
pandas  
scikit-learn  
matplotlib  
seaborn  
xgboost  
joblib  

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Statlog German Credit Data
- Prof. Hans Hofmann for the original dataset
- scikit-learn and XGBoost communities
- Streamlit for the deployment platform

## 📄 License

This project is created for academic purposes 