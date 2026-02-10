"""
Docstring for data_utils
Data utilities for loading and preprocessing the Statlog German Credit Data
data source - https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import os


RANDOM_SEED = 42
TEST_RATIO = 0.3  # test ratio default, for split data

data_path = 'data/german.data'
feature_name_path = 'model/feature_names.pkl'
test_data_path = 'data/test_data.csv'

# Meaningful column names for German Credit Data 
# This need to be done as:
# All attribute names have been changed to meaningless symbols to protect confidentiality of the data.
COLUMN_NAMES = [
    "Status", "Duration", "CreditHistory", "Purpose", "CreditAmount",
    "Savings", "Employment", "InstallmentRate", "PersonalStatusSex",
    "OtherDebtors", "ResidenceDuration", "Property", "Age",
    "OtherInstallmentPlans", "Housing", "ExistingCredits",
    "Job", "NumDependents", "Telephone", "ForeignWorker", "Target"
]
TARGET_COL_NAME = "Target"

# Categorical & Numerical Columns
CATEGORICAL_COLS = [
    "Status", "CreditHistory", "Purpose", "Savings", "Employment",
    "PersonalStatusSex", "OtherDebtors", "Property",
    "OtherInstallmentPlans", "Housing", "Job", "Telephone", "ForeignWorker"
]

NUMERICAL_COLS = [
    "Duration", "CreditAmount", "InstallmentRate",
    "ResidenceDuration", "Age", "ExistingCredits", "NumDependents"
]

n_features = None
n_samples = None


###

def load_data(filepath = data_path):
    """
    load the statlog german credit data from the local file in filepath    
    :param filepath: path to the german data file
    
    reutrns 
    X (pd.DataFrame): features with meaningful column names
    y (np.array): target variable
    """
    
    try:
        global n_features
        global n_samples

        isloaded = False
        print(f'loading stalog german credit data from {filepath} ...')
        
        # load data (space separated, no headers)
        df = pd.read_csv(filepath,  sep=r'\s+', header=None)
        isloaded = True

        # add meaningful column names
        df.columns = COLUMN_NAMES

        # features and target
        X = df.drop('Target', axis=1)
        y = df['Target'].values

        n_samples = X.shape[0]
        n_features  = X.shape[1]
        assert len(COLUMN_NAMES) == len(df.columns), "number of columns is not matching!!!"

        print(f'dataset loading: {'successful' if  isloaded else 'unsuccessful'}')
        print(f'    features:{n_features}')
        print(f'    instances: {n_samples}')
        #print(f'    columns: {X.columns} ')    
        print(f'    number of columns : {len(COLUMN_NAMES)}')
        #print()
        print(f'{df.head()}')

        return X, y
    
    except FileNotFoundError:
        print(f'error: file not foundat {filepath}')
        raise
    except Exception as e:
        print(f'error loading data: {e}')
        raise


def get_dataset_info():
    """
    returns 
    dict: the dataset information
    """
    return {
        'name': 'Statlog (German Credit Data)',
        'source': 'UCI Machine Learning Repository',
        'url': 'https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data',
        'instances': n_samples,
        'features': n_features,
        'task': 'Binary Classification',
        'target': 'Credit Risk (Good/Bad)',
        'missing_values': False,
        'categorical_features': len(CATEGORICAL_COLS),
        'numerical_features': len(NUMERICAL_COLS)
    }

def get_feature_types(X):
    """
    identify the categorical and numerical features
    
    :param X (pd.DataFrame): input features
    
    returns
    categorical (list): categorical column names
    numerical  (list): numerical column names
    """
    categorical = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical = X.select_dtypes(include=[np.number]).columns.tolist()

    return categorical, numerical

def save_feature_names(categorical, numerical, filepath=feature_name_path):
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    feature_names = {
        'categorical': categorical,
        'numerical': numerical,
        'all_features': categorical + numerical
    }

    joblib.dump(feature_names, filepath, compress=3)
    print(f'feature name saved to: {filepath}')

def load_feature_names(filepath=feature_name_path):
    
    feature_names = joblib.load(filepath)

    return feature_names['categorical'], feature_names['numerical']


def get_feature_desc():
    
    return {
        'Status': 'status of existing checking account (categorical)',
        'Duration': 'duration in months (numerical)',
        'CreditHistory': 'credit history (categorical)',
        'Purpose': 'purpose of credit (categorical)',
        'CreditAmount': 'credit amount (numerical)',
        'Savings': 'savings account/bonds (categorical)',
        'Employment': 'present employment since (categorical)',
        'InstallmentRate': 'installment rate % of disposable income (numerical)',
        'PersonalStatusSex': 'personal status and sex (categorical)',
        'OtherDebtors': 'other debtors/guarantors (categorical)',
        'ResidenceDuration': 'present residence since (numerical)',
        'Property': 'property (categorical)',
        'Age': 'age in years (numerical)',
        'OtherInstallmentPlans': 'other installment plans (categorical)',
        'Housing': 'housing (categorical)',
        'ExistingCredits': 'number of existing credits at this bank (numerical)',
        'Job': 'job (categorical)',
        'NumDependents': 'number of people being liable to provide maintenance for (numerical)',
        'Telephone': 'telephone (categorical)',
        'ForeignWorker': 'foreign worker (categorical)',
        'Target': 'credit risk: 1=good, 2=bad (converted to 0=good, 1=bad)'
    }


def change_target_class(y):
    """
    For credit risk, the positive class should be Bad Credit (1) because:
    - We care about identifying risky customers
    - Recall = "of all bad credits, how many did we catch?"
    - Precision = "of those we flagged as bad, how many were actually bad?"
    This is the standard in risk assessment
    So 1 = bad, treating Bad Credit as the positive class.
    """
    y_binary = y -1 # convert 1=good, 2=bad to 0=good, 1=bad
    return y_binary

def split_data(X, y, test_size=TEST_RATIO, random_state = RANDOM_SEED):
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def save_test_data(X_test, y_test, filepath=test_data_path, index=False ):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    test_data = X_test.copy()
    test_data[TARGET_COL_NAME] = y_test
    test_data.to_csv(filepath, index=index)

def load_test_data(filepath=test_data_path):
    df = pd.read_csv(filepath)

    if TARGET_COL_NAME in df.columns:
        y_test = df[TARGET_COL_NAME].values
        X_test = df.drop(TARGET_COL_NAME, axis=1)
        return X_test, y_test
    else:
        return df, None

