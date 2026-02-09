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

data_path = 'data/german.data'

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

        print(f'dataset loading: {'successful' if  isloaded else 'unsuccessful'}')
        print(f'  features:{X.shape[1]}')
        print(f'  instances: {X.shape[0]}')
        print(f'  columns: {X.columns} ')    

        return X, y
    
    except FileNotFoundError:
        print(f'error: file not foundat {filepath}')
        raise
    except Exception as e:
        print(f'error loading data: {e}')
        raise