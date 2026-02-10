"""
Docstring for main
script to do quick runs and check if everything is working fine - instead of having to run everything 
"""

import sys
import os
from pprint import pprint

from src.data_utils import (
    load_data,
    get_dataset_info,
    get_feature_types,
    save_feature_names,
    load_feature_names,
    get_feature_desc,
    change_target_class,
    split_data,
    save_test_data,
    load_test_data,
)

data_path = './data/german.data'

def main():
    """
    quick test and check 
    """
    print(f'\nquick test & verify')

    # Load data from local file
    print(f'\n[1] Loading dataset ...')
    X, y = load_data(filepath= data_path)
    
    info = get_dataset_info()
    pprint(info)
    
    column_desc = get_feature_desc()
    pprint(column_desc, width= 90)

    print(f'\n[2] Feature Types')
    categorical, numerical = get_feature_types(X)
    print(f'    categorical ({len(categorical)}): {categorical}')
    print(f'    numerical ({len(numerical)}): {numerical}')

    print(f'\n[3] Save feature names for later use')
    #save_feature_names(categorical, numerical)

    print(f'\n[4] Change target class to binary class 0=good, 1=bad')
    print(f'before:\n {y[:5]}')
    y_binary = change_target_class(y)
    print(f'after:\n {y_binary[:5]}')

    print(f'\n[5] Split data')
    X_train, X_test, y_train, y_test = split_data(X, y_binary)
    print(X_train.head())
    print(y_train[:5])
    print(X_test.head())
    print(y_test[:5])

    # Save original test data (for Streamlit upload)
    # pipeline will transform it properly
    print(f'\nsaving test data')
    print(X_test.head())
    print(y_test[:10])

    save_test_data(X_test, y_test)
    print(f'saved the split test data for streamlit upload')

    # load test data
    X_test_after, y_test_after = load_test_data()
    print(f'\nloaded test data')
    print(X_test_after.head())
    print(y_test_after[:10])

    assert X_test.columns.equals(X_test_after.columns)
    assert X_test.dtypes.equals(X_test_after.dtypes)



if __name__ == "__main__":
    main()