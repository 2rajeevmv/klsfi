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
    get_feature_desc
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


if __name__ == "__main__":
    main()