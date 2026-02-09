"""
Docstring for main
script to do quick runs and check if everything is working fine - instead of having to run everything 
"""

import sys
import os

from src.data_utils import (
    load_data,
)

data_path = './data/german.data'

def main():
    """
    quick test and check 
    """
    print("testing and verifying")

    # Load data from local file
    print("\n[STEP 1] Loading dataset from local file {data_path}...")
    X, y = load_data(filepath= data_path)



if __name__ == "__main__":
    main()