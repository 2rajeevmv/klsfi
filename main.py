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

from src.pipeline import (
    LOGISTIC_REGRESSION,
    DECISION_TREE,
    KNN,
    NAIVE_BAYES,
    RANDOM_FOREST,
    XGBOOST,
    ALL_MODELS,
    MODEL_NAMES,
    get_model_names,
    get_model_name,
    get_model_kls_name,
    train_models,
    save_models,
    load_models,
    get_model_parameters,
)

from src.metrics import (
    evaluate_models,
    comparison_as_df,
    save_results,
    load_results,
    print_model_summary,

)

data_path = './data/german.data'

initial_models = [LOGISTIC_REGRESSION, KNN]

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
    #TODO: remove commenting -after

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
    print(f'\n[6] Saving test data')
    print(X_test.head())
    print(y_test[:10])

    #TODO:remove commenting - after 
    save_test_data(X_test, y_test)
    print(f'saved the split test data for streamlit upload')

    # load test data
    X_test_after, y_test_after = load_test_data()
    print(f'\nloaded test data')
    print(X_test_after.head())
    print(y_test_after[:10])

    assert X_test.columns.equals(X_test_after.columns)
    assert X_test.dtypes.equals(X_test_after.dtypes)
    model_names = get_model_names()
    print(f'{model_names}, {len(model_names)}')
    model_name = get_model_name(KNN)
    print(f'KNN model Name: {model_name} {KNN}')

    print(f'{ALL_MODELS}')
    print(f'{MODEL_NAMES}')


    print(f'\n[7] main:training on model(s): ')
    """
    print(f'{initial_models}')
    for i in initial_models:
        print(f'{i} in initial models')
        print(f'{MODEL_NAMES[i]} model')
    """
    #sel_models = [MODEL_NAMES[i] for i in initial_models]
    #print(f'{sel_models}')
    """
    for name in sel_models:
        print(f'\nbuilding {name} model pipeline')
        #models[name] = build_model_pipeline(name, categorical, numerical)
    """

    trained_pipelines = train_models(X_train, y_train, categorical, numerical)
    #train_models(X_train, y_train, categorical, numerical, initial_models)
    
    print(f'\n[8]Saving trained pipelines using joblib')
    save_models(trained_pipelines)

    print(f'\n[9]Loading trained model pipelines using joblib ')
    #loaded_models = load_models(initial_models)
    loaded_models = load_models()
    for name, model in loaded_models.items():
        print(f'loaded {name} model pipeline {get_model_kls_name(model)}')

    print(f'\n[9] evaluate models')
    results = evaluate_models(trained_pipelines, X_test, y_test)
    
    print(f'\n[10] Saving evaluation results...')
    save_results(results)

    print("\n[11] Results Summary")
    results_df = comparison_as_df(results)
    print(results_df)

    best_metric = 'AUC'
    print(f'results in descending sort order of {best_metric}')
    sorted_df = results_df.sort_values(by=best_metric, ascending=False)
    print(sorted_df)
    print(f'\nFor credit risk, the positive class is Bad Credit(1) \
            \n-ve class - Good Credit(0) \
            \n+ve class - Bad Credit(1) \
            \nTN : Good Credit correctly predicted as Good Credit\
            \nFP : Good Credit predicted as Bad Credit\
            \nFN : Bad Credit predicted as Good Credit \
            \nTP : Bad Credit correctly predicted as Bad Credit\
          ')

    #print(f'\n[12] Print detailed summary')
    #for name, result in results.items():
        #print_model_summary(name, results[name])
    print(f'\nShow model configuration parametrs')
    for name in MODEL_NAMES:
        params = get_model_parameters(name)
        if params:
            print(f'model name {name}: {params}')
        else:
            print('default')
        

if __name__ == "__main__":
    main()