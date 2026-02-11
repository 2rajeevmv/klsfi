
"""
Source code package
Modules:
- data_utils: data loading and preprocessing
- pipeline: model building with build_preprocessor() and build_model()
- metrics: model evaluation and visualization
"""

__version__ = "1.0.0"


# -------------------------------
# data_utils exports
# -------------------------------
from .data_utils import (
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
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
)

# -------------------------------
# pipeline exports
# -------------------------------
from .pipeline import (
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
    build_preprocessor,
    build_model,
    build_model_pipeline,
    train_model,
    train_models,
    save_model,
    save_models,
    load_model,
    load_models,
    predict,
    get_model_parameters,
)

# -------------------------------
# metrics exports
# -------------------------------
from .metrics import (
    compute_metrics,
    evaluate_model,
    evaluate_models,
    comparison_as_df,
    save_results,
    load_results,
    print_model_summary,
)

__all__ = [
     # data utils
     'load_data',
    'get_dataset_info',
    'get_feature_types',
    'save_feature_names',
    'load_feature_names',
    'get_feature_desc',
    'change_target_class',
    'split_data',
    'save_test_data',
    'load_test_data',
    'CATEGORICAL_COLS',
    'NUMERICAL_COLS',

    # pipeline
    'LOGISTIC_REGRESSION',
    'DECISION_TREE',
    'KNN',
    'NAIVE_BAYES',
    'RANDOM_FOREST',
    'XGBOOST',
    'ALL_MODELS',
    'MODEL_NAMES',
    'get_model_names',
    'get_model_name',
    'get_model_kls_name',
    'build_preprocessor',
    'build_model',
    'build_model_pipeline',
    'train_model',
    'train_models',
    'save_models',
    'save_model',
    'load_models',
    'predict',
    'get_model_parameters',
    
    # metrics
    'compute_metrics',
    'comparison_as_df',
    'evaluate_model',
    'evaluate_models',
    'save_results',
    'load_results',
    'print_model_summary',
]