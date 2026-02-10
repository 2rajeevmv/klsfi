
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import os
import time

# Model Indices
LOGISTIC_REGRESSION = 0
DECISION_TREE = 1
KNN = 2
NAIVE_BAYES = 3
RANDOM_FOREST = 4
XGBOOST = 5
ALL_MODELS = [
    LOGISTIC_REGRESSION, 
    DECISION_TREE,
    KNN,
    NAIVE_BAYES,
    RANDOM_FOREST,
    XGBOOST,
    ]

MODEL_NAMES = [
        'Logistic Regression',
        'Decision Tree',
        'K-Nearest Neighbors',
        'Naive Bayes',
        'Random Forest',
        'XGBoost'
]

PREPROCESSOR = 'preprocessor'
CLASSIFIER = 'classifier'


def get_model_names():
    return MODEL_NAMES

def get_model_name(index):
    return MODEL_NAMES[index]

def pipeline_model_name(pipeline):
    model = pipeline.named_steps['classifier']
    model_name = model.__class__.__name__
    return model_name


def selected_model_names(models=ALL_MODELS):
    sel_model_names = [MODEL_NAMES[idx] for idx in models]
    return sel_model_names

def build_preprocessor(categorical, numerical):

    categorical_xfrmer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    numerical_xfrmer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_xfrmer, numerical),
            ('cat', categorical_xfrmer, categorical)
        ],
        remainder='passthrough'
    )

    return preprocessor


def build_model(model_name, **kwargs):
    
    models = {
        MODEL_NAMES[LOGISTIC_REGRESSION]: LogisticRegression(
            max_iter=kwargs.get('max_iter', 1000),
            random_state=kwargs.get('random_state', 42)
        ),
        MODEL_NAMES[DECISION_TREE]: DecisionTreeClassifier(
            max_depth=kwargs.get('max_depth', 10),
            random_state=kwargs.get('random_state', 42)
        ),
        MODEL_NAMES[KNN]: KNeighborsClassifier(
            n_neighbors=kwargs.get('n_neighbors', 5)
        ),
        MODEL_NAMES[NAIVE_BAYES]: GaussianNB(),
        MODEL_NAMES[RANDOM_FOREST]: RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 10),
            random_state=kwargs.get('random_state', 42)
        ),
        MODEL_NAMES[XGBOOST]: XGBClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 6),
            learning_rate=kwargs.get('learning_rate', 0.1),
            random_state=kwargs.get('random_state', 42),
            eval_metric='logloss'
        )
    }

    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not supported. Available: {list(models.keys())}")
    
    model = models[model_name]

    return model

    

def build_model_pipeline(model_name, categorical, numerical, **kwargs):
    """
    build the complete scikit-learn pipeline with preprocessing and model
    
    :param model_name (str): name of the model
    :param categorical (list): categorical column names
    :param numerical (list): numerical column names
    :param kwargs: model parameters - keyword arguments {key: value}
    """
    assert model_name in MODEL_NAMES, f'{model_name} model is not available'

    preprocessor = build_preprocessor(categorical, numerical)
    model = build_model(model_name, **kwargs)
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    return pipeline

def build_models(categorical, numerical, sel_models=ALL_MODELS ):

    models = {}
    sel_model_names = selected_model_names(sel_models)

    for name in sel_model_names:
        print(f'building {name} model pipeline')
        models[name] = build_model_pipeline(name, categorical, numerical)
    
    return models

def train_model(model_pipeline, X_train, y_train):

    model_class_name = pipeline_model_name(model_pipeline)

    print(f'\nstarted: training {model_class_name} model')
    start_time = time.time()

    model_pipeline.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    mins = int(training_time // 60)
    secs = int(training_time % 60)
    print(f'completed: training {model_class_name} model in {mins} {secs} secs')

    return model_pipeline


def train_models(X_train, y_train, categorical, numerical, sel_models=ALL_MODELS):
    all_trained = {}

    pipelines = build_models(categorical, numerical, sel_models)
    
    for name, model_pipeline in pipelines.items():
        trained = train_model(model_pipeline, X_train, y_train)
        all_trained[name] = trained
    
    return all_trained
