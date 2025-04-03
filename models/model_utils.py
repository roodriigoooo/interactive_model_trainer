"""
Model utilities for the ML Model Trainer application.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import joblib
import os

# Dictionary of available models
CLASSIFICATION_MODELS = {
    "Logistic Regression": {
        "model": "LogisticRegression",
        "module": "sklearn.linear_model",
        "params": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
            "max_iter": [1000]
        }
    },
    "Random Forest": {
        "model": "RandomForestClassifier",
        "module": "sklearn.ensemble",
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    },
    "Gradient Boosting": {
        "model": "GradientBoostingClassifier",
        "module": "sklearn.ensemble",
        "params": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7]
        }
    },
    "Support Vector Machine": {
        "model": "SVC",
        "module": "sklearn.svm",
        "params": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["linear", "rbf"],
            "probability": [True]
        }
    }
}

REGRESSION_MODELS = {
    "Linear Regression": {
        "model": "LinearRegression",
        "module": "sklearn.linear_model",
        "params": {}
    },
    "Ridge Regression": {
        "model": "Ridge",
        "module": "sklearn.linear_model",
        "params": {
            "alpha": [0.01, 0.1, 1.0, 10.0],
            "solver": ["auto"]
        }
    },
    "Random Forest Regressor": {
        "model": "RandomForestRegressor",
        "module": "sklearn.ensemble",
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    },
    "Gradient Boosting Regressor": {
        "model": "GradientBoostingRegressor",
        "module": "sklearn.ensemble",
        "params": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7]
        }
    }
}


def get_model_instance(model_name, is_classification=True):
    """
    Get an instance of the specified model.
    """
    # Select model info based on task type
    models_dict = CLASSIFICATION_MODELS if is_classification else REGRESSION_MODELS
    model_info = models_dict.get(model_name)
    if model_info is None:
        raise ValueError(f"Model '{model_name}' not found")
    
    # Import model class
    module = __import__(model_info["module"], fromlist=[model_info["model"]])
    model_class = getattr(module, model_info["model"])
    
    return model_class()


def train_model(X, y, model_name, is_classification=True, test_size=0.2, random_state=42, cv=5, use_grid_search=True):
    """
    Train a model on the given data.
    Returns a dict: the results of training, including the trained model, metrics, and predictions.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # get model info and instance
    models_dict = CLASSIFICATION_MODELS if is_classification else REGRESSION_MODELS
    model_info = models_dict.get(model_name)
    
    if model_info is None:
        raise ValueError(f"Model '{model_name}' not found")
    
    model = get_model_instance(model_name, is_classification)
    
    # Train the model
    if use_grid_search and model_info["params"]:
        # use grid search for hyperparameter tuning
        grid_search = GridSearchCV(
            model, 
            model_info["params"], 
            cv=cv, 
            scoring='accuracy' if is_classification else 'neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        # get best model 
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
    else:
        # train model with default parameters if no grid search is used
        best_model = model
        best_model.fit(X_train, y_train)
        best_params = {}
    
    # make predictions
    y_pred = best_model.predict(X_test)
    
    # calculate metrics
    if is_classification:
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted') if len(np.unique(y)) > 2 else precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred, average='weighted') if len(np.unique(y)) > 2 else recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred, average='weighted') if len(np.unique(y)) > 2 else f1_score(y_test, y_pred)
        }
        
        # If model has predict_proba method, calculate probabilities
        if hasattr(best_model, 'predict_proba'):
            y_prob = best_model.predict_proba(X_test)
        else:
            y_prob = None
    else:
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        y_prob = None
    
    # Get feature importances if available
    if hasattr(best_model, 'feature_importances_'):
        feature_importances = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        feature_importances = best_model.coef_
        if feature_importances.ndim > 1:
            feature_importances = np.mean(np.abs(feature_importances), axis=0)
    else:
        feature_importances = None
    
    # Return results
    return {
        'model': best_model,
        'best_params': best_params,
        'metrics': metrics,
        'predictions': {
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob
        },
        'feature_importances': feature_importances
    }


def save_model(model, model_name, feature_names=None, target_encoder=None):
    """
    Save a trained model to disk.
    Returns the path to the saved model.
    """
    # Create models directory if it doesn't exist
    if not os.path.exists("models/saved"):
        os.makedirs("models/saved")
    
    # Create model info dictionary
    model_info = {
        'model': model,
        'feature_names': feature_names,
        'target_encoder': target_encoder
    }
    
    # Save model info
    model_path = f"models/saved/{model_name}.joblib"
    joblib.dump(model_info, model_path)
    
    return model_path


def load_model(model_path):
    """
    Load a saved model from disk.
    """
    # Load model info
    model_info = joblib.load(model_path)
    
    return model_info 