"""
Model utilities for the ML Model Trainer application.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import os
import pickle
from datetime import datetime

# Dictionary of available models
CLASSIFICATION_MODELS = {
    "Logistic Regression": {
        "model": "LogisticRegression",
        "module": "sklearn.linear_model",
        "params": {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "penalty": ["l2"],
            "solver": ["lbfgs", "liblinear"],
            "class_weight": [None, "balanced"]
        }
    },
    "Random Forest": {
        "model": "RandomForestClassifier",
        "module": "sklearn.ensemble",
        "params": {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False],
            "class_weight": [None, "balanced", "balanced_subsample"]
        }
    },
    "Gradient Boosting": {
        "model": "GradientBoostingClassifier",
        "module": "sklearn.ensemble",
        "params": {
            "n_estimators": [50, 100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7, 9],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    },
    "Support Vector Machine": {
        "model": "SVC",
        "module": "sklearn.svm",
        "params": {
            "C": [0.1, 1.0, 10.0, 100.0],
            "kernel": ["linear", "rbf", "poly"],
            "gamma": ["scale", "auto", 0.1, 0.01],
            "probability": [True],
            "class_weight": [None, "balanced"]
        }
    },
    "K-Nearest Neighbors": {
        "model": "KNeighborsClassifier",
        "module": "sklearn.neighbors",
        "params": {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "leaf_size": [20, 30, 40],
            "p": [1, 2]  # Manhattan or Euclidean
        }
    },
    "Decision Tree": {
        "model": "DecisionTreeClassifier",
        "module": "sklearn.tree",
        "params": {
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "criterion": ["gini", "entropy"],
            "class_weight": [None, "balanced"]
        }
    }
}

REGRESSION_MODELS = {
    "Linear Regression": {
        "model": "LinearRegression",
        "module": "sklearn.linear_model",
        "params": {
            "fit_intercept": [True, False],
            "normalize": [False, True],
            "copy_X": [True],
            "n_jobs": [None, -1]
        }
    },
    "Ridge Regression": {
        "model": "Ridge",
        "module": "sklearn.linear_model",
        "params": {
            "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
            "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
            "fit_intercept": [True, False],
            "normalize": [False, True]
        }
    },
    "Lasso Regression": {
        "model": "Lasso",
        "module": "sklearn.linear_model",
        "params": {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
            "fit_intercept": [True, False],
            "normalize": [False, True],
            "selection": ["cyclic", "random"]
        }
    },
    "ElasticNet": {
        "model": "ElasticNet",
        "module": "sklearn.linear_model",
        "params": {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
            "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
            "fit_intercept": [True, False],
            "normalize": [False, True],
            "selection": ["cyclic", "random"]
        }
    },
    "Random Forest Regressor": {
        "model": "RandomForestRegressor",
        "module": "sklearn.ensemble",
        "params": {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False]
        }
    },
    "Gradient Boosting Regressor": {
        "model": "GradientBoostingRegressor",
        "module": "sklearn.ensemble",
        "params": {
            "n_estimators": [50, 100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7, 9],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "subsample": [0.8, 0.9, 1.0]
        }
    },
    "SVR": {
        "model": "SVR",
        "module": "sklearn.svm",
        "params": {
            "C": [0.1, 1.0, 10.0, 100.0],
            "kernel": ["linear", "rbf", "poly"],
            "gamma": ["scale", "auto", 0.1, 0.01],
            "epsilon": [0.1, 0.2, 0.5]
        }
    }
}


def get_model_instance(model_name, is_classification=True):
    """
    get an instance of the specified model.
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


def get_param_grid(model_name, is_classification=True, custom_params=None):
    """
    get the parameter grid for a model.
    """
    # If custom params are provided, use those
    if custom_params:
        return custom_params
        
    # Otherwise use default params
    models_dict = CLASSIFICATION_MODELS if is_classification else REGRESSION_MODELS
    model_info = models_dict.get(model_name)
    if model_info is None:
        raise ValueError(f"Model '{model_name}' not found")
    
    return model_info["params"]


def train_model(X, y, model_name, is_classification=True, test_size=0.2, random_state=42, 
                cv=5, use_grid_search=True, use_randomized_search=False, n_iter=10,
                custom_params=None, scoring=None):
    """
    train a model on the given data with flexible hyperparameter tuning.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Get model info and instance
    models_dict = CLASSIFICATION_MODELS if is_classification else REGRESSION_MODELS
    model_info = models_dict.get(model_name)
    
    if model_info is None:
        raise ValueError(f"Model '{model_name}' not found")
    
    model = get_model_instance(model_name, is_classification)
    param_grid = get_param_grid(model_name, is_classification, custom_params)
    
    # Set default scoring if not provided
    if scoring is None:
        scoring = 'accuracy' if is_classification else 'neg_mean_squared_error'
    
    # Train the model
    if (use_grid_search or use_randomized_search) and param_grid:
        # Choose search method
        if use_randomized_search:
            search = RandomizedSearchCV(
                model, 
                param_grid, 
                n_iter=n_iter,
                cv=cv, 
                scoring=scoring,
                n_jobs=-1,
                random_state=random_state
            )
        else:
            search = GridSearchCV(
                model, 
                param_grid, 
                cv=cv, 
                scoring=scoring,
                n_jobs=-1
            )
        
        # Perform search
        search.fit(X_train, y_train)
        
        # Get best model 
        best_model = search.best_estimator_
        best_params = search.best_params_
        
        # Get all results for parameter analysis
        cv_results = pd.DataFrame(search.cv_results_)
    else:
        # Train model with default parameters
        best_model = model
        best_model.fit(X_train, y_train)
        best_params = {}
        cv_results = None
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    
    # Calculate metrics
    if is_classification:
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted') if len(np.unique(y)) > 2 else precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred, average='weighted') if len(np.unique(y)) > 2 else recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred, average='weighted') if len(np.unique(y)) > 2 else f1_score(y_test, y_pred)
        }
        
        # Calculate probabilities if available
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
        'feature_importances': feature_importances,
        'cv_results': cv_results
    }


def save_model(model, model_name, feature_names=None, target_encoder=None):
    """
    save a trained model to disk.
    """
    try:
        # Get absolute path for models directory
        current_dir = os.getcwd()
        models_dir = os.path.join(current_dir, "models")
        saved_dir = os.path.join(models_dir, "saved")
        
        # Create directories if they don't exist
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(saved_dir, exist_ok=True)
        
        # Create model info dictionary
        model_info = {
            'model': model,
            'feature_names': feature_names,
            'target_encoder': target_encoder,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Create full path for the model file
        model_path = os.path.join(saved_dir, f"{model_name}.pkl")
        
        # Save model info using pickle
        with open(model_path, 'wb') as f:
            pickle.dump(model_info, f)
        
        print(f"Model saved successfully to: {model_path}")  # Debug print
        return model_path
        
    except Exception as e:
        print(f"Error while saving model: {str(e)}")  # Debug print
        raise Exception(f"Failed to save model: {str(e)}")


def load_model(model_path):
    """
    it would be nice to be able to load a saved model from disk.
    """
    try:
        with open(model_path, 'rb') as f:
            model_info = pickle.load(f)
        return model_info
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}") 