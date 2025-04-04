"""
Unit tests for model utilities.
"""
import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from models.model_utils import (
    get_model_instance,
    get_param_grid,
    train_model,
    save_model,
    load_model
)


class TestModelUtils(unittest.TestCase):
    """Test cases for model utilities."""
    
    def setUp(self):
        """Set up test data."""
        # Classification dataset
        X_cls, y_cls = make_classification(
            n_samples=100,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=42
        )
        self.X_cls = pd.DataFrame(X_cls, columns=[f'feature_{i}' for i in range(X_cls.shape[1])])
        self.y_cls = pd.Series(y_cls, name='target')
        
        # Regression dataset
        X_reg, y_reg = make_regression(
            n_samples=100,
            n_features=5,
            n_informative=3,
            noise=0.1,
            random_state=42
        )
        self.X_reg = pd.DataFrame(X_reg, columns=[f'feature_{i}' for i in range(X_reg.shape[1])])
        self.y_reg = pd.Series(y_reg, name='target')
    
    def test_get_model_instance_classification(self):
        """Test getting classification model instances."""
        # Test Logistic Regression
        model = get_model_instance('Logistic Regression', is_classification=True)
        self.assertIsNotNone(model)
        self.assertEqual(model.__class__.__name__, 'LogisticRegression')
        
        # Test Random Forest
        model = get_model_instance('Random Forest', is_classification=True)
        self.assertIsNotNone(model)
        self.assertEqual(model.__class__.__name__, 'RandomForestClassifier')
        
        # Test model not found
        with self.assertRaises(ValueError):
            get_model_instance('Invalid Model', is_classification=True)
    
    def test_get_model_instance_regression(self):
        """Test getting regression model instances."""
        # Test Linear Regression
        model = get_model_instance('Linear Regression', is_classification=False)
        self.assertIsNotNone(model)
        self.assertEqual(model.__class__.__name__, 'LinearRegression')
        
        # Test Random Forest Regressor
        model = get_model_instance('Random Forest Regressor', is_classification=False)
        self.assertIsNotNone(model)
        self.assertEqual(model.__class__.__name__, 'RandomForestRegressor')
        
        # Test model not found
        with self.assertRaises(ValueError):
            get_model_instance('Invalid Model', is_classification=False)
    
    def test_get_param_grid(self):
        """Test getting parameter grids."""
        # Test default params
        params = get_param_grid('Logistic Regression', is_classification=True)
        self.assertIsNotNone(params)
        self.assertIn('C', params)
        
        # Test custom params
        custom_params = {'C': [0.1, 1.0], 'penalty': ['l2']}
        params = get_param_grid('Logistic Regression', is_classification=True, custom_params=custom_params)
        self.assertEqual(params, custom_params)
        
        # Test model not found
        with self.assertRaises(ValueError):
            get_param_grid('Invalid Model', is_classification=True)
    
    def test_train_model_classification(self):
        """Test training classification models."""
        # Basic training without grid search
        results = train_model(
            self.X_cls,
            self.y_cls,
            'Logistic Regression',
            is_classification=True,
            use_grid_search=False
        )
        
        self.assertIsNotNone(results['model'])
        self.assertIn('accuracy', results['metrics'])
        self.assertIn('precision', results['metrics'])
        self.assertIn('recall', results['metrics'])
        self.assertIn('f1', results['metrics'])
        
        # With grid search
        results = train_model(
            self.X_cls,
            self.y_cls,
            'Logistic Regression',
            is_classification=True,
            use_grid_search=True
        )
        
        self.assertIsNotNone(results['model'])
        self.assertIsNotNone(results['best_params'])
        self.assertIsNotNone(results['cv_results'])
        
        # With randomized search
        results = train_model(
            self.X_cls,
            self.y_cls,
            'Logistic Regression',
            is_classification=True,
            use_grid_search=False,
            use_randomized_search=True,
            n_iter=5
        )
        
        self.assertIsNotNone(results['model'])
        self.assertIsNotNone(results['best_params'])
        self.assertIsNotNone(results['cv_results'])
    
    def test_train_model_regression(self):
        """Test training regression models."""
        # Basic training without grid search
        results = train_model(
            self.X_reg,
            self.y_reg,
            'Linear Regression',
            is_classification=False,
            use_grid_search=False
        )
        
        self.assertIsNotNone(results['model'])
        self.assertIn('mse', results['metrics'])
        self.assertIn('rmse', results['metrics'])
        self.assertIn('mae', results['metrics'])
        self.assertIn('r2', results['metrics'])
        
        # With grid search
        results = train_model(
            self.X_reg,
            self.y_reg,
            'Ridge Regression',
            is_classification=False,
            use_grid_search=True
        )
        
        self.assertIsNotNone(results['model'])
        self.assertIsNotNone(results['best_params'])
        self.assertIsNotNone(results['cv_results'])
    
    def test_save_and_load_model(self):
        """Test saving and loading models."""
        # Train a model
        results = train_model(
            self.X_cls,
            self.y_cls,
            'Logistic Regression',
            is_classification=True,
            use_grid_search=False
        )
        
        model = results['model']
        feature_names = [f'feature_{i}' for i in range(self.X_cls.shape[1])]
        
        # Save the model
        model_path = save_model(
            model,
            'test_model',
            feature_names=feature_names
        )
        
        self.assertIsNotNone(model_path)
        
        # Load the model
        loaded_model_info = load_model(model_path)
        
        self.assertIsNotNone(loaded_model_info['model'])
        self.assertEqual(loaded_model_info['feature_names'], feature_names)


if __name__ == '__main__':
    unittest.main() 