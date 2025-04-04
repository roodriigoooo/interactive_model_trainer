"""
Unit tests for ModelHistory class.
"""
import unittest
import pandas as pd
import numpy as np
import os
import shutil
from datetime import datetime
from models.model_history import ModelHistory


class TestModelHistory(unittest.TestCase):
    """Test cases for ModelHistory class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create test directory
        self.test_dir = 'models/history_test'
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Override history directory
        ModelHistory.HISTORY_DIR = self.test_dir
        
        # Create history instance
        self.history = ModelHistory()
        
        # Sample model info
        self.model_info = {
            'model_name': 'Test Model',
            'model_type': 'Logistic Regression'
        }
        
        # Sample metrics
        self.metrics = {
            'accuracy': 0.85,
            'precision': 0.83,
            'recall': 0.87,
            'f1': 0.85
        }
        
        # Sample dataset info
        self.dataset_name = 'test_dataset'
        self.features = ['feature_1', 'feature_2', 'feature_3']
        self.target = 'target'
        
        # Sample parameters
        self.model_params = {
            'C': 1.0,
            'penalty': 'l2',
            'solver': 'lbfgs'
        }
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove test directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_init(self):
        """Test initialization."""
        self.assertEqual(self.history.HISTORY_DIR, self.test_dir)
        self.assertEqual(self.history.history, [])
    
    def test_add_model(self):
        """Test adding a model to history."""
        # Add a model
        model_id = self.history.add_model(
            self.model_info,
            self.metrics,
            self.dataset_name,
            self.features,
            self.target,
            self.model_params
        )
        
        # Check that model was added
        self.assertEqual(len(self.history.history), 1)
        self.assertEqual(self.history.history[0]['model_id'], model_id)
        self.assertEqual(self.history.history[0]['model_name'], 'Test Model')
        self.assertEqual(self.history.history[0]['model_type'], 'Logistic Regression')
        self.assertEqual(self.history.history[0]['dataset_name'], 'test_dataset')
        self.assertEqual(self.history.history[0]['features'], ['feature_1', 'feature_2', 'feature_3'])
        self.assertEqual(self.history.history[0]['target'], 'target')
        self.assertEqual(self.history.history[0]['metrics'], self.metrics)
        self.assertEqual(self.history.history[0]['parameters'], self.model_params)
        
        # Check that timestamp was added
        self.assertIn('timestamp', self.history.history[0])
    
    def test_get_history(self):
        """Test getting history."""
        # Add some models
        model_id1 = self.history.add_model(
            self.model_info,
            self.metrics,
            self.dataset_name,
            self.features,
            self.target,
            self.model_params
        )
        
        model_id2 = self.history.add_model(
            {'model_name': 'Another Model', 'model_type': 'Random Forest'},
            {'accuracy': 0.90, 'precision': 0.89, 'recall': 0.91, 'f1': 0.90},
            'another_dataset',
            ['feature_1', 'feature_2'],
            'target',
            {'n_estimators': 100}
        )
        
        # Get history as list
        history_list = self.history.get_history(as_dataframe=False)
        self.assertEqual(len(history_list), 2)
        self.assertEqual(history_list[0]['model_id'], model_id1)
        self.assertEqual(history_list[1]['model_id'], model_id2)
        
        # Get history as dataframe
        history_df = self.history.get_history(as_dataframe=True)
        self.assertEqual(len(history_df), 2)
        self.assertEqual(history_df['model_id'].tolist(), [model_id1, model_id2])
        self.assertEqual(history_df['model_name'].tolist(), ['Test Model', 'Another Model'])
        self.assertEqual(history_df['dataset_name'].tolist(), ['test_dataset', 'another_dataset'])
        self.assertEqual(history_df['n_features'].tolist(), [3, 2])
    
    def test_get_model(self):
        """Test getting a specific model."""
        # Add a model
        model_id = self.history.add_model(
            self.model_info,
            self.metrics,
            self.dataset_name,
            self.features,
            self.target,
            self.model_params
        )
        
        # Get the model
        model = self.history.get_model(model_id)
        self.assertIsNotNone(model)
        self.assertEqual(model['model_id'], model_id)
        self.assertEqual(model['model_name'], 'Test Model')
        
        # Try to get a non-existent model
        model = self.history.get_model('non-existent-id')
        self.assertIsNone(model)
    
    def test_compare_models(self):
        """Test comparing models."""
        # Add some models
        model_id1 = self.history.add_model(
            self.model_info,
            self.metrics,
            self.dataset_name,
            self.features,
            self.target,
            self.model_params
        )
        
        model_id2 = self.history.add_model(
            {'model_name': 'Another Model', 'model_type': 'Random Forest'},
            {'accuracy': 0.90, 'precision': 0.89, 'recall': 0.91, 'f1': 0.90},
            'another_dataset',
            ['feature_1', 'feature_2'],
            'target',
            {'n_estimators': 100}
        )
        
        # Compare models
        comparison = self.history.compare_models([model_id1, model_id2])
        self.assertEqual(len(comparison), 2)
        self.assertEqual(comparison['model_id'].tolist(), [model_id1, model_id2])
        self.assertEqual(comparison['model_name'].tolist(), ['Test Model', 'Another Model'])
        
        # Check metrics
        self.assertEqual(comparison['accuracy'].tolist(), [0.85, 0.90])
        self.assertEqual(comparison['precision'].tolist(), [0.83, 0.89])
        self.assertEqual(comparison['recall'].tolist(), [0.87, 0.91])
        self.assertEqual(comparison['f1'].tolist(), [0.85, 0.90])
        
        # Compare non-existent models
        comparison = self.history.compare_models(['non-existent-id'])
        self.assertTrue(comparison.empty)
    
    def test_delete_model(self):
        """Test deleting a model."""
        # Add a model
        model_id = self.history.add_model(
            self.model_info,
            self.metrics,
            self.dataset_name,
            self.features,
            self.target,
            self.model_params
        )
        
        # Delete the model
        result = self.history.delete_model(model_id)
        self.assertTrue(result)
        self.assertEqual(len(self.history.history), 0)
        
        # Try to delete a non-existent model
        result = self.history.delete_model('non-existent-id')
        self.assertFalse(result)
    
    def test_save_and_load(self):
        """Test saving and loading history."""
        # Add a model
        model_id = self.history.add_model(
            self.model_info,
            self.metrics,
            self.dataset_name,
            self.features,
            self.target,
            self.model_params
        )
        
        # Create a new history instance (should load from file)
        new_history = ModelHistory()
        self.assertEqual(len(new_history.history), 1)
        self.assertEqual(new_history.history[0]['model_id'], model_id)


if __name__ == '__main__':
    unittest.main() 