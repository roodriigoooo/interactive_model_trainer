"""
Unit tests for the DataProcessor class.
"""
import unittest
import pandas as pd
import numpy as np
from utils.data_processor import DataProcessor


class TestDataProcessor(unittest.TestCase):
    """Test cases for DataProcessor class."""
    
    def setUp(self):
        """Set up test data."""
        self.data = pd.DataFrame({
            'numeric1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'numeric2': [5.0, 4.0, 3.0, 2.0, 1.0],
            'categorical1': ['A', 'B', 'C', 'A', 'B'],
            'categorical2': ['X', 'Y', 'X', 'Y', 'X'],
            'target_num': [10, 20, 30, 40, 50],
            'target_cat': ['Class1', 'Class2', 'Class1', 'Class2', 'Class1']
        })
        
        self.numeric_features = ['numeric1', 'numeric2']
        self.categorical_features = ['categorical1', 'categorical2']
        
        self.processor = DataProcessor()
    
    def test_init(self):
        """Test initialization."""
        self.assertIsNone(self.processor.numeric_scaler)
        self.assertIsNone(self.processor.categorical_encoder)
        self.assertIsNone(self.processor.label_encoder)
        self.assertIsNone(self.processor.feature_names)
    
    def test_preprocess_numeric_features(self):
        """Test preprocessing of numeric features."""
        processed = self.processor._preprocess_numeric_features(
            self.data[self.numeric_features],
            fit=True
        )
        
        # Check that the processor fitted the scaler
        self.assertIsNotNone(self.processor.numeric_scaler)
        
        # Check that the data was scaled
        self.assertTrue(np.allclose(processed.mean(), [0, 0], atol=1e-10))
        self.assertTrue(np.allclose(processed.std(), [1, 1], atol=1e-10))
        
        # Test transform without fitting
        processed2 = self.processor._preprocess_numeric_features(
            self.data[self.numeric_features],
            fit=False
        )
        
        # Results should be the same
        self.assertTrue(np.allclose(processed, processed2, atol=1e-10))
    
    def test_preprocess_categorical_features(self):
        """Test preprocessing of categorical features."""
        processed = self.processor._preprocess_categorical_features(
            self.data[self.categorical_features],
            fit=True
        )
        
        # Check that the processor fitted the encoder
        self.assertIsNotNone(self.processor.categorical_encoder)
        
        # Check that the data was one-hot encoded
        self.assertEqual(processed.shape[1], 4)  # 2 + 2 categories
        
        # Test transform without fitting
        processed2 = self.processor._preprocess_categorical_features(
            self.data[self.categorical_features],
            fit=False
        )
        
        # Results should be the same
        self.assertTrue(np.allclose(processed, processed2, atol=1e-10))
    
    def test_preprocess_features(self):
        """Test preprocessing of all features together."""
        processed = self.processor.preprocess_features(
            self.data,
            self.numeric_features,
            self.categorical_features
        )
        
        # Check that the processor fitted the scaler and encoder
        self.assertIsNotNone(self.processor.numeric_scaler)
        self.assertIsNotNone(self.processor.categorical_encoder)
        
        # Check that the feature names were set
        self.assertIsNotNone(self.processor.feature_names)
        
        # Check shape
        self.assertEqual(processed.shape[1], 6)  # 2 numeric + 4 one-hot
        
        # Check that feature names were set correctly
        self.assertEqual(len(self.processor.feature_names), 6)
    
    def test_preprocess_target_numeric(self):
        """Test preprocessing of numeric target."""
        target = self.data['target_num']
        processed = self.processor.preprocess_target(target)
        
        # Check that no encoder was used for numeric target
        self.assertIsNone(self.processor.label_encoder)
        
        # Check that the values were not changed
        self.assertTrue(np.array_equal(processed, target.values))
    
    def test_preprocess_target_categorical(self):
        """Test preprocessing of categorical target."""
        target = self.data['target_cat']
        processed = self.processor.preprocess_target(target)
        
        # Check that the processor fitted the encoder
        self.assertIsNotNone(self.processor.label_encoder)
        
        # Check that the target was encoded
        self.assertEqual(len(np.unique(processed)), 2)
        
        # Test transform without fitting
        processed2 = self.processor.preprocess_target(target, fit=False)
        
        # Results should be the same
        self.assertTrue(np.array_equal(processed, processed2))
    
    def test_get_feature_names(self):
        """Test getting feature names."""
        # Process features to set feature names
        self.processor.preprocess_features(
            self.data,
            self.numeric_features,
            self.categorical_features
        )
        
        # Get feature names
        feature_names = self.processor.get_feature_names()
        
        # Check that feature names were returned
        self.assertIsNotNone(feature_names)
        self.assertEqual(len(feature_names), 6)


if __name__ == '__main__':
    unittest.main() 