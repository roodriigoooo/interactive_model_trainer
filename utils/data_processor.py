"""
Data processing utilities for the ML Model Trainer application.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

class DataProcessor:
    """
    Class for preprocessing data before model training.
    """
    def __init__(self):
        self.numeric_imputer = None
        self.categorical_imputer = None
        self.numeric_scaler = None
        self.categorical_encoder = None
        self.label_encoder = None
        self.feature_names = []
        
    def preprocess_features(self, df, numeric_features, categorical_features, fit=True):
        """
        Preprocess features by imputing missing values, scaling numeric features,
        and encoding categorical features.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The dataframe containing the features.
        numeric_features : list
            List of numeric feature column names.
        categorical_features : list
            List of categorical feature column names.
        fit : bool, default=True
            Whether to fit the preprocessors on the data or just transform.
            
        Returns:
        --------
        X : numpy.ndarray
            The preprocessed feature matrix.
        """
        # Initialize preprocessors if fitting
        if fit:
            self.numeric_imputer = SimpleImputer(strategy='mean')
            self.categorical_imputer = SimpleImputer(strategy='most_frequent')
            self.numeric_scaler = StandardScaler()
            self.categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            
        # Preprocess numeric features
        numeric_data = df[numeric_features].copy() if numeric_features else pd.DataFrame()
        
        # Preprocess categorical features
        categorical_data = df[categorical_features].copy() if categorical_features else pd.DataFrame()
        
        # Process numeric features if present
        if not numeric_data.empty:
            if fit:
                numeric_data_imputed = self.numeric_imputer.fit_transform(numeric_data)
                numeric_data_scaled = self.numeric_scaler.fit_transform(numeric_data_imputed)
            else:
                numeric_data_imputed = self.numeric_imputer.transform(numeric_data)
                numeric_data_scaled = self.numeric_scaler.transform(numeric_data_imputed)
        else:
            numeric_data_scaled = np.array([]).reshape(df.shape[0], 0)
            
        # Process categorical features if present
        if not categorical_data.empty:
            if fit:
                categorical_data_imputed = self.categorical_imputer.fit_transform(categorical_data)
                categorical_data_encoded = self.categorical_encoder.fit_transform(categorical_data_imputed)
                # Store feature names for one-hot encoded features
                self.feature_names = (
                    numeric_features +
                    [f"{col}_{val}" for col, vals in zip(categorical_features, self.categorical_encoder.categories_)
                     for val in vals]
                )
            else:
                categorical_data_imputed = self.categorical_imputer.transform(categorical_data)
                categorical_data_encoded = self.categorical_encoder.transform(categorical_data_imputed)
        else:
            categorical_data_encoded = np.array([]).reshape(df.shape[0], 0)
            if fit:
                self.feature_names = numeric_features
                
        # Combine numeric and categorical features
        X = np.hstack([numeric_data_scaled, categorical_data_encoded])
        
        return X
    
    def preprocess_target(self, series, fit=True):
        """
        Preprocess the target variable.
        
        Parameters:
        -----------
        series : pandas.Series
            The target variable.
        fit : bool, default=True
            Whether to fit the encoder on the data or just transform.
            
        Returns:
        --------
        y : numpy.ndarray
            The preprocessed target variable.
        """
        # For classification problems (if target is categorical)
        if series.dtype == 'object' or series.dtype.name == 'category' or series.nunique() < 10:
            if fit:
                self.label_encoder = LabelEncoder()
                return self.label_encoder.fit_transform(series)
            else:
                return self.label_encoder.transform(series)
        # For regression problems
        else:
            return series.values
    
    def get_feature_names(self):
        """Return the feature names after preprocessing."""
        return self.feature_names 