"""
Data processing utilities for the ML Model Trainer application.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from scipy import stats

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
        self.preprocessing_config = {
            'numeric_imputer_strategy': 'mean',
            'categorical_imputer_strategy': 'most_frequent',
            'numeric_scaler_type': 'standard',
            'handle_outliers': False
        }
        
    def analyze_data_characteristics(self, df, numeric_features, categorical_features):
        """
        Analyze dataset characteristics to suggest preprocessing strategies.
        Analyze numeric features, check for outliers using basic IQR method, and do basic analysis on missing values. 
        """
        analysis = {
            'has_outliers': False,
            'missing_value_patterns': {},
            'suggested_strategies': {}
        }
        
        if numeric_features:
            numeric_data = df[numeric_features]
        
            Q1 = numeric_data.quantile(0.25)
            Q3 = numeric_data.quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = ((numeric_data < (Q1 - 1.5 * IQR)) | 
                          (numeric_data > (Q3 + 1.5 * IQR)))
            outlier_features = outlier_mask.any()
            outlier_features = outlier_features[outlier_features].index.tolist()
            
            if outlier_features:
                analysis['has_outliers'] = True
                analysis['outlier_features'] = outlier_features
                analysis['suggested_strategies']['numeric_scaler'] = 'robust'
                analysis['suggested_strategies']['numeric_imputer'] = 'median'
    
            missing_stats = numeric_data.isnull().sum()
            if missing_stats.any():
                analysis['missing_value_patterns']['numeric'] = {
                    'features': missing_stats[missing_stats > 0].to_dict(),
                    'suggested_strategy': 'knn' if missing_stats.max() / len(df) < 0.2 else 'median'
                }
        
        # Analyze categorical features
        if categorical_features:
            categorical_data = df[categorical_features]
            missing_stats = categorical_data.isnull().sum()
            if missing_stats.any():
                analysis['missing_value_patterns']['categorical'] = {
                    'features': missing_stats[missing_stats > 0].to_dict(),
                    'suggested_strategy': 'most_frequent'
                }
        
        return analysis
    
    def configure_preprocessing(self, config):
        """
        Configure preprocessing strategies.
        """
        self.preprocessing_config.update(config)
    
    def preprocess_features(self, df, numeric_features, categorical_features, fit=True):
        """
        Preprocess features by imputing missing values, scaling numeric features,
        and encoding categorical features, depending on the variable type.
        """
        # Initialize preprocessors if fitting
        if fit:
            # Configure numeric imputer based on strategy
            if self.preprocessing_config['numeric_imputer_strategy'] == 'knn':
                self.numeric_imputer = KNNImputer(n_neighbors=5)
            else:
                self.numeric_imputer = SimpleImputer(
                    strategy=self.preprocessing_config['numeric_imputer_strategy']
                )
            
            # Configure categorical imputer
            self.categorical_imputer = SimpleImputer(
                strategy=self.preprocessing_config['categorical_imputer_strategy']
            )
            
            # Configure numeric scaler
            if self.preprocessing_config['numeric_scaler_type'] == 'robust':
                self.numeric_scaler = RobustScaler()
            else:
                self.numeric_scaler = StandardScaler()
            
            self.categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            
        # Preprocess numeric and categorical features
        numeric_data = df[numeric_features].copy() if numeric_features else pd.DataFrame()
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