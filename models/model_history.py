"""
Model history tracking module for the ML Model Trainer application.
Stores and retrieves training history for comparison.
"""
import datetime
import json
import os
import pandas as pd
import uuid


class ModelHistory:
    """
    class for tracking model training history, storage, retrieval, and comparison of trained models.
    """
    HISTORY_DIR = "models/history"
    
    def __init__(self):
        """Initialize model history."""
        # Create history directory if it doesn't exist
        if not os.path.exists(self.HISTORY_DIR):
            os.makedirs(self.HISTORY_DIR)
        
        # Initialize history df if it exists, otherwise create new
        self.history_file = f"{self.HISTORY_DIR}/history.json"
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = []
    
    def add_model(self, model_info, metrics, dataset_name, features, target, model_params=None):
        """
        add a trained model to history.
        """
        model_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        
        # Create history entry
        entry = {
            'model_id': model_id,
            'timestamp': timestamp,
            'model_name': model_info.get('model_name', 'Unknown'),
            'model_type': model_info.get('model_type', 'Unknown'),
            'dataset_name': dataset_name,
            'features': features,
            'target': target,
            'metrics': metrics,
            'parameters': model_params or {}
        }

        self.history.append(entry)
        self._save_history()
        
        return model_id
    
    def get_history(self, as_dataframe=True):
        """
        get model training history.
        """
        if not self.history:
            return pd.DataFrame() if as_dataframe else []
        
        if as_dataframe:
            # Create a normalized dataframe for easier comparison
            records = []
            for entry in self.history:
                record = {
                    'model_id': entry['model_id'],
                    'timestamp': entry['timestamp'],
                    'model_name': entry['model_name'],
                    'model_type': entry['model_type'],
                    'dataset_name': entry['dataset_name'],
                    'target': entry['target'],
                    'n_features': len(entry['features'])
                }
                
                # Add metrics as columns
                for metric, value in entry['metrics'].items():
                    record[f'metric_{metric}'] = value
                
                records.append(record)
            
            return pd.DataFrame(records)
        else:
            return self.history
    
    def get_model(self, model_id):
        """
        get a specific model entry by ID.
        """
        for entry in self.history:
            if entry['model_id'] == model_id:
                return entry
        return None
    
    def compare_models(self, model_ids):
        """
        compare multiple models based on their metrics.
        """
        models = [self.get_model(model_id) for model_id in model_ids]
        models = [m for m in models if m is not None]
        
        if not models:
            return pd.DataFrame()
        
        # Create comparison dataframe
        comparison = []
        for model in models:
            row = {
                'model_id': model['model_id'],
                'model_name': model['model_name'],
                'dataset': model['dataset_name'],
                'timestamp': model['timestamp']
            }
            
            # Add metrics
            for metric, value in model['metrics'].items():
                row[metric] = value
            
            comparison.append(row)
        
        return pd.DataFrame(comparison)
    
    def delete_model(self, model_id):
        """
        delete a model from history.
        """
        for i, entry in enumerate(self.history):
            if entry['model_id'] == model_id:
                self.history.pop(i)
                self._save_history()
                return True
        return False
    
    def _save_history(self):
        """Save history to disk."""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f) 