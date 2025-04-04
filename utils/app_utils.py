"""Helper functions and nice-to-have functions for the app."""
import pandas as pd
import seaborn as sns
import streamlit as st
import os 
import json 
import numpy as np

def get_available_seaborn_datasets():
    """Return a list of available datasets in seaborn."""
    # Get all dataset names from seaborn
    dataset_names = sns.get_dataset_names()
    pretty_names = {name: name.replace('_', ' ').title() for name in dataset_names}
    return {pretty_names[name]: name for name in dataset_names}

def identify_variable_types(df):
    """Identify numeric and categorical variables in a dataframe."""
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # Some numeric columns with few unique values might be better treated as categorical
    for col in numeric_cols.copy():
        if df[col].nunique() < 10:  # Threshold for considering a numeric column as categorical
            categorical_cols.append(col)
            numeric_cols.remove(col)
    
    return numeric_cols, categorical_cols

def save_session_state():
    """Save current session state to a file for persistence."""
    state_to_save = {
        "dataset_name": st.session_state.dataset_name,
        "features": st.session_state.features,
        "target": st.session_state.target,
        "active_tab": st.session_state.active_tab
    }
    
    if not os.path.exists(".streamlit/data"):
        os.makedirs(".streamlit/data")
    
    with open(".streamlit/data/session_state.json", "w") as f:
        json.dump(state_to_save, f)

def load_session_state():
    """Load session state from file if it exists."""
    try:
        if os.path.exists(".streamlit/data/session_state.json"):
            with open(".streamlit/data/session_state.json", "r") as f:
                saved_state = json.load(f)
                
            # Only restore if we don't already have a dataset loaded
            if st.session_state.dataset is None:
                for key, value in saved_state.items():
                    if key in st.session_state:
                        st.session_state[key] = value
            
                if st.session_state.dataset_name:
                    try:
                        load_dataset(st.session_state.dataset_name)
                    except:
                        pass  # ff loading fails, we'll just start fresh
    except:
        # if loading fails for any reason start fresh again
        pass

def load_dataset(dataset_name):
    """Load the selected dataset."""
    # Reset model-related session state
    st.session_state.target = None
    st.session_state.trained_model = None
    st.session_state.model_results = None
    st.session_state.is_classification = None
    st.session_state.data_processor = None
    
    if dataset_name == "Upload Custom Dataset":
        if st.session_state.uploaded_dataset is not None:
            st.session_state.dataset = st.session_state.uploaded_dataset
            st.session_state.dataset_name = "Custom Dataset"
    else:
        # Get the original dataset name from the pretty name
        name_to_func = get_available_seaborn_datasets()
        dataset_name_original = name_to_func.get(dataset_name)
        
        if dataset_name_original:
            st.session_state.dataset = sns.load_dataset(dataset_name_original)
            st.session_state.dataset_name = dataset_name
    
    # If a dataset was successfully loaded, identify variable types
    if st.session_state.dataset is not None:
        numeric_cols, categorical_cols = identify_variable_types(st.session_state.dataset)
        st.session_state.features = {"numeric": numeric_cols, "categorical": categorical_cols}
        st.session_state.original_features = {
            "numeric": numeric_cols.copy(),
            "categorical": categorical_cols.copy()
        }
        # Save the current state!!!
        save_session_state()

def check_high_correlations(df, numeric_features, threshold=0.95):
    """
    Check for highly correlated features in the dataset, to give warnings to the user. 
    """
    if len(numeric_features) < 2:
        return []
        
    corr_matrix = df[numeric_features].corr().abs()
    high_corr_pairs = []
    
    # Get upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find feature pairs with correlation above threshold
    for col in upper.columns:
        high_corr = upper[col][upper[col] > threshold]
        for idx, corr_value in high_corr.items():
            high_corr_pairs.append((col, idx, corr_value))
    
    return high_corr_pairs