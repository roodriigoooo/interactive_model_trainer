"""
Callback functions for the ML Model Trainer application.
These functions handle state management and user interactions.
"""
import streamlit as st
from typing import Optional, Dict, Any, List
import pandas as pd
from .app_utils import save_session_state, load_dataset
from .data_processor import DataProcessor


def initialize_session_state():
    """Initialize all session state variables with default values."""
    defaults = {
        "dataset": None,
        "dataset_name": None,
        "features": {"numeric": [], "categorical": []},
        "target": None,
        "uploaded_dataset": None,
        "active_tab": "Data Selection",
        "data_processor": None,
        "trained_model": None,
        "model_results": None,
        "is_classification": None,
        "training_tab_state": {
            "selected_model": None,
            "test_size": 0.2,
            "random_state": 42,
            "use_grid_search": True,
            "cv_folds": 5
        },
        "viz_settings": {
            "library": "altair",
            "theme": "default"
        },
        "latest_training_results": None,
        "selected_dataset": None,
        "selected_target": None,
        "selected_numeric": [],
        "selected_categorical": []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def on_dataset_change():
    """Handle dataset selection changes."""
    if st.session_state.selected_dataset != st.session_state.dataset_name:
        load_dataset(st.session_state.selected_dataset)
        st.session_state.active_tab = "Data Selection"


def on_target_change(arg=None):
    """
    handle target variable selection changes. the unused arg is just so streamlit does not raise a num of parameters passed error. 
    """
    # Get the target value from either widget
    new_target = st.session_state.get("data_selection_target") or st.session_state.get("feature_selection_target")
    
    if new_target != st.session_state.target:
        st.session_state.target = new_target
        # Remove target from features if present
        if st.session_state.target in st.session_state.features["numeric"]:
            st.session_state.features["numeric"].remove(st.session_state.target)
        if st.session_state.target in st.session_state.features["categorical"]:
            st.session_state.features["categorical"].remove(st.session_state.target)
        
        # Keep both target selection widgets in sync
        if "data_selection_target" in st.session_state:
            st.session_state.data_selection_target = new_target
        if "feature_selection_target" in st.session_state:
            st.session_state.feature_selection_target = new_target
            
        save_session_state()


def on_feature_selection(feature_type) :
    """
    handle feature selection changes.
    """
    if feature_type == "numeric":
        st.session_state.features["numeric"] = st.session_state.selected_numeric
    else:
        st.session_state.features["categorical"] = st.session_state.selected_categorical
    save_session_state()


def on_preprocessing_config_change(config):
    """
    handle preprocessing configuration changes.
    """
    if st.session_state.data_processor is None:
        st.session_state.data_processor = DataProcessor()
    st.session_state.data_processor.configure_preprocessing(config)


def on_viz_settings_change():
    """Handle visualization settings changes."""
    st.session_state.viz_settings["library"] = "altair" if st.session_state.viz_lib_radio == "Altair (Interactive)" else "matplotlib"


def on_model_train(model_name=None, state_getter=None): #state_getter is to avoid the issue where switching datasets resulted in unwanted amounts of parameters passed to the function
    """
    Handle model training initiation.
    
    Parameters
    ----------
    model_name : str, optional
        Name of the selected model
    state_getter : callable, optional
        Function to get the current state
    """
    if "training_tab_state" not in st.session_state:
        return
    
    # Update training parameters based on widget states
    st.session_state.training_tab_state.update({
        "test_size": st.session_state.get("test_size", 0.2),
        "random_state": st.session_state.get("random_state", 42),
        "use_grid_search": st.session_state.get("use_grid_search", True),
        "cv_folds": st.session_state.get("cv_folds", 5)
    })
    
    if model_name is not None:
        st.session_state.training_tab_state["selected_model"] = model_name
    
    if state_getter is not None:
        try:
            new_state = state_getter()
            st.session_state.training_tab_state.update(new_state)
        except Exception:
            pass 