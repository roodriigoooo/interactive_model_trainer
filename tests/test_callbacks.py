"""
Tests for callback functions in the ML Model Trainer application.
"""
import pytest
import streamlit as st
import pandas as pd
import numpy as np
from ..utils.callbacks import (
    initialize_session_state,
    on_dataset_change,
    on_target_change,
    on_feature_selection,
    on_preprocessing_config_change,
    on_viz_settings_change,
    on_model_train
)
from ..utils.data_processor import DataProcessor


@pytest.fixture
def mock_session_state(monkeypatch):
    """Create a mock session state for testing."""
    class MockSessionState(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._mock_callbacks = {}

        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            if key in self._mock_callbacks:
                self._mock_callbacks[key](value)

    mock_state = MockSessionState()
    monkeypatch.setattr(st, "session_state", mock_state)
    return mock_state


def test_initialize_session_state(mock_session_state):
    """Test session state initialization with default values."""
    initialize_session_state()
    
    # Check that all expected keys are present with correct default values
    assert "dataset" in mock_session_state
    assert "dataset_name" in mock_session_state
    assert "features" in mock_session_state
    assert "target" in mock_session_state
    assert "uploaded_dataset" in mock_session_state
    assert "active_tab" in mock_session_state
    assert "data_processor" in mock_session_state
    assert "trained_model" in mock_session_state
    assert "model_results" in mock_session_state
    assert "is_classification" in mock_session_state
    assert "training_tab_state" in mock_session_state
    assert "viz_settings" in mock_session_state
    
    # Check specific default values
    assert mock_session_state["features"] == {"numeric": [], "categorical": []}
    assert mock_session_state["active_tab"] == "Data Selection"
    assert mock_session_state["viz_settings"]["library"] == "altair"


def test_on_target_change(mock_session_state):
    """Test target variable change handling."""
    # Setup initial state
    mock_session_state.update({
        "target": None,
        "features": {"numeric": ["feature1", "target_var"], "categorical": []},
        "data_selection_target": "target_var",
        "feature_selection_target": None
    })
    
    # Test target change
    on_target_change()
    
    # Check that target was updated
    assert mock_session_state["target"] == "target_var"
    # Check that target was removed from features
    assert "target_var" not in mock_session_state["features"]["numeric"]
    # Check that both widgets are in sync
    assert mock_session_state["data_selection_target"] == "target_var"


def test_on_feature_selection(mock_session_state):
    """Test feature selection handling."""
    # Setup initial state
    mock_session_state.update({
        "features": {"numeric": [], "categorical": []},
        "selected_numeric": ["feature1", "feature2"],
        "selected_categorical": ["category1"]
    })
    
    # Test numeric feature selection
    on_feature_selection("numeric")
    assert mock_session_state["features"]["numeric"] == ["feature1", "feature2"]
    
    # Test categorical feature selection
    on_feature_selection("categorical")
    assert mock_session_state["features"]["categorical"] == ["category1"]


def test_on_viz_settings_change(mock_session_state):
    """Test visualization settings change handling."""
    # Test Altair selection
    mock_session_state["viz_lib_radio"] = "Altair (Interactive)"
    on_viz_settings_change()
    assert mock_session_state["viz_settings"]["library"] == "altair"
    
    # Test Matplotlib selection
    mock_session_state["viz_lib_radio"] = "Matplotlib/Seaborn"
    on_viz_settings_change()
    assert mock_session_state["viz_settings"]["library"] == "matplotlib"


def test_on_model_train(mock_session_state):
    """Test model training callback."""
    # Setup initial state
    mock_session_state.update({
        "training_tab_state": {
            "selected_model": None,
            "test_size": 0.2,
            "random_state": 42,
            "use_grid_search": True,
            "cv_folds": 5
        },
        "test_size": 0.3,  # New value from widget
        "random_state": 123  # New value from widget
    })
    
    # Test with model name
    on_model_train(model_name="RandomForest")
    assert mock_session_state["training_tab_state"]["selected_model"] == "RandomForest"
    assert mock_session_state["training_tab_state"]["test_size"] == 0.3
    assert mock_session_state["training_tab_state"]["random_state"] == 123
    
    # Test with state getter
    def mock_state_getter():
        return {"cv_folds": 10}
    
    on_model_train(state_getter=mock_state_getter)
    assert mock_session_state["training_tab_state"]["cv_folds"] == 10


def test_on_preprocessing_config_change(mock_session_state):
    """Test preprocessing configuration change handling."""
    config = {
        "numeric_imputer": "mean",
        "categorical_imputer": "mode",
        "scaler": "standard"
    }
    
    on_preprocessing_config_change(config)
    assert isinstance(mock_session_state["data_processor"], DataProcessor)
    # Verify that the config was applied to the data processor
    processor = mock_session_state["data_processor"]
    assert processor.preprocessing_config == config


@pytest.mark.skip(reason="Requires dataset loading functionality")
def test_on_dataset_change(mock_session_state):
    """Test dataset change handling."""
    # Setup initial state
    mock_session_state.update({
        "selected_dataset": "new_dataset",
        "dataset_name": "old_dataset"
    })
    
    on_dataset_change()
    
    assert mock_session_state["active_tab"] == "Data Selection"
    # Additional assertions would depend on dataset loading implementation 