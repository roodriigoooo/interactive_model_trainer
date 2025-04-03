import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
from io import StringIO
import json

# Import utility functions
from utils.visualizer import (
    plot_correlation_matrix,
    plot_categorical_distributions,
    plot_numeric_distributions,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importances
)

# Import model utilities
from models.model_utils import (
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
    train_model,
    save_model
)

# Import data processor
from utils.data_processor import DataProcessor

# Set page config
st.set_page_config(
    page_title="ML Model Trainer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for better styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        padding: 0rem 1rem;
        font-size: 1rem;
    }
    /* Active tab */
    .stTabs [aria-selected="true"] {
        background-color: rgba(75, 139, 245, 0.1);
        font-weight: bold;
    }
    /* Streamlit info box */
    .stAlert {
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None
if "features" not in st.session_state:
    st.session_state.features = {"numeric": [], "categorical": []}
if "target" not in st.session_state:
    st.session_state.target = None
if "uploaded_dataset" not in st.session_state:
    st.session_state.uploaded_dataset = None
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Data Selection"

# Add new session state variables
if "data_processor" not in st.session_state:
    st.session_state.data_processor = None
if "trained_model" not in st.session_state:
    st.session_state.trained_model = None
if "model_results" not in st.session_state:
    st.session_state.model_results = None
if "is_classification" not in st.session_state:
    st.session_state.is_classification = None
if "training_tab_state" not in st.session_state:
    st.session_state.training_tab_state = {
        "selected_model": None,
        "test_size": 0.2,
        "random_state": 42,
        "use_grid_search": True,
        "cv_folds": 5
    }

# App title and description
st.title("Interactive ML Model Trainer")
st.markdown("""
This application allows you to train machine learning models interactively.
Select a dataset, explore the data, choose features, and train various models to find the best one.
""")

# Helper functions
def get_available_seaborn_datasets():
    """Return a list of available datasets in seaborn."""
    # Get all dataset names from seaborn
    dataset_names = sns.get_dataset_names()
    
    # Create a mapping from names to a function that will load that dataset
    name_to_func = {name: name for name in dataset_names}
    
    # Convert names to prettier format for display
    pretty_names = {name: name.replace('_', ' ').title() for name in dataset_names}
    
    # Create final mapping of pretty names to original names
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
                
                # If we have a dataset name, try to load that dataset
                if st.session_state.dataset_name:
                    try:
                        load_dataset(st.session_state.dataset_name)
                    except:
                        pass  # If loading fails, we'll just start fresh
    except:
        # If loading fails for any reason, we'll just start fresh
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
            # Load the dataset using seaborn's load_dataset function
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
        # Save the current state
        save_session_state()

def check_high_correlations(df, numeric_features, threshold=0.95):
    """
    Check for highly correlated features in the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataset
    numeric_features : list
        List of numeric feature names
    threshold : float, default=0.95
        Correlation threshold to consider features as highly correlated
    
    Returns:
    --------
    list of tuples
        List of (feature1, feature2, correlation) for highly correlated pairs
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

# Sidebar for dataset selection
with st.sidebar:
    st.header("Dataset Selection")
    
    # Get available seaborn datasets
    name_to_func = get_available_seaborn_datasets()
    dataset_options = list(name_to_func.keys())
    dataset_options.append("Upload Custom Dataset")
    
    selected_dataset = st.selectbox(
        "Choose a dataset:",
        options=dataset_options,
        index=0 if st.session_state.dataset_name is None else dataset_options.index(st.session_state.dataset_name) if st.session_state.dataset_name in dataset_options else 0
    )
    
    # Handle custom dataset upload
    if selected_dataset == "Upload Custom Dataset":
        uploaded_file = st.file_uploader("Upload your CSV file:", type=["csv"])
        if uploaded_file is not None:
            # Read the file
            df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_dataset = df
            
            # Show preview of the uploaded data
            st.write("Preview of uploaded data:")
            st.dataframe(df.head(5), use_container_width=True)
            
            if st.button("Use this dataset"):
                load_dataset("Upload Custom Dataset")
                st.success("Custom dataset loaded successfully!")
                st.session_state.active_tab = "Data Selection"
    else:
        # Button to load selected seaborn dataset
        if st.button("Load Dataset"):
            load_dataset(selected_dataset)
            st.success(f"{selected_dataset} dataset loaded successfully!")
            st.session_state.active_tab = "Data Selection"
    
    # Display current dataset and target if selected
    if st.session_state.dataset is not None:
        st.write("---")
        st.write(f"**Current Dataset:** {st.session_state.dataset_name}")
        st.write(f"**Rows:** {st.session_state.dataset.shape[0]}, **Columns:** {st.session_state.dataset.shape[1]}")
        
        if st.session_state.target:
            st.write(f"**Target Variable:** {st.session_state.target}")
    
    # Add about section
    st.sidebar.write("---")
    st.sidebar.info("""
    ### About
    This interactive tool allows you to:
    - Select or upload datasets
    - Explore and visualize data
    - Select features for modeling
    - Train and evaluate ML models
    """)

# Main area with tabs
if st.session_state.dataset is not None:
    # Create tabs
    tabs = ["Data Selection", "Data Exploration", "Feature Selection", "Model Training", "Model Evaluation"]
    active_tab = st.tabs(tabs)
    
    # Data Selection Tab
    with active_tab[0]:
        st.header("Dataset: " + st.session_state.dataset_name)
        
        # Display dataset info
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dataset Preview")
            st.dataframe(st.session_state.dataset.head(10), use_container_width=True)
        
        with col2:
            st.subheader("Dataset Info")
            buffer = StringIO()
            st.session_state.dataset.info(buf=buffer)
            st.text(buffer.getvalue())
            
            # Basic statistics
            st.subheader("Basic Statistics")
            st.dataframe(st.session_state.dataset.describe(), use_container_width=True)
    
    # Data Exploration Tab
    with active_tab[1]:
        st.header("Data Exploration")

        if "original_features" not in st.session_state:
            st.session_state.original_features = {
                "numeric": st.session_state.features["numeric"].copy(),
                "categorical": st.session_state.features["categorical"].copy()
            }
        
        high_corr_pairs = check_high_correlations(st.session_state.dataset, st.session_state.original_features["numeric"])
        if high_corr_pairs:
            st.warning("⚠️ High correlation detected between features. Consider potential information leakage when selecting features and targets.")
            for feat1, feat2, corr in high_corr_pairs:
                st.write(f"- '{feat1}' and '{feat2}' (correlation: {corr:.3f})")
            st.write("---")
        
        # Correlation matrix for numeric features
        if st.session_state.original_features["numeric"]:
            st.subheader("Correlation Matrix")
            corr_fig = plot_correlation_matrix(st.session_state.dataset, st.session_state.original_features["numeric"])
            st.pyplot(corr_fig)
        
        # Distribution of numeric features
        if st.session_state.original_features["numeric"]:
            st.subheader("Numeric Feature Distributions")
            # Allow selecting which features to plot
            selected_numeric = st.multiselect(
                "Select numeric features to visualize:",
                options=st.session_state.original_features["numeric"],
                default=st.session_state.original_features["numeric"][:min(5, len(st.session_state.original_features["numeric"]))]
            )
            
            if selected_numeric:
                # Add option to show distribution by target
                show_by_target = False
                target_var = None
                
                if (st.session_state.target and 
                    st.session_state.target in st.session_state.dataset.columns and 
                    st.session_state.dataset[st.session_state.target].nunique() <= 10):
                    show_by_target = st.checkbox("Show distributions by target variable")
                    if show_by_target:
                        target_var = st.session_state.target
                
                num_dist_fig = plot_numeric_distributions(
                    st.session_state.dataset, 
                    selected_numeric, 
                    target=target_var if show_by_target else None
                )
                st.pyplot(num_dist_fig)
        
        # Distribution of categorical features
        if st.session_state.original_features["categorical"]:
            st.subheader("Categorical Feature Distributions")
            # Allow selecting which features to plot
            selected_categorical = st.multiselect(
                "Select categorical features to visualize:",
                options=st.session_state.original_features["categorical"],
                default=st.session_state.original_features["categorical"][:min(5, len(st.session_state.original_features["categorical"]))]
            )
            
            if selected_categorical:
                # Add option to show distribution by target
                show_by_target = False
                target_var = None
                
                if (st.session_state.target and 
                    st.session_state.target in st.session_state.dataset.columns and 
                    st.session_state.dataset[st.session_state.target].nunique() <= 10):
                    show_by_target = st.checkbox("Show categorical distributions by target variable")
                    if show_by_target:
                        target_var = st.session_state.target
                
                cat_dist_fig = plot_categorical_distributions(
                    st.session_state.dataset, 
                    selected_categorical, 
                    target=target_var if show_by_target else None
                )
                st.pyplot(cat_dist_fig)
    
    # Feature Selection Tab
    with active_tab[2]:
        st.header("Feature Selection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Numeric Features")
            if st.session_state.original_features["numeric"]:
                selected_numeric = st.multiselect(
                    "Select numeric features for modeling:",
                    options=st.session_state.original_features["numeric"],
                    default=st.session_state.features["numeric"]
                )
                st.session_state.features["numeric"] = selected_numeric
            else:
                st.info("No numeric features detected in this dataset.")
        
        with col2:
            st.subheader("Categorical Features")
            if st.session_state.original_features["categorical"]:
                selected_categorical = st.multiselect(
                    "Select categorical features for modeling:",
                    options=st.session_state.original_features["categorical"],
                    default=st.session_state.features["categorical"]
                )
                st.session_state.features["categorical"] = selected_categorical
            else:
                st.info("No categorical features detected in this dataset.")
        
        # Target variable selection
        all_columns = st.session_state.dataset.columns.tolist()
        st.subheader("Target Variable")
        selected_target = st.selectbox(
            "Select the target variable:",
            options=all_columns,
            index=0 if st.session_state.target is None else all_columns.index(st.session_state.target) if st.session_state.target in all_columns else 0
        )
        
        if st.button("Set as Target"):
            st.session_state.target = selected_target
            # If target is in features, remove it
            if selected_target in st.session_state.features["numeric"]:
                st.session_state.features["numeric"].remove(selected_target)
            if selected_target in st.session_state.features["categorical"]:
                st.session_state.features["categorical"].remove(selected_target)
            save_session_state()
            st.success(f"Target variable set to: {selected_target}")
        
        if st.session_state.target:
            st.info(f"Current target variable: {st.session_state.target}")
            
            # Display correlation with target for numeric features
            if st.session_state.features["numeric"] and st.session_state.target in st.session_state.dataset.columns:
                if st.session_state.dataset[st.session_state.target].dtype in ['int64', 'float64']:
                    st.subheader("Correlation with Target")
                    
                    # Calculate correlations
                    corr_data = st.session_state.dataset[st.session_state.features["numeric"] + [st.session_state.target]]
                    correlations = corr_data.corr()[st.session_state.target].sort_values(ascending=False)
                    correlations = correlations.drop(st.session_state.target)
                    
                    # Plot correlations
                    fig, ax = plt.subplots(figsize=(10, 6))
                    correlations.plot(kind='bar', ax=ax)
                    plt.title(f'Feature Correlation with {st.session_state.target}')
                    plt.tight_layout()
                    st.pyplot(fig)
    
    # Model Training Tab
    with active_tab[3]:
        st.header("Model Training")
        
        if not st.session_state.target:
            st.warning("Please select a target variable in the Feature Selection tab before proceeding with model training.")
        elif not (st.session_state.features["numeric"] or st.session_state.features["categorical"]):
            st.warning("Please select at least one feature in the Feature Selection tab before proceeding with model training.")
        else:
            # Initialize data processor if not already done
            if st.session_state.data_processor is None:
                st.session_state.data_processor = DataProcessor()

            st.subheader("Data Preprocessing")
            analysis = st.session_state.data_processor.analyze_data_characteristics(
                st.session_state.dataset,
                st.session_state.features["numeric"],
                st.session_state.features["categorical"]
            )
            
            if analysis['has_outliers']:
                st.warning("⚠️ Outliers detected in the following features: " + 
                         ", ".join(analysis['outlier_features']))
                st.info("💡 Suggestion: Consider using robust scaling and median imputation for better handling of outliers.")
            
            if analysis['missing_value_patterns']:
                st.warning("⚠️ Missing values detected:")
                if 'numeric' in analysis['missing_value_patterns']:
                    st.write("Numeric features with missing values:")
                    for feat, count in analysis['missing_value_patterns']['numeric']['features'].items():
                        st.write(f"- {feat}: {count} missing values")
                if 'categorical' in analysis['missing_value_patterns']:
                    st.write("Categorical features with missing values:")
                    for feat, count in analysis['missing_value_patterns']['categorical']['features'].items():
                        st.write(f"- {feat}: {count} missing values")
                
            with st.expander("Configure Preprocessing", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Numeric Features")
                    if st.session_state.features["numeric"]:
                        suggested_num_imputer = analysis.get('suggested_strategies', {}).get('numeric_imputer', 'mean')
                        num_imputer_options = {
                            'mean': 'Mean (default)',
                            'median': 'Median (robust to outliers)',
                            'knn': 'K-Nearest Neighbors (for complex patterns)',
                        }
                        num_imputer = st.selectbox(
                            "Missing Value Strategy:",
                            options=list(num_imputer_options.keys()),
                            format_func=lambda x: num_imputer_options[x],
                            index=list(num_imputer_options.keys()).index(suggested_num_imputer),
                            help="Strategy for handling missing values in numeric features"
                        )

                        suggested_scaler = analysis.get('suggested_strategies', {}).get('numeric_scaler', 'standard')
                        scaler_options = {
                            'standard': 'Standard Scaler (default)',
                            'robust': 'Robust Scaler (handles outliers better)'
                        }
                        scaler_type = st.selectbox(
                            "Scaling Method:",
                            options=list(scaler_options.keys()),
                            format_func=lambda x: scaler_options[x],
                            index=list(scaler_options.keys()).index(suggested_scaler),
                            help="Method for scaling numeric features"
                        )
                    else:
                        st.info("No numeric features selected.")
                
                with col2:
                    st.subheader("Categorical Features")
                    if st.session_state.features["categorical"]:
                        cat_imputer_options = {
                            'most_frequent': 'Most Frequent (default)',
                            'constant': 'Constant (specify value)'
                        }
                        cat_imputer = st.selectbox(
                            "Missing Value Strategy:",
                            options=list(cat_imputer_options.keys()),
                            format_func=lambda x: cat_imputer_options[x],
                            help="Strategy for handling missing values in categorical features"
                        )
                        
                        if cat_imputer == 'constant':
                            constant_value = st.text_input(
                                "Constant Value:",
                                value='missing',
                                help="Value to use for filling missing categorical values"
                            )
                    else:
                        st.info("No categorical features selected.")

                if st.button("Apply Preprocessing Configuration"):
                    config = {
                        'numeric_imputer_strategy': num_imputer if st.session_state.features["numeric"] else 'mean',
                        'categorical_imputer_strategy': cat_imputer if st.session_state.features["categorical"] else 'most_frequent',
                        'numeric_scaler_type': scaler_type if st.session_state.features["numeric"] else 'standard'
                    }
                    st.session_state.data_processor.configure_preprocessing(config)
                    st.success("Preprocessing configuration applied successfully!")
                    
            st.markdown("---")
                    
            # Determine if classification or regression task
            if st.session_state.is_classification is None:
                target_series = st.session_state.dataset[st.session_state.target]
                st.session_state.is_classification = (
                    target_series.dtype == 'object' or 
                    target_series.dtype.name == 'category' or 
                    target_series.nunique() < 10
                )
            
            # Create two columns for model selection and training parameters
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Model Selection")
                
                # Get available models based on task type
                available_models = (
                    CLASSIFICATION_MODELS if st.session_state.is_classification 
                    else REGRESSION_MODELS
                )
                
                # Model selection
                selected_model = st.selectbox(
                    "Select a model:",
                    options=list(available_models.keys()),
                    index=0 if st.session_state.training_tab_state["selected_model"] is None 
                    else list(available_models.keys()).index(st.session_state.training_tab_state["selected_model"])
                )
                st.session_state.training_tab_state["selected_model"] = selected_model
                
                # Display model description
                st.markdown(f"""
                **Selected Model**: {selected_model}
                
                This model is suitable for {'classification' if st.session_state.is_classification else 'regression'} tasks.
                """)
            
            with col2:
                st.subheader("Training Parameters")
                
                # Training parameters
                test_size = st.slider(
                    "Test Set Size:",
                    min_value=0.1,
                    max_value=0.4,
                    value=st.session_state.training_tab_state["test_size"],
                    step=0.05,
                    help="Proportion of data to use for testing"
                )
                st.session_state.training_tab_state["test_size"] = test_size

                random_state_options = {
                    "None (non-deterministic)": None,
                    "0 (zero seed)": 0,
                    "42 (common seed)": 42
                }
                
                random_state_selection= st.selectbox(
                    "Seed for reproducibility (random_state)",
                    options=list(random_state_options.keys()),
                    index=2, #default is 42, most frequent seed
                    help="Seed for reproducibility. Use None for non-deterministic behavior, or a fixed value for reproducible results."
                )
                random_state = random_state_options[random_state_selection]
                st.session_state.training_tab_state["random_state"] = random_state
                
                use_grid_search = st.checkbox(
                    "Use Grid Search",
                    value=st.session_state.training_tab_state["use_grid_search"],
                    help="Enable hyperparameter tuning using grid search"
                )
                st.session_state.training_tab_state["use_grid_search"] = use_grid_search
                
                if use_grid_search:
                    cv_folds = st.slider(
                        "Cross-validation Folds:",
                        min_value=2,
                        max_value=10,
                        value=st.session_state.training_tab_state["cv_folds"],
                        help="Number of folds for cross-validation"
                    )
                    st.session_state.training_tab_state["cv_folds"] = cv_folds
            
            # Training button and progress
            if st.button("Train Model", type="primary"):
                with st.spinner("Training model..."):
                    try:
                        # Prepare the data
                        X = st.session_state.dataset[
                            st.session_state.features["numeric"] + 
                            st.session_state.features["categorical"]
                        ]
                        y = st.session_state.dataset[st.session_state.target]
                        
                        # Preprocess features
                        X_processed = st.session_state.data_processor.preprocess_features(
                            X,
                            st.session_state.features["numeric"],
                            st.session_state.features["categorical"]
                        )
                        
                        # Preprocess target
                        y_processed = st.session_state.data_processor.preprocess_target(y)
                        
                        # Train the model
                        results = train_model(
                            X_processed,
                            y_processed,
                            selected_model,
                            is_classification=st.session_state.is_classification,
                            test_size=test_size,
                            random_state=random_state,
                            cv=cv_folds if use_grid_search else None,
                            use_grid_search=use_grid_search
                        )
                        
                        # Store results in session state
                        st.session_state.trained_model = results['model']
                        st.session_state.model_results = results

                        st.success("Model trained successfully! Go to the Model Evaluation tab to see the results.")

                        st.markdown("---")
                        st.subheader("Save Model")
                        save_col1, save_col2 = st.columns([2, 1])
                        with save_col1:
                            model_filename = st.text_input(
                                "Model filename (without extension):",
                                value=f"{selected_model}_{st.session_state.dataset_name}",
                                help="Enter a name for your model file"
                            )
                        with save_col2:
                            if st.button("Save Model", key="save_model_btn"):
                                try:
                                    save_model(
                                        results['model'],
                                        model_filename,
                                        feature_names=st.session_state.data_processor.get_feature_names(),
                                        target_encoder=st.session_state.data_processor.label_encoder
                                    )
                                    st.success(f"Model saved successfully as '{model_filename}'!")
                                except Exception as e:
                                    st.error(f"Error saving model: {str(e)}")
                        
                    except Exception as e:
                        st.error(f"An error occurred during training: {str(e)}")
                        st.session_state.trained_model = None
                        st.session_state.model_results = None
                        
    
    # Model Evaluation Tab
    with active_tab[4]:
        st.header("Model Evaluation")
        
        if st.session_state.trained_model is None:
            st.warning("Please train a model first in the Model Training tab.")
        else:
            # Get the results
            results = st.session_state.model_results
            
            # Create tabs for different evaluation aspects
            eval_tabs = st.tabs([
                "Model Performance",
                "Feature Importance",
                "Predictions Analysis"
            ])
            
            # Model Performance Tab
            with eval_tabs[0]:
                st.subheader("Model Performance Metrics")
                
                # Display metrics based on task type
                if st.session_state.is_classification:
                    metrics_df = pd.DataFrame({
                        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                        'Value': [
                            results['metrics']['accuracy'],
                            results['metrics']['precision'],
                            results['metrics']['recall'],
                            results['metrics']['f1']
                        ]
                    })
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    # Plot confusion matrix if available
                    if results['predictions']['y_test'] is not None:
                        st.subheader("Confusion Matrix")
                        fig = plot_confusion_matrix(
                            results['predictions']['y_test'],
                            results['predictions']['y_pred']
                        )
                        st.pyplot(fig)
                    
                    # Plot ROC curve if probabilities are available
                    if results['predictions']['y_prob'] is not None:
                        st.subheader("ROC Curve")
                        fig = plot_roc_curve(
                            results['predictions']['y_test'],
                            results['predictions']['y_prob']
                        )
                        st.pyplot(fig)
                else:
                    metrics_df = pd.DataFrame({
                        'Metric': ['MSE', 'RMSE', 'MAE', 'R² Score'],
                        'Value': [
                            results['metrics']['mse'],
                            results['metrics']['rmse'],
                            results['metrics']['mae'],
                            results['metrics']['r2']
                        ]
                    })
                    st.dataframe(metrics_df, use_container_width=True)
            
            # Feature Importance Tab
            with eval_tabs[1]:
                st.subheader("Feature Importance")
                
                if results['feature_importances'] is not None:
                    fig = plot_feature_importances(
                        results['feature_importances'],
                        st.session_state.data_processor.get_feature_names()
                    )
                    st.pyplot(fig)
                else:
                    st.info("Feature importance not available for this model.")
            
            # Predictions Analysis Tab
            with eval_tabs[2]:
                st.subheader("Predictions Analysis")
                
                # Create a dataframe with actual vs predicted values
                predictions_df = pd.DataFrame({
                    'Actual': results['predictions']['y_test'],
                    'Predicted': results['predictions']['y_pred']
                })
                
                # Display sample of predictions
                st.write("Sample of Predictions (first 10 rows):")
                st.dataframe(predictions_df.head(10), use_container_width=True)
                
                # Add download button for all predictions
                csv = predictions_df.to_csv(index=False)
                st.download_button(
                    label="Download All Predictions",
                    data=csv,
                    file_name="model_predictions.csv",
                    mime="text/csv"
                )
                
                # Plot actual vs predicted values for regression
                if not st.session_state.is_classification:
                    st.subheader("Actual vs Predicted Values")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(predictions_df['Actual'], predictions_df['Predicted'], alpha=0.5)
                    ax.plot([predictions_df['Actual'].min(), predictions_df['Actual'].max()],
                           [predictions_df['Actual'].min(), predictions_df['Actual'].max()],
                           'r--', lw=2)
                    ax.set_xlabel('Actual Values')
                    ax.set_ylabel('Predicted Values')
                    ax.set_title('Actual vs Predicted Values')
                    st.pyplot(fig)
else:
    st.info("👈 Please select a dataset from the sidebar to get started.")

# Load session state on startup
load_session_state()
