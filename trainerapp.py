import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from io import StringIO
import json

from utils.app_utils import (
    get_available_seaborn_datasets,
    load_dataset,
    save_session_state,
    load_session_state
)

from utils.callbacks import (
    initialize_session_state,
    on_dataset_change,
    on_target_change,
    on_feature_selection,
    on_preprocessing_config_change,
    on_viz_settings_change,
    on_model_train
)

from utils.fragments import (
    render_dataset_overview,
    render_data_preview,
    render_feature_distributions,
    render_model_metrics,
    render_confusion_matrix,
    render_roc_curve,
    render_feature_importance
)

from utils.visualizer import (
    plot_correlation_matrix,
    plot_categorical_distributions,
    plot_numeric_distributions,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importances,
)

from utils.altair_visualizer import (
    plot_correlation_matrix_alt,
    plot_feature_importances_alt,
    plot_distplot_alt,
    plot_boxplot_alt,
    plot_confusion_matrix_alt,
    plot_roc_curve_alt,
    plot_scatter_matrix_alt,
    plot_categorical_distributions_alt,
    plot_parameter_comparison_alt,
    plot_model_comparison_alt
)

# Import model utilities
from models.model_utils import (
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
    train_model,
    save_model,
)
from models.model_history import ModelHistory

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
initialize_session_state()

# App title and description
st.title("Interactive ML Model Trainer")
st.markdown("""
This application allows you to train machine learning models interactively.
Select a dataset, explore the data, choose features, and train various models to find the best one.
""")

# Add sidebar section for settings
with st.sidebar:
    st.header("Settings")
    
    # Visualization settings
    st.subheader("Visualization")
    current_viz_lib = "Altair (Interactive)" if st.session_state.viz_settings["library"] == "altair" else "Matplotlib/Seaborn"
    viz_lib = st.radio(
        "Visualization Library",
        options=["Altair (Interactive)", "Matplotlib/Seaborn"],
        index=["Altair (Interactive)", "Matplotlib/Seaborn"].index(current_viz_lib),
        help="Choose the visualization library. Altair provides interactive plots.",
        key="viz_lib_radio",
        on_change=on_viz_settings_change,
        args=("altair",) if st.session_state.get("viz_lib_radio") == "Altair (Interactive)" else ("matplotlib",)
    )
    
    st.markdown("---")

with st.sidebar:
    st.header("Dataset Selection")
    
    # Get available seaborn datasets
    name_to_func = get_available_seaborn_datasets()
    dataset_options = list(name_to_func.keys())
    dataset_options.append("Upload Custom Dataset")
    
    selected_dataset = st.selectbox(
        "Choose a dataset:",
        options=dataset_options,
        index=0 if st.session_state.dataset_name is None else dataset_options.index(st.session_state.dataset_name) if st.session_state.dataset_name in dataset_options else 0,
        key="selected_dataset",
        on_change=on_dataset_change
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
                on_dataset_change()
                st.success("Custom dataset loaded successfully!")
    else:
        # Button to load selected seaborn dataset
        if st.button("Load Dataset"):
            on_dataset_change()
            st.success(f"{selected_dataset} dataset loaded successfully!")
    
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
    tabs = ["Data Selection", "Data Exploration", "Feature Selection", "Model Training", "Model Evaluation", "Model History"]
    active_tab = st.tabs(tabs)
    
    # Data Selection Tab
    with active_tab[0]:
        st.header("Dataset: " + st.session_state.dataset_name)

        st.subheader("🎯 Target Variable Selection")
        all_columns = st.session_state.dataset.columns.tolist()
        col1, col2 = st.columns([3,1])

        with col1:
            selected_target = st.selectbox(
                "Select the target variable for your analysis:",
                options=all_columns,
                index=0 if st.session_state.target is None else all_columns.index(st.session_state.target) if st.session_state.target in all_columns else 0,
                help="This is the variable you want to predict",
                key="data_selection_target",
                on_change=on_target_change
            )
        
        with col2:
            if st.button("Set Target", type="primary", key="data_selection_set_target"):
                on_target_change()
                st.success(f"Target variable set to: {selected_target}")
                    
        st.markdown("---")

        st.subheader("📊 Dataset Overview")
        render_dataset_overview(st.session_state.dataset, st.session_state.features)
        
        # Create tabs for different dataset views
        data_tabs = st.tabs(["Preview", "Summary Statistics"])

        with data_tabs[0]:
            st.subheader("Data Preview")
            n_rows = st.slider("Number of rows to display", 5, 50, 10)
            selected_columns = st.multiselect(
                "Select columns to display",
                options=all_columns,
                default=all_columns[:6] if len(all_columns) > 6 else all_columns
            )

            render_data_preview(
                st.session_state.dataset,
                selected_columns,
                n_rows,
                st.session_state.target
            )
        
        with data_tabs[1]:
            st.subheader("Summary Statistics")

            stat_type = st.radio(
                "Select feature type:",
                ["Numeric", "Categorical", "All"],
                horizontal=True
            )

            if stat_type == "Numeric":
                if st.session_state.features["numeric"]:
                    stats_df = st.session_state.dataset[st.session_state.features["numeric"]].describe()
                    st.dataframe(stats_df, use_container_width=True)
                else:
                    st.info("No numeric features available in the dataset.")

            elif stat_type == "Categorical":
                if st.session_state.features["categorical"]:
                    cat_stats = {}
                    for col in st.session_state.features["categorical"]:
                        value_counts = st.session_state.dataset[col].value_counts()
                        unique_count = len(value_counts)
                        most_common = value_counts.index[0]
                        most_common_count = value_counts.iloc[0]
                        missing = st.session_state.dataset[col].isnull().sum()
                        
                        cat_stats[col] = {
                            "Unique Count": unique_count,
                            "Most Common": most_common,
                            "Most Common Count": most_common_count,
                            "Missing Values": missing
                        }
                    
                    st.dataframe(pd.DataFrame(cat_stats).T, use_container_width=True)
                else:
                    st.info("No categorical features available in the dataset.")
            else:
                st.write("Numeric Features:")
                if st.session_state.features["numeric"]:
                    st.dataframe(st.session_state.dataset[st.session_state.features["numeric"]].describe(), use_container_width=True)
                else:
                    st.info("No numeric features available.")

                st.write("Categorical Features:")
                if st.session_state.features["categorical"]:
                    cat_stats = {}
                    for col in st.session_state.features["categorical"]:
                        value_counts = st.session_state.dataset[col].value_counts()
                        cat_stats[col] = {
                            "Unique Values": len(value_counts),
                            "Most Common": value_counts.index[0],
                            "Most Common Count": value_counts.iloc[0],
                            "Missing Values": st.session_state.dataset[col].isnull().sum()
                        }
                    st.dataframe(pd.DataFrame(cat_stats).T, use_container_width=True)
                else:
                    st.info("No categorical features available.")
                    
            
    # Data Exploration Tab
    with active_tab[1]:
        st.header("Data Exploration")

        if "original_features" not in st.session_state:
            st.session_state.original_features = {
                "numeric": st.session_state.features["numeric"].copy(),
                "categorical": st.session_state.features["categorical"].copy()
            }
        
        # Correlation matrix for numeric features
        if st.session_state.original_features["numeric"]:
            st.subheader("Correlation Matrix")
            
            # Select visualization based on user preference
            if st.session_state.viz_settings["library"] == "altair":
                corr_chart = plot_correlation_matrix_alt(
                    st.session_state.dataset[st.session_state.original_features["numeric"]]
                )
                st.altair_chart(corr_chart, use_container_width=True)
            else:
                corr_fig = plot_correlation_matrix(
                    st.session_state.dataset, 
                    st.session_state.original_features["numeric"]
                )
                st.pyplot(corr_fig)
                
            # Add scatter plot matrix for Altair
            if st.session_state.viz_settings["library"] == "altair" and len(st.session_state.original_features["numeric"]) > 1:
                st.subheader("Feature Relationships")
                
                # Allow selecting features to include
                selected_features_scatter = st.multiselect(
                    "Select features for scatter plot matrix:",
                    options=st.session_state.original_features["numeric"],
                    default=st.session_state.original_features["numeric"][:min(4, len(st.session_state.original_features["numeric"]))]
                )
                
                if selected_features_scatter and len(selected_features_scatter) > 1:
                    # Choose whether to color by target
                    color_by = None
                    if st.session_state.target and (
                        st.session_state.dataset[st.session_state.target].nunique() <= 10 or 
                        st.session_state.dataset[st.session_state.target].dtype == 'object' or
                        st.session_state.dataset[st.session_state.target].dtype.name == 'category'
                    ):
                        color_by_target = st.checkbox("Color by target variable")
                        if color_by_target:
                            color_by = st.session_state.target
                    
                    # Create the chart
                    scatter_matrix = plot_scatter_matrix_alt(
                        st.session_state.dataset,
                        selected_features_scatter,
                        color_by=color_by
                    )
                    st.altair_chart(scatter_matrix, use_container_width=True)
        
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
                # Check if target variable is suitable for distribution plots
                show_by_target = False
                if (st.session_state.target and 
                    st.session_state.target in st.session_state.dataset.columns and 
                    st.session_state.dataset[st.session_state.target].nunique() <= 10):
                    show_by_target = st.checkbox("Show numeric distributions by target variable")
                
                render_feature_distributions(
                    st.session_state.dataset, 
                    selected_numeric, 
                    "numeric",
                    st.session_state.target,
                    st.session_state.viz_settings["library"],
                    show_by_target
                )
        
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
                # Check if target variable is suitable for distribution plots
                show_by_target = False
                if (st.session_state.target and 
                    st.session_state.target in st.session_state.dataset.columns and 
                    st.session_state.dataset[st.session_state.target].nunique() <= 10):
                    show_by_target = st.checkbox("Show categorical distributions by target variable")
                
                render_feature_distributions(
                    st.session_state.dataset, 
                    selected_categorical, 
                    "categorical",
                    st.session_state.target,
                    st.session_state.viz_settings["library"],
                    show_by_target
                )
    
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
                    default=st.session_state.features["numeric"],
                    key="selected_numeric",
                    on_change=on_feature_selection,
                    args=("numeric",)
                )
            else:
                st.info("No numeric features detected in this dataset.")
        
        with col2:
            st.subheader("Categorical Features")
            if st.session_state.original_features["categorical"]:
                selected_categorical = st.multiselect(
                    "Select categorical features for modeling:",
                    options=st.session_state.original_features["categorical"],
                    default=st.session_state.features["categorical"],
                    key="selected_categorical",
                    on_change=on_feature_selection,
                    args=("categorical",)
                )
            else:
                st.info("No categorical features detected in this dataset.")
        
        # Target variable selection
        all_columns = st.session_state.dataset.columns.tolist()
        col1, col2 = st.columns([3,1])

        with col1:
            selected_target = st.selectbox(
                "Select the target variable:",
                options=all_columns,
                index=0 if st.session_state.target is None else all_columns.index(st.session_state.target) if st.session_state.target in all_columns else 0,
                help="This is the variable you want to predict",
                key="feature_selection_target",
                on_change=on_target_change
            )

        with col2:
            if st.button("Set Target", type="primary", key="feature_selection_set_target"):
                on_target_change()
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
                st.warning("⚠️ Missing values detected")
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
                    on_preprocessing_config_change(config)
                    st.success("Preprocessing configuration applied successfully!")
                    
            st.markdown("---")
                    
            # Determine if classification or regression task
            target_series = st.session_state.dataset[st.session_state.target]
            st.session_state.is_classification = (
                target_series.dtype == 'object' or 
                target_series.dtype.name == 'category' or 
                target_series.nunique() < 10
            )

            available_models = CLASSIFICATION_MODELS if st.session_state.is_classification else REGRESSION_MODELS

            selected_model = st.selectbox(
                "Select a model", 
                         options=list(available_models.keys()), 
                         key="training_tab_state.selected_model", 
                help="Choose a model for training",
                on_change=on_model_train,
                args=(lambda: st.session_state.selected_model, lambda: st.session_state.training_tab_state)
            )
            
            st.markdown(f"""
            **Selected Model**: {selected_model}
            
            This model is suitable for {'classification' if st.session_state.is_classification else 'regression'} tasks.
            """)
            
            st.subheader("Training Parameters")
                
                # Training parameters
            test_size = st.slider(
                "Test Set Size:",
                min_value=0.1,
                max_value=0.4,
                value=st.session_state.training_tab_state["test_size"],
                step=0.05,
                help="Proportion of data to use for testing",
                on_change=on_model_train,
                args=(selected_model, lambda: {"test_size": st.session_state.test_size})
            )

            random_state_options = {
                "None (non-deterministic)": None,
                "0 (zero seed)": 0,
                "42 (common seed)": 42
            }
                
            random_state_selection = st.selectbox(
                "Seed for reproducibility (random_state)",
                options=list(random_state_options.keys()),
                index=2, #default is 42, most frequent seed
                help="Seed for reproducibility. Use None for non-deterministic behavior, or a fixed value for reproducible results.",
                on_change=on_model_train,
                args=(selected_model, lambda: {"random_state": random_state_options[st.session_state.random_state_selection]})
            )
                
            use_grid_search = st.checkbox(
                "Use Grid Search",
                value=st.session_state.training_tab_state["use_grid_search"],
                help="Enable hyperparameter tuning using grid search",
                on_change=on_model_train,
                args=(selected_model, lambda: {"use_grid_search": st.session_state.use_grid_search})
            )
                
            if use_grid_search:
                cv_folds = st.slider(
                    "Cross-validation Folds:",
                    min_value=2,
                    max_value=10,
                    value=st.session_state.training_tab_state["cv_folds"],
                    help="Number of folds for cross-validation",
                    on_change=on_model_train,
                    args=(selected_model, lambda: {"cv_folds": st.session_state.cv_folds})
                )
            
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
                        test_size=st.session_state.training_tab_state["test_size"],
                        random_state=st.session_state.training_tab_state["random_state"],
                        cv=st.session_state.training_tab_state["cv_folds"] if st.session_state.training_tab_state["use_grid_search"] else None,
                        use_grid_search=st.session_state.training_tab_state["use_grid_search"]
                    )
                        
                        # Store results in session state
                    st.session_state.trained_model = results['model']
                    st.session_state.model_results = results

                    st.success("Model trained successfully! Go to the Model Evaluation tab to see the results.")

                        # Save to model history
                    st.session_state.latest_training_results = {
                            'model_info': {
                                'model_name': selected_model,
                                'model_type': 'Classification' if st.session_state.is_classification else 'Regression'
                            },
                            'metrics': results['metrics'],
                            'dataset_name': st.session_state.dataset_name,
                            'features': st.session_state.features["numeric"] + st.session_state.features["categorical"],
                            'target': st.session_state.target,
                            'model_params': results['best_params']
                        }
                    
                    st.markdown("---")
                    st.subheader("Save Model")

                    save_col1, save_col2 = st.columns([2, 1])
                    with save_col1:
                        model_filename = st.text_input(
                            "Model filename (without extension):",
                                value=f"{selected_model}_{st.session_state.dataset_name}".replace(" ", "_"),
                                help="Enter a name for your model file",
                                key="model_filename_input"
                            )
                    with save_col2:
                        if st.button("Save Model", key="save_model_btn", use_container_width=True):
                            try:
                                st.write("Starting model save process...")  # Debug message
                                    
                                    # Verify model exists in results
                                if 'model' not in results:
                                    raise ValueError("No trained model found in results")
                                    
                                    # Get feature names
                                feature_names = (
                                    st.session_state.features["numeric"] + 
                                    st.session_state.features["categorical"]
                                )
                                st.write(f"Feature names prepared: {len(feature_names)} features")  # Debug message
                                    
                                    # Save the model
                                st.write("Attempting to save model...")  # Debug message
                                model_path = save_model(
                                    model=results['model'],
                                    model_name=model_filename,
                                    feature_names=feature_names,
                                    target_encoder=st.session_state.data_processor.label_encoder if st.session_state.is_classification else None
                                )
                                    
                                if os.path.exists(model_path):
                                    st.success(f"Model saved successfully to: {model_path}")
                                    st.write(f"File size: {os.path.getsize(model_path)} bytes")  # Debug message
                                else:
                                    st.error(f"Model file was not created at: {model_path}")
                                    
                            except Exception as e:
                                st.error(f"Error saving model: {str(e)}")
                                st.error("Please check the console for detailed error information.")
                                import traceback
                                st.write("Detailed error:")
                                st.code(traceback.format_exc())
                except Exception as e:
                    st.error(f"An error occurred during training: {str(e)}")
                    st.session_state.trained_model = None
                    st.session_state.model_results = None
                        
                    st.markdown("---")
    
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
                
                # Display metrics
                render_model_metrics(results['metrics'], st.session_state.is_classification)
                    
                    # Plot confusion matrix if available
                if st.session_state.is_classification and results['predictions']['y_test'] is not None:
                        st.subheader("Confusion Matrix")
                    
                    # Get class names if available
                        class_names = None
                        if st.session_state.data_processor and st.session_state.data_processor.label_encoder:
                            class_names = list(st.session_state.data_processor.label_encoder.classes_)
                    
                        render_confusion_matrix(
                            results['predictions']['y_test'],
                            results['predictions']['y_pred'],
                            class_names,
                            st.session_state.viz_settings["library"]
                            )
                    
                    # Plot ROC curve if probabilities are available
                        if results['predictions']['y_prob'] is not None:
                            st.subheader("ROC Curve")
                        
                            render_roc_curve(
                                results['predictions']['y_test'],
                                results['predictions']['y_prob'],
                                class_names,
                                st.session_state.viz_settings["library"]
                            )
                
                # Parameter tuning results if available
                        if results['cv_results'] is not None:
                            st.subheader("Hyperparameter Tuning Results")
                    
                    # Get parameter columns
                            param_cols = [col for col in results['cv_results'].columns if col.startswith('param_')]
                    
                            if param_cols:
                        # Allow user to select a parameter to visualize
                                param_names = [col.replace('param_', '') for col in param_cols]
                                selected_param = st.selectbox(
                                    "Select parameter to visualize:",
                                    options=param_names
                                )
                                param_col = f"param_{selected_param}"
                        
                        # Check if the parameter has multiple values
                        if len(results['cv_results'][param_col].unique()) > 1:
                            # Select visualization based on user preference
                            if st.session_state.viz_settings["library"] == "altair":
                                # Create a parameter effect visualization
                                chart = plot_parameter_comparison_alt(
                                    results['cv_results'],
                                    param_col,
                                    'mean_test_score',
                                    title=f"Effect of {selected_param} on Model Performance"
                                )
                                st.altair_chart(chart, use_container_width=True)
                            else:
                                # Basic matplotlib chart
                                fig, ax = plt.subplots(figsize=(10, 6))
                                sns.lineplot(
                                    x=param_col,
                                    y='mean_test_score',
                                    data=results['cv_results'],
                                    marker='o',
                                    ax=ax
                                )
                                plt.title(f"Effect of {selected_param} on Model Performance")
                                plt.xlabel(selected_param)
                                plt.ylabel('Mean Test Score')
                                plt.grid(True)
                        st.pyplot(fig)
                else:
                        st.info(f"Parameter '{selected_param}' has only one value.")
            
            # Feature Importance Tab
            with eval_tabs[1]:
                st.subheader("Feature Importance")
                
                if results['feature_importances'] is not None:
                    render_feature_importance(
                        st.session_state.data_processor.get_feature_names(),
                        results['feature_importances'],
                        st.session_state.viz_settings["library"]
                    )
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
    
    # Model History tab
    with active_tab[5]:
        st.header("📚 Model History")
        
        # Initialize model history if needed
        if "model_history" not in st.session_state:
            st.session_state.model_history = ModelHistory()
        
        # Check if there are new results to save
        if "latest_training_results" in st.session_state and st.session_state.latest_training_results:
            results = st.session_state.latest_training_results
            
            # Create a unique model name that includes features and parameters
            feature_str = '+'.join(sorted(results['features']))[:30] + ('...' if len('+'.join(results['features'])) > 30 else '')
            param_str = '+'.join(f"{k}={v}" for k, v in results['model_params'].items())[:30] + ('...' if len(str(results['model_params'])) > 30 else '')
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            
            # Update model info with detailed information
            results['model_info'].update({
                'model_name': f"{results['model_info']['model_name']}_{timestamp}",
                'features_used': feature_str,
                'parameters': param_str,
                'timestamp': timestamp
            })
            
            st.session_state.model_history.add_model(
                model_info=results['model_info'],
                metrics=results['metrics'],
                dataset_name=results['dataset_name'],
                features=results['features'],
                target=results['target'],
                model_params=results['model_params']
            )
            # Clear the latest results after saving
            st.session_state.latest_training_results = None
        
        # Get history
        history_df = st.session_state.model_history.get_history(as_dataframe=True)
        
        if history_df.empty:
            st.info("No models have been trained yet. Train a model in the Model Training tab to see history.")
        else:
            # Create a more detailed display DataFrame
            display_df = history_df.copy()
            
            # Format timestamp if it exists
            if 'timestamp' in display_df.columns:
                # First try to convert all timestamps to datetime using a flexible parser
                display_df['timestamp'] = pd.to_datetime(display_df['timestamp'])
                # Then format them consistently
                display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Reorder and rename columns for better display
            display_columns = {
                'model_name': 'Model',
                'timestamp': 'Trained At',
                'dataset_name': 'Dataset',
                'target': 'Target',
                'features_used': 'Features',
                'parameters': 'Parameters'
            }
            
            # Add metric columns
            metric_columns = [col for col in history_df.columns if col.startswith('metric_')]
            for col in metric_columns:
                display_columns[col] = col.replace('metric_', '').replace('_', ' ').title()
            
            # Select and rename columns that exist in the DataFrame
            available_columns = [col for col in display_columns.keys() if col in display_df.columns]
            display_df = display_df[available_columns]
            display_df.columns = [display_columns[col] for col in available_columns]
            
            # Display history table
            st.subheader("Training History")
            st.dataframe(display_df, use_container_width=True)
            
            # Model comparison
            st.subheader("Model Comparison")
            
            # Get metrics columns from original history_df
            metric_columns = [col for col in history_df.columns if col.startswith('metric_')]
            
            if metric_columns:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Select metric for comparison
                    selected_metric = st.selectbox(
                        "Select metric for comparison:",
                        options=metric_columns,
                        format_func=lambda x: x.replace('metric_', '').capitalize()
                    )
                
                with col2:
                    # Add sorting option
                    sort_order = st.selectbox(
                        "Sort order:",
                        options=["Descending", "Ascending"],
                        index=0
                    )
                
                # Filter options
                with st.expander("Comparison Filters"):
                    # Filter by model type (using base model name)
                    model_names = history_df['model_name'].apply(lambda x: x.split('_')[0]).unique()
                    selected_models = st.multiselect(
                        "Filter by model type:",
                        options=model_names,
                        default=model_names
                    )
                    
                    # Filter by dataset
                    datasets = history_df['dataset_name'].unique()
                    selected_datasets = st.multiselect(
                        "Filter by dataset:",
                        options=datasets,
                        default=datasets
                    )
                    
                    # Filter by target
                    targets = history_df['target'].unique()
                    selected_targets = st.multiselect(
                        "Filter by target variable:",
                        options=targets,
                        default=targets
                    )
                
                # Apply filters
                filtered_df = history_df[
                    (history_df['model_name'].apply(lambda x: x.split('_')[0]).isin(selected_models)) &
                    (history_df['dataset_name'].isin(selected_datasets)) &
                    (history_df['target'].isin(selected_targets))
                ]
                
                if not filtered_df.empty:
                    # Sort the DataFrame
                    filtered_df = filtered_df.sort_values(
                        by=selected_metric,
                        ascending=(sort_order == "Ascending")
                    )
                    
                    # Select visualization based on user preference
                    if st.session_state.viz_settings["library"] == "altair":
                        comparison_chart = plot_model_comparison_alt(
                            filtered_df,
                            selected_metric,
                            title=f"Model Comparison by {selected_metric.replace('metric_', '').capitalize()}"
                        )
                        st.altair_chart(comparison_chart, use_container_width=True)
                    else:
                        # Basic matplotlib chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        # Create labels that include timestamp for uniqueness
                        labels = [f"{row['model_name'].split('_')[0]}\n({pd.to_datetime(row['timestamp']).strftime('%H:%M:%S')})"
                                for _, row in filtered_df.iterrows()]
                        
                        # Plot
                        bars = ax.bar(range(len(filtered_df)), filtered_df[selected_metric])
                        ax.set_xticks(range(len(filtered_df)))
                        ax.set_xticklabels(labels, rotation=45, ha='right')
                        ax.set_title(f"Model Comparison by {selected_metric.replace('metric_', '').capitalize()}")
                        ax.set_ylabel(selected_metric.replace('metric_', '').capitalize())
                        
                        # Add value labels on top of bars
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                  f'{height:.3f}',
                                  ha='center', va='bottom')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                else:
                    st.warning("No models match the selected filters.")
            
            # Model deletion
            st.subheader("Delete Model")
            model_ids = filtered_df['model_id'].tolist() if 'filtered_df' in locals() else history_df['model_id'].tolist()
            model_names = filtered_df['model_name'].tolist() if 'filtered_df' in locals() else history_df['model_name'].tolist()
            timestamps = filtered_df['timestamp'].tolist() if 'filtered_df' in locals() else history_df['timestamp'].tolist()
            
            model_options = [f"{name.split('_')[0]} ({pd.to_datetime(ts).strftime('%Y-%m-%d %H:%M:%S')})" 
                           for name, ts in zip(model_names, timestamps)]
            
            if model_options:
                selected_model_idx = st.selectbox(
                    "Select model to delete:",
                    options=range(len(model_options)),
                    format_func=lambda i: model_options[i]
                )
                
                if st.button("Delete Selected Model", type="secondary"):
                    model_id = model_ids[selected_model_idx]
                    if st.session_state.model_history.delete_model(model_id):
                        st.success("Model deleted successfully.")
                        # Refresh history
                        st.rerun()
                    else:
                        st.error("Failed to delete model.")
else:
    st.info("👈 Please select a dataset from the sidebar to get started.")

# Load session state on startup
load_session_state()
