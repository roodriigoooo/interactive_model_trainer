"""
UI fragments for the ML Model Trainer application.
These functions handle different sections of the UI using st.cache_data and st.fragment decorators.
"""
import streamlit as st
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from .visualizer import (
    plot_correlation_matrix,
    plot_categorical_distributions,
    plot_numeric_distributions,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importances,
)
from .altair_visualizer import (
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


@st.cache_data
def render_dataset_overview(dataset: pd.DataFrame, features: Dict[str, List[str]]) -> None:
    """Render dataset overview metrics."""
    overview_cols = st.columns(5)
    overview_cols[0].metric("Rows", dataset.shape[0])
    overview_cols[1].metric("Columns", dataset.shape[1])
    overview_cols[2].metric("Numeric Features", len(features["numeric"]))
    overview_cols[3].metric("Categorical Features", len(features["categorical"]))
    overview_cols[4].metric("Memory Usage", f"{dataset.memory_usage(deep=True).sum()} bytes")


@st.cache_data
def render_data_preview(
    dataset: pd.DataFrame,
    selected_columns: List[str],
    n_rows: int,
    target: Optional[str] = None
) -> None:
    """Render data preview with optional target highlighting."""
    if selected_columns:
        preview_df = dataset[selected_columns].head(n_rows)
        if len(selected_columns) == 1:
            preview_df = preview_df.to_frame()
    else:
        preview_df = dataset.head(n_rows)

    def highlight_target(data):
        if target:
            return pd.Series(
                ['background-color: rgba(75, 139, 245, 0.1)' if data.name == target else ''
                 for _ in range(len(data))],
                 index=data.index
            )
        return pd.Series([''] * len(data), index=data.index)
    
    st.dataframe(
        preview_df.style.apply(highlight_target),
        use_container_width=True
    )


@st.cache_data
def render_feature_distributions(
    dataset: pd.DataFrame,
    features: List[str],
    feature_type: str,
    target: Optional[str] = None,
    viz_library: str = "altair",
    show_by_target: bool = False #to avoid putting a widget inside a cached function
) -> None:
    """Render distribution plots for selected features."""
    if not features:
        return
        
    target_var = target if show_by_target else None
    
    if viz_library == "altair":
        if feature_type == "numeric":
            for feature in features:
                st.subheader(f"Distribution of {feature}")
                if show_by_target and target_var:
                    chart = plot_distplot_alt(
                        dataset, 
                        feature, 
                        title=f"Distribution of {feature} by {target_var}"
                    )
                    st.altair_chart(chart, use_container_width=True)
                    
                    box_chart = plot_boxplot_alt(
                        dataset,
                        feature,
                        by=target_var
                    )
                    st.altair_chart(box_chart, use_container_width=True)
                else:
                    chart = plot_distplot_alt(dataset, feature)
                    st.altair_chart(chart, use_container_width=True)
        else:  # categorical
            charts = plot_categorical_distributions_alt(
                dataset,
                features,
                target=target_var
            )
            for chart in charts:
                st.altair_chart(chart, use_container_width=True)
    else:
        if feature_type == "numeric":
            fig = plot_numeric_distributions(
                dataset, 
                features, 
                target=target_var
            )
        else:
            fig = plot_categorical_distributions(
                dataset, 
                features, 
                target=target_var
            )
        st.pyplot(fig)


@st.cache_data
def render_model_metrics(metrics: Dict[str, float], is_classification: bool) -> None:
    """Render model performance metrics."""
    if is_classification:
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Value': [
                metrics['accuracy'],
                metrics['precision'],
                metrics['recall'],
                metrics['f1']
            ]
        })
    else:
        metrics_df = pd.DataFrame({
            'Metric': ['MSE', 'RMSE', 'MAE', 'R² Score'],
            'Value': [
                metrics['mse'],
                metrics['rmse'],
                metrics['mae'],
                metrics['r2']
            ]
        })
    st.dataframe(metrics_df, use_container_width=True)


@st.cache_data
def render_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    viz_library: str = "altair"
) -> None:
    """Render confusion matrix visualization."""
    if viz_library == "altair":
        chart = plot_confusion_matrix_alt(
            y_test,
            y_pred,
            labels=class_names
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        fig = plot_confusion_matrix(
            y_test,
            y_pred,
            class_names=class_names
        )
        st.pyplot(fig)


@st.cache_data
def render_roc_curve(
    y_test: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
    viz_library: str = "altair"
) -> None:
    """Render ROC curve visualization."""
    if viz_library == "altair":
        chart = plot_roc_curve_alt(
            y_test,
            y_prob,
            classes=class_names
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        fig = plot_roc_curve(
            y_test,
            y_prob,
            classes=class_names
        )
        st.pyplot(fig)


@st.cache_data
def render_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    viz_library: str = "altair"
) -> None:
    """Render feature importance visualization."""
    if viz_library == "altair":
        chart = plot_feature_importances_alt(
            feature_names,
            importances
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        fig = plot_feature_importances(
            importances,
            feature_names
        )
        st.pyplot(fig) 