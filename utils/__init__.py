"""
Utilities package for the ML Model Trainer application.
"""
from utils.data_processor import DataProcessor
from utils.visualizer import (
    plot_feature_importances,
    plot_correlation_matrix,
    plot_categorical_distributions,
    plot_numeric_distributions,
    plot_confusion_matrix,
    plot_roc_curve
)
from utils.altair_visualizer import (
    plot_feature_importances_alt,
    plot_correlation_matrix_alt,
    plot_scatter_matrix_alt,
    plot_distplot_alt,
    plot_boxplot_alt,
    plot_confusion_matrix_alt,
    plot_roc_curve_alt,
    plot_residuals_alt,
    plot_model_comparison_alt,
    plot_parameter_comparison_alt,
    plot_categorical_distributions_alt
)

__all__ = [
    'DataProcessor',
    # Matplotlib/Seaborn visualizers
    'plot_feature_importances',
    'plot_correlation_matrix',
    'plot_categorical_distributions',
    'plot_numeric_distributions',
    'plot_confusion_matrix',
    'plot_roc_curve',
    # Altair visualizers
    'plot_feature_importances_alt',
    'plot_correlation_matrix_alt',
    'plot_scatter_matrix_alt',
    'plot_distplot_alt',
    'plot_boxplot_alt',
    'plot_confusion_matrix_alt',
    'plot_roc_curve_alt',
    'plot_residuals_alt',
    'plot_model_comparison_alt',
    'plot_parameter_comparison_alt',
    'plot_categorical_distributions_alt'
] 