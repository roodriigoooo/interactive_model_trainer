"""
Visualization utilities for the ML Model Trainer application.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    precision_recall_curve,
    auc
)

def plot_feature_importances(feature_importances, feature_names, top_n=10):
    """
    Plot feature importances.
    
    Parameters:
    -----------
    feature_importances : numpy.ndarray
        Array of feature importances.
    feature_names : list
        List of feature names.
    top_n : int, default=10
        Number of top features to display.
    """
    # Sort features by importance
    indices = np.argsort(feature_importances)[-top_n:]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create horizontal bar plot
    ax.barh(range(len(indices)), feature_importances[indices])
    
    # Set y-axis labels to feature names
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    
    # Add labels and title
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Top {top_n} Feature Importances')
    
    return fig

def plot_correlation_matrix(df, numeric_features=None):
    """
    Plot correlation matrix for numeric features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe containing the features.
    numeric_features : list, optional
        List of numeric feature names. If None, all numeric features are used.
    """
    # If numeric_features is not provided, use all numeric columns
    if numeric_features is None:
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_features].corr()
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Draw heatmap
    sns.heatmap(
        corr_matrix, 
        mask=mask, 
        cmap=cmap, 
        vmax=1, 
        vmin=-1, 
        center=0,
        square=True, 
        linewidths=.5, 
        cbar_kws={"shrink": .5},
        annot=True,
        fmt=".2f"
    )
    
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    
    return fig

def plot_categorical_distributions(df, categorical_features, target=None):
    """
    Plot the distribution of categorical features, optionally grouped by target.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe containing the features.
    categorical_features : list
        List of categorical feature names.
    target : str, optional
        Name of the target variable for grouping.
    """
    # Calculate number of plots needed
    n_features = len(categorical_features)
    n_cols = 2
    n_rows = (n_features + 1) // n_cols
    
    # Create subplot grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 4))
    axes = axes.flatten()
    
    # Plot each categorical feature
    for i, feature in enumerate(categorical_features):
        if i < len(axes):  # Ensure we don't exceed the number of subplots
            if target is not None and target in df.columns:
                # Group by target
                ax = sns.countplot(x=feature, hue=target, data=df, ax=axes[i])
                # Rotate x-axis labels if needed
                if df[feature].nunique() > 5:
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            else:
                # Simple count plot
                ax = sns.countplot(x=feature, data=df, ax=axes[i])
                # Rotate x-axis labels if needed
                if df[feature].nunique() > 5:
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            axes[i].set_title(f'Distribution of {feature}')
            axes[i].set_xlabel('')
    
    # Hide any unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_numeric_distributions(df, numeric_features, target=None, bins=20):
    """
    Plot the distribution of numeric features, optionally grouped by target.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe containing the features.
    numeric_features : list
        List of numeric feature names.
    target : str, optional
        Name of the target variable for grouping.
    bins : int, default=20
        Number of bins for histograms.
    """
    # Calculate number of plots needed
    n_features = len(numeric_features)
    n_cols = 2
    n_rows = (n_features + 1) // n_cols
    
    # Create subplot grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 4))
    axes = axes.flatten()
    
    # Plot each numeric feature
    for i, feature in enumerate(numeric_features):
        if i < len(axes):  # Ensure we don't exceed the number of subplots
            if target is not None and target in df.columns and df[target].nunique() <= 5:
                # For categorical targets with few unique values, use KDE plots
                for target_val in df[target].unique():
                    subset = df[df[target] == target_val]
                    sns.kdeplot(x=subset[feature], ax=axes[i], label=f'{target}={target_val}')
                axes[i].legend()
            else:
                # Simple histogram
                sns.histplot(x=feature, data=df, bins=bins, kde=True, ax=axes[i])
            
            axes[i].set_title(f'Distribution of {feature}')
            axes[i].set_xlabel('')
    
    # Hide any unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Plot confusion matrix for classification models.
    
    Parameters:
    -----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    class_names : list, optional
        List of class names.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names if class_names else "auto",
        yticklabels=class_names if class_names else "auto"
    )
    
    # Add labels and title
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    
    return fig

def plot_roc_curve(y_true, y_prob, classes=None):
    """
    Plot ROC curve for binary or multiclass classification.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels.
    y_prob : array-like
        Predicted probabilities.
    classes : list, optional
        List of class names for multiclass problems.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # For binary classification
    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        
    # For multiclass classification
    else:
        # One-vs-Rest ROC curves for each class
        for i, cls in enumerate(np.unique(y_true)):
            # Convert to one-vs-rest
            y_true_bin = (y_true == cls).astype(int)
            
            # Get class probabilities
            y_prob_cls = y_prob[:, i]
            
            fpr, tpr, _ = roc_curve(y_true_bin, y_prob_cls)
            roc_auc = auc(fpr, tpr)
            
            cls_name = classes[i] if classes else f'Class {cls}'
            ax.plot(fpr, tpr, lw=2, label=f'{cls_name} (area = {roc_auc:.2f})')
    
    # Add labels and legend
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right')
    
    return fig 