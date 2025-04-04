"""
Interactive visualization utilities using Altair. to be chosen by the user.
"""
import altair as alt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Set Altair rendering options
alt.data_transformers.disable_max_rows()

def plot_feature_importances_alt(feature_names, importances, title="Feature Importance"):
    """
    interactive feature importance visualization using Altair.
    """
    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=True)
    
    # Create the chart
    chart = alt.Chart(importance_df).mark_bar().encode(
        y=alt.Y('Feature:N', 
                sort=None,  # Preserve the sorted order
                axis=alt.Axis(
                    labelLimit=200,  # Increase label length limit
                    labelAngle=0,    # Keep labels horizontal
                )),
        x=alt.X('Importance:Q',
                axis=alt.Axis(title='Importance')),
        tooltip=['Feature', 'Importance']
    ).properties(
        title=title,
        width=600,
        height=max(300, len(feature_names) * 25)  # Dynamic height based on number of features
    ).configure_axis(
        labelFontSize=11,
        titleFontSize=13
    ).configure_title(
        fontSize=14,
        anchor='middle'
    )
    
    return chart

def plot_correlation_matrix_alt(df, method='pearson', title="Feature Correlation Matrix"):
    """
    interactive correlation matrix visualization using Altair.
    """
    # Calculate correlation matrix
    corr = df.corr(method=method)
    
    # Convert to long format for Altair
    corr_long = corr.reset_index().melt(
        id_vars='index',
        var_name='variable',
        value_name='correlation'
    )
    
    # Create chart
    chart = alt.Chart(corr_long).mark_rect().encode(
        x=alt.X('variable:N', title=None),
        y=alt.Y('index:N', title=None),
        color=alt.Color(
            'correlation:Q',
            scale=alt.Scale(domain=[-1, 1], scheme='blueorange'),
            legend=alt.Legend(title='Correlation')
        ),
        tooltip=['index', 'variable', alt.Tooltip('correlation:Q', format='.2f')]
    ).properties(
        title=title,
        width=600,
        height=600
    ).configure_axis(
        labelAngle=0
    ).interactive()
    
    return chart

def plot_scatter_matrix_alt(df, features, color_by=None):
    """
    interactive scatter plot matrix visualization.
    """
    # Limit to a reasonable number of features
    if len(features) > 5:
        features = features[:5]
        
    # Create base chart
    chart = alt.Chart(df).mark_circle(size=60, opacity=0.6).encode(
        tooltip=[alt.Tooltip(f) for f in features]
    )
    
    # Create scatter matrix
    if color_by:
        matrix = alt.Chart(df).mark_circle(size=60, opacity=0.6).encode(
            alt.X(alt.repeat("column"), type='quantitative'),
            alt.Y(alt.repeat("row"), type='quantitative'),
            color=alt.Color(color_by)
        ).properties(
            width=150,
            height=150
        ).repeat(
            row=features,
            column=features
        ).interactive()
    else:
        matrix = alt.Chart(df).mark_circle(size=60, opacity=0.6).encode(
            alt.X(alt.repeat("column"), type='quantitative'),
            alt.Y(alt.repeat("row"), type='quantitative')
        ).properties(
            width=150,
            height=150
        ).repeat(
            row=features,
            column=features
        ).interactive()
    
    return matrix

def plot_distplot_alt(data, column, title=None):
    """
    interactive distribution plot (histogram + density).

    """
    # Calculate histogram values
    hist_data = pd.DataFrame({
        'value': data[column]
    })
    
    # Create chart
    base = alt.Chart(hist_data).encode(
        x=alt.X('value:Q', title=column, bin=alt.Bin(maxbins=30)),
    )
    
    histogram = base.mark_bar().encode(
        y=alt.Y('count()', title='Frequency')
    )
    
    # Calculate the KDE for a smooth density
    kde = alt.Chart(hist_data).transform_density(
        'value',
        as_=['value', 'density'],
    ).mark_line(color='red').encode(
        x='value:Q',
        y='density:Q',
    )
    
    # Combine the plots
    chart = (histogram + kde).properties(
        title=title or f'Distribution of {column}',
        width=600,
        height=400
    ).interactive()
    
    return chart

def plot_boxplot_alt(data, column, by=None, title=None):
    """
    interactive boxplot 
    """
    if by:
        chart = alt.Chart(data).mark_boxplot().encode(
            x=alt.X(f'{by}:N', title=by),
            y=alt.Y(f'{column}:Q', title=column),
            color=alt.Color(f'{by}:N', legend=None)
        ).properties(
            title=title or f'Boxplot of {column} by {by}',
            width=600,
            height=400
        ).interactive()
    else:
        chart = alt.Chart(data).mark_boxplot().encode(
            y=alt.Y(f'{column}:Q', title=column)
        ).properties(
            title=title or f'Boxplot of {column}',
            width=600,
            height=400
        ).interactive()
    
    return chart

def plot_confusion_matrix_alt(y_true, y_pred, labels=None, title="Confusion Matrix"):
    """
    interactive confusion matrix visualization 
    
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Convert to dataframe
    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])]
    
    cm_df = pd.DataFrame(cm, columns=labels, index=labels)
    cm_df.index.name = 'Actual'
    cm_df.columns.name = 'Predicted'
    
    # Convert to long format for Altair
    cm_long = cm_df.reset_index().melt(
        id_vars='Actual', 
        var_name='Predicted',
        value_name='Count'
    )
    
    # Create chart
    chart = alt.Chart(cm_long).mark_rect().encode(
        x=alt.X('Predicted:N', title='Predicted'),
        y=alt.Y('Actual:N', title='Actual', sort=None),
        color=alt.Color(
            'Count:Q',
            scale=alt.Scale(scheme='blues'),
            legend=alt.Legend(title='Count')
        ),
        tooltip=['Actual', 'Predicted', 'Count']
    ).properties(
        title=title,
        width=400,
        height=400
    )
    
    # Add value text
    text = chart.mark_text(baseline='middle').encode(
        text='Count:Q',
        color=alt.condition(
            alt.datum.Count > cm.max() / 2,
            alt.value('white'),
            alt.value('black')
        )
    )
    
    return (chart + text).interactive()

def plot_roc_curve_alt(y_true, y_prob, classes=None, title="ROC Curve"):
    """
    interactive ROC curve visualization.
    """
    # Binary classification case
    if y_prob.shape[1] == 2:
        # Use probability of positive class
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        
        # Create dataframe for visualization
        roc_df = pd.DataFrame({
            'False Positive Rate': fpr,
            'True Positive Rate': tpr,
            'Class': 'ROC Curve (AUC = {:.3f})'.format(roc_auc)
        })
        
        # Create chart
        chart = alt.Chart(roc_df).mark_line().encode(
            x=alt.X('False Positive Rate', title='False Positive Rate'),
            y=alt.Y('True Positive Rate', title='True Positive Rate'),
            color=alt.Color('Class', legend=alt.Legend(title=''))
        )
        
        # Add diagonal reference line
        diag_df = pd.DataFrame({
            'x': [0, 1],
            'y': [0, 1]
        })
        diag_line = alt.Chart(diag_df).mark_line(
            strokeDash=[6, 4],
            color='gray'
        ).encode(
            x='x',
            y='y'
        )
        
        # Combine
        return (chart + diag_line).properties(
            title=title,
            width=500,
            height=500
        ).interactive()
    
    # Multiclass case
    else:
        n_classes = y_prob.shape[1]
        class_names = classes if classes else [f'Class {i}' for i in range(n_classes)]
        
        # Calculate ROC curve for each class
        roc_dfs = []
        for i in range(n_classes):
            # One-vs-rest ROC
            y_true_bin = (y_true == i).astype(int)
            y_score = y_prob[:, i]
            
            fpr, tpr, _ = roc_curve(y_true_bin, y_score)
            roc_auc = auc(fpr, tpr)
            
            # Create dataframe
            roc_df = pd.DataFrame({
                'False Positive Rate': fpr,
                'True Positive Rate': tpr,
                'Class': f'{class_names[i]} (AUC = {roc_auc:.3f})'
            })
            
            roc_dfs.append(roc_df)
        
        # Combine all classes
        combined_df = pd.concat(roc_dfs)
        
        # Create chart
        chart = alt.Chart(combined_df).mark_line().encode(
            x=alt.X('False Positive Rate', title='False Positive Rate'),
            y=alt.Y('True Positive Rate', title='True Positive Rate'),
            color=alt.Color('Class', legend=alt.Legend(title=''))
        )
        
        # Add diagonal reference line
        diag_df = pd.DataFrame({
            'x': [0, 1],
            'y': [0, 1]
        })
        diag_line = alt.Chart(diag_df).mark_line(
            strokeDash=[6, 4],
            color='gray'
        ).encode(
            x='x',
            y='y'
        )
        
        # Combine
        return (chart + diag_line).properties(
            title=title,
            width=500,
            height=500
        ).interactive()

def plot_residuals_alt(y_true, y_pred, title="Residual Plot"):
    """
    interactive residual plot visualization.
    """
    # Calculate residuals
    residuals = y_true - y_pred
    
    # Create dataframe for visualization
    residual_df = pd.DataFrame({
        'Predicted': y_pred,
        'Residual': residuals
    })
    
    # Zero line
    zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(
        strokeDash=[6, 4],
        color='red'
    ).encode(y='y')
    
    # Create scatter plot
    scatter = alt.Chart(residual_df).mark_circle(size=60, opacity=0.6).encode(
        x=alt.X('Predicted', title='Predicted Values'),
        y=alt.Y('Residual', title='Residuals'),
        tooltip=['Predicted', 'Residual']
    )
    
    # Combine
    return (scatter + zero_line).properties(
        title=title,
        width=600,
        height=400
    ).interactive()

def plot_model_comparison_alt(history_df, metric_column, title="Model Comparison"):
    """
    interactive model comparison visualization.
    """
    # Sort by metric
    sorted_df = history_df.sort_values(metric_column)
    
    # Create chart
    chart = alt.Chart(sorted_df).mark_bar().encode(
        x=alt.X(metric_column, title=metric_column.replace('metric_', '')),
        y=alt.Y('model_name', title='Model', sort=None),
        color=alt.Color('model_name', legend=None),
        tooltip=['model_name', metric_column, 'dataset_name', 'timestamp']
    ).properties(
        title=title,
        width=600,
        height=400
    ).interactive()
    
    return chart

def plot_parameter_comparison_alt(results_df, param_name, metric_name, title=None):
    """
    interactive parameter comparison visualization.
    """
    # Create chart
    chart = alt.Chart(results_df).mark_line(point=True).encode(
        x=alt.X(param_name, title=param_name),
        y=alt.Y(metric_name, title=metric_name),
        tooltip=[param_name, metric_name]
    ).properties(
        title=title or f'Effect of {param_name} on {metric_name}',
        width=600,
        height=400
    ).interactive()
    
    return chart

def plot_categorical_distributions_alt(df, features, target=None, title=None):
    """
    interactive categorical distribution visualization.
    """
    charts = []
    
    for feature in features:
        # Get value counts for plotting
        if target is None:
            # Simple count
            value_counts = df[feature].value_counts().reset_index()
            value_counts.columns = [feature, 'count']
            
            # Create the chart
            chart = alt.Chart(value_counts).mark_bar().encode(
                x=alt.X(f'{feature}:N', title=feature),
                y=alt.Y('count:Q', title='Count'),
                tooltip=[feature, 'count']
            ).properties(
                title=title or f'Distribution of {feature}',
                width=600,
                height=400
            ).interactive()
            
        else:
            # Group by target
            # Convert to long format
            chart_data = df[[feature, target]].copy()
            
            # Create the chart with color by target
            chart = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X(f'{feature}:N', title=feature),
                y=alt.Y('count()', title='Count'),
                color=alt.Color(f'{target}:N', title=target),
                tooltip=[feature, target, alt.Tooltip('count()', title='Count')]
            ).properties(
                title=title or f'Distribution of {feature} by {target}',
                width=600,
                height=400
            ).interactive()
        
        charts.append(chart)
    
    return charts 