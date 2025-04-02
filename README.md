# Interactive ML Model Trainer

A Streamlit application for interactive machine learning model training. This tool provides a user-friendly interface for:

- Selecting from built-in Seaborn datasets or uploading custom datasets
- Automatic identification of feature types (numeric vs. categorical)
- Interactive feature selection
- Target variable selection and correlation analysis
- (Coming soon) Model training, evaluation, and hyperparameter tuning

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the application with:
```
streamlit run app.py
```

## Features

- **Dataset Selection**: Choose from built-in Seaborn datasets or upload your own CSV file
- **Dataset Preview**: View dataset information and preview the first few rows
- **Feature Type Detection**: Automatic identification of numeric and categorical features
- **Feature Selection**: Select which features to use for model training
- **Target Variable Selection**: Choose the target variable for prediction
- **Session Persistence**: Your selections are saved between app sessions

## Coming Soon

- Model selection and training
- Hyperparameter tuning
- Model evaluation metrics
- Feature importance visualization
- Model export functionality 