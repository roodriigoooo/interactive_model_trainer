# Interactive ML Model Trainer

An interactive tool for training and evaluating machine learning models with a user-friendly interface.

## Features

- **Data Selection**: Choose from various sample datasets or upload your own.
- **Data Exploration**: Visualize data distributions, correlations, and statistics.
- **Feature Selection**: Choose relevant features for model training.
- **Model Training**: Train various ML models with customizable parameters.
- **Model Evaluation**: Evaluate model performance with various metrics and visualizations.
- **Model History**: Track and compare previous model runs to find the best model.
- **Interactive Visualizations**: Explore data and model results with interactive Altair charts.
- **Parameter Tuning**: Fine-tune model parameters using GridSearch or RandomizedSearch.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/interactive-model-trainer.git
cd interactive-model-trainer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
cd interactive_model_trainer
streamlit run trainerapp.py
```

2. Navigate through the tabs to:
   - Select and explore your dataset
   - Choose features and target variable
   - Select and train models
   - Evaluate and compare model performance
   - Save your models

## Model Types

### Classification Models
- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine
- K-Nearest Neighbors
- Decision Tree

### Regression Models
- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- Random Forest Regressor
- Gradient Boosting Regressor
- SVR

## Testing

Run the test suite with pytest:
```bash
pytest interactive_model_trainer/tests/
```

## Project Structure

- `trainerapp.py`: Main Streamlit application
- `models/`: Model-related functionality
  - `model_utils.py`: Model training and utility functions
  - `model_history.py`: Model history tracking
- `utils/`: Utility modules
  - `data_processor.py`: Data preprocessing utilities
  - `visualizer.py`: Matplotlib/Seaborn visualization utilities
  - `altair_visualizer.py`: Interactive Altair visualizations
- `tests/`: Unit tests

## License

This project is licensed under the MIT License - see the LICENSE file for details. 