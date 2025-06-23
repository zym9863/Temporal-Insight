# Temporal Insight (时序洞察)

[中文文档](README.md) | [English Version](README_EN.md)

📈 Intelligent Time Series Forecasting System

## Project Overview

Temporal Insight is a Python-based intelligent time series analysis and forecasting tool, offering a user-friendly web interface, supporting multiple forecasting models and interactive visualizations.

## Core Features

### 🤖 Model-Driven Intelligent Forecasting
- **Multiple Forecasting Models**: ARIMA, Linear Regression, Random Forest, Moving Average
- **Automatic Parameter Optimization**: Intelligently selects optimal model parameters
- **Model Evaluation**: Provides MSE, RMSE, MAE, MAPE and other evaluation metrics
- **Batch Forecasting**: Supports multi-step forecasting and different time frequencies

### 📊 Visualization of Forecast Results and Confidence Intervals
- **Interactive Charts**: Dynamic visualizations based on Plotly
- **Confidence Intervals**: Intuitively display forecast uncertainty
- **Historical Comparison**: Compare historical data with forecast results
- **Multi-Model Comparison**: Display forecast results of multiple models simultaneously

## Tech Stack

- **Backend**: Python 3.12+
- **Web Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, Statsmodels
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Package Management**: uv

## Project Structure

```
Temporal-Insight/
├── src/
│   └── temporal_insight/
│       ├── __init__.py
│       ├── data/                 # Data processing module
│       │   ├── __init__.py
│       │   └── processor.py
│       ├── models/               # Forecasting models module
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── predictors.py
│       │   └── engine.py
│       ├── visualization/        # Visualization module
│       │   ├── __init__.py
│       │   └── charts.py
│       └── utils/               # Utility module
│           ├── __init__.py
│           └── helpers.py
├── tests/                       # Test files
├── data/                        # Data files
├── main.py                      # Main application entry
├── pyproject.toml               # Project configuration
└── README.md
```

## Installation & Running

### Requirements
- Python 3.12+
- uv package manager

### Installation Steps

1. **Clone the project**
```bash
git clone https://github.com/zym9863/Temporal-Insight.git
cd Temporal-Insight
```

2. **Install dependencies**
```bash
uv sync
```

3. **Run the application**
```bash
uv run streamlit run main.py
```

4. **Access the app**
Open your browser and visit: http://localhost:8501

## User Guide

### 1. Data Upload
- Supports CSV and Excel files
- Data should include a date column and a value column
- Sample data available for download and testing

### 2. Data Processing
- Select date and value columns
- Configure missing value handling
- Optionally remove outliers

### 3. Model Selection
- Choose forecasting model (ARIMA, Linear Regression, Random Forest, Moving Average)
- Configure model parameters
- Train model and view evaluation metrics

### 4. Forecast Analysis
- Set forecast steps and time frequency
- Generate forecast results
- View visual charts and data tables
- Download forecast results

## Model Descriptions

### ARIMA Model
- Suitable for time series with trend and seasonality
- Automatically selects optimal parameters (p,d,q)
- Provides statistical confidence intervals
- Suitable for short- and medium-term forecasting

### Linear Regression Model
- Feature engineering based on sliding window
- Simple calculation, fast speed
- Suitable for linear trend data
- Highly interpretable

### Random Forest Model
- Ensemble learning method
- Captures nonlinear relationships
- Provides uncertainty estimation for predictions
- Robust to outliers

### Moving Average Model
- Simplest forecasting method
- Suitable for stationary time series
- Extremely fast computation
- Can be used as a baseline model

## Development

### Adding a New Model
1. In `src/temporal_insight/models/predictors.py`, inherit the `BasePredictor` class
2. Implement the `fit`, `predict`, `get_parameters`, and `set_parameters` methods
3. Register the new model in `engine.py`

### Custom Visualization
1. Add new chart methods in `src/temporal_insight/visualization/charts.py`
2. Call the new visualization feature in the main interface

## License

MIT License

## Contribution

Issues and Pull Requests are welcome!

## Contact

For questions or suggestions, please contact via:
- Submit a GitHub Issue
