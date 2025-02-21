# ML Workshop February 2025 - Reproduction Guide

This document provides the sequence of prompts and outputs used to develop this project with AI assistance. For project overview and features, see [README.md](README.md).

## Development Sequence

### 1. Manual Start Notebook
Prompt:
```
Create a notebook to download QQQ data using yfinance, implement basic dynamic visualization with plotly express, define train/validation/test splits, and set up returns-based prediction framework.

Output should be notebooks/manual_start.ipynb with data loading, visualization, and split definition.
```

Output: `notebooks/manual_start.ipynb`
- Data download with yfinance
- Interactive plotly visualization
- Train/val/test split implementation
- Returns calculation setup

### 2. Prophet Model Development
Sequence of prompts:

1. Basic Prophet Setup:
```
Create a Prophet experiment notebook that loads the data similarly to manual_start.ipynb, but formats it for Prophet (ds/y columns) and implements basic prediction.

Output should be notebooks/prophet_experiment.ipynb with initial Prophet implementation.
```

2. Model Improvement:
```
Enhance the Prophet model by:
- Testing different seasonality settings
- Adding holiday effects
- Optimizing hyperparameters
Update prophet_experiment.ipynb with improved model.
```

3. Visualization Development:
```
Add comprehensive visualization including:
- Predictions with confidence intervals
- Residuals analysis
- Component plots
Also implement trading metrics (win rate, returns, P/L ratio).
```

Output: Enhanced `notebooks/prophet_experiment.ipynb`

### 3. ARIMA Model Development
Sequence:

1. Initial Setup:
```
Create ARIMA experiment notebook with same data loading and transformation as Prophet, but prepared for ARIMA modeling.

Output should be notebooks/arima_experiment.ipynb with data preparation.
```

2. Model Implementation:
```
Implement ARIMA(2,1,2) model with:
- Basic feature engineering
- Parameter estimation
- Prediction generation
Use same visualization and metrics as Prophet for comparison.
```

Output: Complete `notebooks/arima_experiment.ipynb`

### 4. Deep Neural Network Development
Sequence:

1. Initial Setup:
```
Create DNN experiment notebook with enhanced feature engineering:
- Multiple timeframe returns
- Technical indicators
- Proper scaling
Output should be notebooks/dnn_experiment.ipynb with data preparation.
```

2. Model Architecture:
```
Implement and test various architectures, starting complex and simplifying to:
- Two layers (8->8)
- Batch normalization
- Dropout
- Returns prediction
Use same evaluation framework as other models.
```

3. Model Refinement:
```
Optimize the simple architecture:
- Learning rate tuning
- Training process improvement
- Performance evaluation
```

Output: Final `notebooks/dnn_experiment.ipynb`

## Key Points

- Each notebook follows similar structure:
  * Data preparation
  * Model implementation
  * Visualization
  * Performance metrics

- Common evaluation framework across all models:
  * Win/Loss rates
  * Market outperformance
  * Risk-adjusted returns

- Focus on simplicity and interpretability:
  * Started complex, ended simpler
  * Consistent visualization
  * Comparable metrics
