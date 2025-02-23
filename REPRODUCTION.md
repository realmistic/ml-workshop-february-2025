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

## Production Implementation

### 1. Database Setup
Sequence:
```
Create SQLite database structure with:
- Raw market data table
- Feature tables for each model
- Predictions tables
- Performance metrics table

Output: scripts/init_db.py for database initialization
```

Output: Database schema implementation with proper foreign key relationships and timestamps.

### 2. Data Pipeline
Sequence:

1. Data Collection:
```
Implement market data updates:
- Download from Stooq (after YFinance rate limiting)
- Calculate technical indicators
- Store in SQLite database
Output: scripts/update_data.py
```

2. Feature Engineering:
```
For each model:
- ARIMA: returns, volatility, moving averages
- Prophet: formatted closing prices
- DNN: extended feature set with RSI
Implementation in update_data.py
```

### 3. Model Implementation
Sequence:

1. ARIMA Model:
```
Convert notebook to production:
- Class-based implementation
- Training pipeline
- Prediction generation
- Performance tracking
Output: scripts/models/arima.py
```

2. Prophet Model:
```
Production implementation with:
- Automated training
- Confidence intervals
- Performance metrics
Output: scripts/models/prophet.py
```

3. DNN Model:
```
Production-ready implementation:
- Sequence generation
- Model architecture
- Training process
- Prediction pipeline
Output: scripts/models/dnn.py
```

### 4. Training Pipeline
```
Create unified training script:
- Load and prepare data
- Train all models
- Generate predictions
- Update metrics
Output: scripts/train_models.py
```

### 5. Streamlit Dashboard
```
Create interactive dashboard:
- Combined predictions view
- Individual model metrics
- Performance comparison
Output: app/main.py
```

### 6. Automation
```
Implement GitHub Actions:
- Daily data updates
- Model retraining
- Database commits
- Streamlit deployment
Output: .github/workflows/daily_update.yml
```

## Running the Project

1. Setup:
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. Initialize:
```bash
python scripts/init_db.py
python scripts/update_data.py
python scripts/train_models.py
```

3. Run Dashboard:
```bash
streamlit run app/main.py
```

## Troubleshooting

- Data Source: Using Stooq after YFinance rate limiting issues
- Database: Check table relationships and constraints
- Models: Monitor training logs and metrics
- Dashboard: Verify data pipeline completion
