# FinTech-Market-Analytics

A comprehensive Python toolkit for **financial market data analysis**, **machine learning model development**, and **trading strategy evaluation**. This project provides tools for automated analysis of stock market and cryptocurrency data, including data preprocessing, feature engineering, model training, and performance evaluation.

## Core Components

1. **Data Processing Pipeline**
   - **Data Collection**  
     - S&P 500 stocks historical data  
     - Major cryptocurrencies (BTC, ETH, SOL, XRP)  
     - Automatic handling of real-time updates  
   - **Data Preprocessing**  
     - Missing value handling  
     - Outlier detection and removal  
     - Data normalization and cleaning  
     - Automated quality checks  
   - **Feature Engineering**  
     - Price-based features (Moving averages, price channels)  
     - Volume-based features (Volume profiles, price-volume correlations)  
     - Volatility indicators (Rolling volatility, volatility regimes)  
     - Technical indicators (RSI, MACD, Bollinger Bands, etc.) via TA‑Lib  
     - Machine learning–specific features

2. **Machine Learning Models**
   - **Classical ML Models**  
     - Random Forest Strategy  
     - Gradient Boosting Strategy  
     - Regularized Logistic Strategy  
   - **Deep Learning Models**  
     - LSTM/GRU for time-series  
     - Transformer-based approaches  
     - Pretrained neural networks for advanced sequence analysis  
   - **Time Series Models**
     - SARIMAX for price prediction  
     - Volatility forecasting  
     - Trend analysis  

3. **Trading Strategies**
   - **Technical Analysis Based**  
     - Moving Average Crossover  
     - RSI Strategy  
     - MACD Strategy  
     - Bollinger Bands Strategy  
   - **ML-Based Strategies**  
     - Automated signal generation  
     - Risk management  
     - Portfolio optimization  
   - **Neural Network Strategies**  
     - LSTM-based classification/regression for trading actions  
     - Custom or pretrained architectures

4. **Backtesting & Evaluation**
   - **Data Split Management**  
     - Training set (60%)  
     - Testing set (20%)  
     - Validation set (20%)  
   - **Performance Metrics**  
     - Accuracy, Precision, Recall, F1  
     - Returns and Sharpe ratio  
     - Maximum drawdown  
     - Trading costs consideration  

5. **Visualization & Monitoring**
   - **Dashboards**  
     - `strategy_analysis.py` or `nn_dashboard.py` for interactive charts  
     - Plot signals, portfolio values, distribution of returns, drawdown  
   - **Experiment Tracker**  
     - Saves metrics for each run (train, validation, test)  
     - JSON-based or advanced frameworks (e.g., MLflow)

---

## Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/yourusername/FinTech-Market-Analytics.git
   cd FinTech-Market-Analytics
   ```

2. **Create and Activate a Virtual Environment**  
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Requirements**  
   ```bash
   pip install -r requirements.txt
   ```
   - Includes libraries like **pandas**, **numpy**, **scikit-learn**, **tensorflow/keras** (or PyTorch), **TA-Lib** (note: TA-Lib may require a special install on Windows), **backoff**, etc.

---

## Usage Examples

### 1. Data Processing & Feature Engineering

```python
from market_analyzer import MarketDataAnalyzer
from market_analyzer.preprocessor import DataPreprocessor

analyzer = MarketDataAnalyzer()
analyzer.download_data(period="1y")  # e.g. 1 year
df = analyzer.get_asset_data('BTC-USD')

preprocessor = DataPreprocessor()
cleaned_df = preprocessor.clean_data(df)
features = preprocessor.engineer_features(cleaned_df)
# Now 'features' contains advanced indicators for ML or strategy logic
```

### 2. Training a Neural Network Strategy

1. **`nn_model_training.py`**  
   - Loads data  
   - Creates a **NeuralNetworkStrategy**  
   - Trains & evaluates it with a **Backtester** on a validation set  
   - Logs metrics via **ExperimentTracker**  

Typical run:
```bash
python nn_model_training.py
```
This script prints training accuracy, validation metrics, backtest results, and possibly stores them in `results/experiment_results.json`.

### 3. Viewing Dashboard Results

After training, run:
```bash
python nn_dashboard.py
```
This script:
- Loads or constructs dictionaries of final results (e.g., `portfolio_value`, `annual_return`, etc.).  
- Passes them to `StrategyDashboard` for plotting **portfolio value**, **returns distribution**, **performance metrics** (Sharpe, drawdown, etc.), and **drawdown** charts.  

If your data is non-empty and properly aligned, you’ll see interactive charts showing how the **Neural Network Strategy** performed.

---

## **Typical Workflow**

Below is the **usual** sequence of commands and scripts when using a **Neural Network–based** trading strategy in this project:

1. **`python nn_model_training.py`**  
   - Downloads or loads historical market data.  
   - Creates **train/val** splits.  
   - Builds `(X_train,y_train)` and `(X_val,y_val)`.  
   - **Trains** the neural network model (e.g., LSTM) in `NeuralNetworkStrategy`.  
   - **Backtests** on the validation set via `Backtester`, computing metrics like `annual_return`, `sharpe_ratio`, etc.  
   - Saves or logs these results (e.g., to `experiment_results.json`).

2. **`python nn_dashboard.py`**  
   - Loads the final or partial results (from JSON, in-memory, or a custom method).  
   - Creates a `StrategyDashboard()` instance.  
   - Calls methods like `plot_portfolio_values(...)`, `plot_returns_distribution(...)`, `plot_performance_metrics(...)`, and `plot_drawdown(...)`.  
   - Displays interactive or static charts showing how the neural network strategy performed (e.g. line plots of **portfolio_value** over time, bar charts of **annual_return** or **sharpe_ratio**, etc.).

3. (Optional) **Compare** multiple strategies:
   - Run classical strategies (e.g., MACD, Moving Average Cross) or older ML approaches.  
   - Gather each strategy’s dictionary (with `'annual_return','volatility','portfolio_value'`, etc.).  
   - Pass them in a list to the same dashboard methods for side-by-side comparison.

---

## Project Structure

```
FinTech-Market-Analytics/
├── src/
│   └── market_analyzer/
│       ├── __init__.py
│       ├── analyzer.py          # MarketDataAnalyzer for data collection
│       ├── preprocessor.py      # Data cleaning & feature creation
│       ├── feature_engineering.py
│       ├── ml_models.py         # Non-neural ML models
│       ├── nn_strategy.py       # NeuralNetworkStrategy (LSTM, etc.)
│       ├── backtester.py        # Backtesting engine for strategies
│       ├── experiment_tracker.py # Logging experiments
│       ├── dashboard.py         # StrategyDashboard for plotting
│       └── utils.py             # Utility functions (validate_data, etc.)
├── nn_model_training.py         # Script to train & test the NN strategy
├── nn_dashboard.py              # Script to display NN strategy results
├── data/                        # Data storage
├── models/                      # Saved models
├── results/                     # Experiment results
└── tests/                       # Unit tests
```

---

## Contributing

1. Fork the repository  
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)  
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)  
4. Push to the branch (`git push origin feature/AmazingFeature`)  
5. Open a Pull Request  

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.