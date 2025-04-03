# FinTech-Market-Analytics

A comprehensive Python toolkit for **financial market data analysis**, **machine learning model development**, **ensemble modeling**, and **trading strategy evaluation**. This project provides tools for automated analysis of stock market and cryptocurrency data, including data preprocessing, feature engineering, model training, performance evaluation, and strategy dashboards.

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
     - Technical indicators (RSI, MACD, Bollinger Bands, etc.) via **TA‑Lib**
     - Machine learning–specific features

2. **Machine Learning / Deep Learning Models**
   - **Classical ML** (Random Forest, Gradient Boosting, Logistic Regression)
   - **Neural Networks** (LSTM/GRU, Transformers, custom architectures)
   - **LLM-based Sentiment** (Hugging Face Transformers for news sentiment)
   - **Reinforcement Learning** (e.g. Stable-Baselines3 for agent training)
   - **Model Management**
     - Experiment tracking
     - Model versioning
     - Performance monitoring
     - Hyperparameter optimization

3. **Ensemble Strategies**
   - **Heterogeneous Ensemble**:
     - **Time-Series NN** (LSTM or Transformer)
     - **LLM-based Sentiment** (news sentiment → signals)
     - **RL-based Trading Agent** (policy gradient or PPO)
   - **Fusion Methods**:
     - Voting (majority or weighted)
     - Meta-learning (train a meta-model on sub-model outputs)

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
   - **Dashboards**:
     - Plot signals and portfolio values
     - Display returns distribution
     - Show performance metrics (annual return, Sharpe ratio)
     - Compare multiple strategies (ensemble vs. sub-models)
   - **Experiment Tracker**:
     - JSON-based or advanced frameworks (e.g., MLflow)
     - Logs each experiment’s metrics and parameters

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
   - This typically includes **pandas**, **numpy**, **scikit-learn**, **ta-lib** (note: TA-Lib may require special install steps on Windows), **matplotlib**, **seaborn**, **tensorflow** or **pytorch**, etc.

   - To “Complete” the market_analyzer Module:
     Ensure `__init__.py` is present in `src/market_analyzer/`.
     If you have a `setup.py`, you can install in editable mode:
     ```bash
     pip install -e .
     ```

4. **Install Additional Packages** for the Ensemble:
   - **Stable-Baselines3** (for Reinforcement Learning):
     ```bash
     pip install stable-baselines3
     ```
   - **Transformers** (for LLM-based sentiment):
     ```bash
     pip install transformers
     ```
   - **Optional**: 
     ```bash
     pip install backoff  # or any other library you need
     ```

---

## Usage Examples

### 1. Data Processing & Feature Engineering

```python
from market_analyzer.analyzer import MarketDataAnalyzer
from market_analyzer.preprocessor import DataPreprocessor

analyzer = MarketDataAnalyzer()
analyzer.download_data(period="1y")
df = analyzer.get_asset_data('BTC-USD')

preprocessor = DataPreprocessor()
cleaned_df = preprocessor.clean_data(df)
features = preprocessor.engineer_features(cleaned_df)
```

### 2. Training & Evaluating the Ensemble

1. **`ensemble_model_training.py`** (Example)
   - Loads data (`MarketDataAnalyzer`)
   - Builds/Trains sub-models:
     - **TimeSeriesNN** (LSTM/Transformer)
     - **LLM-based Sentiment** (Hugging Face Transformers pipeline)
     - **RLAgent** (Stable-Baselines3)
   - Combines them in an `EnsembleStrategy` (voting or meta-learning)
   - **Backtests** on validation set (via `Backtester`)
   - Logs metrics (annual return, Sharpe ratio, etc.) using `ExperimentTracker`

Typical run:
```bash
python ensemble_model_training.py
```
That will print training logs, backtest results, and store final metrics in e.g. `results/experiment_results.json`.

### 3. Viewing Dashboard Results

After training/evaluating the ensemble, run:
```bash
python ensemble_dashboard.py
```
It will:
- Load or construct sub-model + ensemble results
- Pass them to `StrategyDashboard` for line charts (portfolio value), distribution plots, bar charts (performance metrics), drawdown graphs, etc.

---

## **Typical Workflow**

1. **`python ensemble_model_training.py`**  
   - Downloads or loads market data  
   - Creates sub-model strategies (TimeSeriesNN, LLM-based sentiment, RL agent)  
   - Optionally trains them  
   - Creates **EnsembleStrategy**  
   - Evaluates on validation set using **Backtester** → logs `'annual_return'`, `'sharpe_ratio'`, etc.  
   - Saves results in `ExperimentTracker` or prints them.

2. **`python ensemble_dashboard.py`**  
   - Loads the final ensemble (and sub-model) results from disk or memory  
   - Uses `StrategyDashboard` to plot each strategy’s portfolio vs. the ensemble  
   - Displays interactive charts for performance metrics, distribution of returns, drawdown, etc.

3. (Optional) **Compare** multiple ensemble configurations:
   - Weighted vs. majority vote, or different meta-models  
   - RL agent variations, different LLM sentiment models  
   - Log each experiment’s results in `experiment_results.json`  
   - Visualize side-by-side in the same dashboard.

---

## Docker & Scheduling (Hourly Updates + Email)

### 1) Docker & Cron

To run the project **hourly** (e.g. to update crypto data and generate signals), you can:

1. **Create a Dockerfile** installing `cron` and copying a `crontab`:
   ```dockerfile
   FROM python:3.11-slim

   # Install cron
   RUN apt-get update && apt-get install -y cron

   WORKDIR /app
   COPY . .  # includes your code, crontab, requirements.txt, etc.

   RUN pip install --no-cache-dir -r requirements.txt

   # Copy and install crontab
   COPY crontab /etc/cron.d/cronjob
   RUN chmod 0644 /etc/cron.d/cronjob
   RUN crontab /etc/cron.d/cronjob

   CMD ["cron", "-f"]
   ```
2. **`crontab`** (every hour):
   ```bash
   # crontab
   0 * * * * python /app/hourly_update_and_predict.py >> /var/log/cron.log 2>&1
   ```
3. **Email Notifications**  
   - In `hourly_update_and_predict.py`, read your Gmail (or other SMTP) credentials from environment variables:
     ```python
     import os
     SMTP_USER = os.environ["SMTP_USER"]   # e.g. "youname@gmail.com"
     SMTP_PASS = os.environ["SMTP_PASS"]
     # ...
     ```
   - Use `smtplib` to send signals: SELL/BUY/HOLD.

Then build and run:
```bash
docker build -t crypto-cron .
docker run -e SMTP_USER="yourname@gmail.com" -e SMTP_PASS="myappassword" crypto-cron
```
At minute 0 every hour, `hourly_update_and_predict.py` is run, new signals are computed, and you get an email.

---

## Running on Azure

### 1) Push Your Docker Image

1. **Login** to Azure / Container Registry. For example:
   ```bash
   az login
   az acr create --resource-group MyResourceGroup --name MyRegistryName --sku Basic
   az acr login --name MyRegistryName
   ```
2. **Tag** and **Push**:
   ```bash
   docker tag crypto-cron MyRegistryName.azurecr.io/crypto-cron:latest
   docker push MyRegistryName.azurecr.io/crypto-cron:latest
   ```

### 2) Azure Container Instances

Now create an **Azure Container Instance** that runs continuously:

```bash
az container create \
  --resource-group MyResourceGroup \
  --name my-crypto-cron \
  --image MyRegistryName.azurecr.io/crypto-cron:latest \
  --registry-login-server MyRegistryName.azurecr.io \
  --registry-username <ACR username> \
  --registry-password <ACR password> \
  --os-type Linux \
  --cpu 1 --memory 1 \
  --restart-policy Always \
  --environment-variables \
    SMTP_USER=james.p.naive@gmail.com \
    SMTP_PASS=my_app_password \
    SMTP_HOST=smtp.gmail.com \
    SMTP_PORT=587
```

**Key Points**:

- `--restart-policy Always` ensures it stays up.  
- The **cron** inside the container triggers the script hourly, which uses environment variables to email you the signals.  
- Check logs using:
  ```bash
  az container logs --resource-group MyResourceGroup --name my-crypto-cron
  ```
  to see any cron or script output.

**Now** your Docker container runs 24/7 on Azure, updates crypto data each hour, generates signals, and emails you at `yourname@gmail.com` using your **`SMTP_PASS`** from environment variables.

---

## Project Structure

```
FinTech-Market-Analytics/
├── src/
│   └── market_analyzer/
│       ├── __init__.py
│       ├── analyzer.py            # MarketDataAnalyzer
│       ├── preprocessor.py        # Data cleaning & feature creation
│       ├── feature_engineering.py # Additional feature logic
│       ├── nn_strategy.py         # or timeseries_nn_strategy.py
│       ├── llm_sentiment_strategy.py
│       ├── rl_agent_strategy.py
│       ├── ensemble_strategy.py   # merges sub-models
│       ├── backtester.py          # Backtesting engine
│       ├── experiment_tracker.py  # Logging experiments
│       ├── dashboard.py           # StrategyDashboard
│       └── utils.py               # e.g. validate_data
├── ensemble_model_training.py     # script to train and test ensemble
├── ensemble_dashboard.py          # script to display ensemble results
├── hourly_update_and_predict.py   # fetch new data hourly, produce signals, email
├── data/
├── models/
├── results/
├── Dockerfile
├── crontab
└── tests/
```

---

## Contributing

1. **Fork** the repository  
2. **Create** your feature branch (`git checkout -b feature/AmazingFeature`)  
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)  
4. **Push** to the branch (`git push origin feature/AmazingFeature`)  
5. **Open a Pull Request**  

---

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.
```

> **Note:**  
> - You must set **`SMTP_USER`** and **`SMTP_PASS`** as environment variables when running the Docker container (either locally or on Azure).  
> - **Gmail requires** you to create an **App Password** if you have 2FA enabled—normal passwords won’t work. See [Google’s documentation](https://support.google.com/accounts/answer/185833) for details.  