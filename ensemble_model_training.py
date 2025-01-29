"""
ensemble_model_training.py - script to train sub-models and build an ensemble strategy.
"""

import pandas as pd
import numpy as np

from market_analyzer.analyzer import MarketDataAnalyzer
from market_analyzer.experiment_tracker import ExperimentTracker
from market_analyzer.backtester import Backtester

# Our ensemble and sub-strategies:
from market_analyzer.ensemble_strategy import (
    TimeSeriesNNStrategy, LLMSentimentStrategy, RLAgentStrategy, EnsembleStrategy
)

def main():
    # 1) Load data
    analyzer = MarketDataAnalyzer()
    analyzer.download_data(period="1y")
    data = analyzer.get_asset_data("BTC-USD")

    # 2) Sub-model placeholders
    ts_nn = TimeSeriesNNStrategy("time_series_nn")
    llm_sent = LLMSentimentStrategy("llm_sentiment")
    rl_agent = RLAgentStrategy("rl_agent")

    # Possibly "train" them => in reality we'd do real training steps
    # e.g. ts_nn.train(...), llm_sent.load_pretrained(...), rl_agent.fit(...)
    # but here we'll skip for brevity.

    # 3) Build an ensemble (voting_mode=True or meta-learning as we wish)
    ensemble = EnsembleStrategy(sub_strategies=[ts_nn, llm_sent, rl_agent],
                                voting_mode=True,
                                name="ensemble_strategy")

    # 4) Backtester => evaluate on validation set
    backtester = Backtester(data, train_size=0.6, test_size=0.2)
    val_results = backtester.evaluate_strategy(ensemble, backtester.validation_data)
    print("[Ensemble] Validation results:", val_results)

    # If we want to log this in experiment tracker, convert Series to list for JSON:
    for key in ["portfolio_value", "signals", "returns"]:
        if isinstance(val_results.get(key), pd.Series):
            val_results[key] = val_results[key].tolist()

    # 5) Save experiment
    tracker = ExperimentTracker(results_dir="results")
    tracker.save_experiment(
        experiment_name="ensemble_trading",
        model_name="EnsembleStrategy",
        train_metrics={},
        test_metrics={},  # or some real test
        validation_metrics=val_results,
        params={"voting_mode": True, "sub_models": ["ts_nn", "llm_sent", "rl_agent"]}
    )

if __name__=="__main__":
    main()
