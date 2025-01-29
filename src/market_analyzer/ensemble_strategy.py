"""
ensemble_strategy.py
Combine TimeSeriesNNStrategy, LLMSentimentStrategy, RLAgentStrategy
into an EnsembleStrategy that merges signals.
"""

import pandas as pd
import numpy as np
from typing import List
from .strategy import TradingStrategy
from .timeseries_nn_strategy import TimeSeriesNNStrategy
from .llm_sentiment_strategy import LLMSentimentStrategy
from .rl_agent_strategy import RLAgentStrategy

class EnsembleStrategy(TradingStrategy):
    """
    Merges sub-strategies signals via either:
      - majority vote
      - or meta-learner (omitted for brevity)
    """

    def __init__(self,
                 sub_strategies: List[TradingStrategy],
                 voting=True,
                 name="ensemble_strategy"):
        super().__init__(name)
        self.sub_strategies = sub_strategies
        self.voting = voting
        # If meta-learning => store a small model, etc.

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if not self.sub_strategies:
            return pd.Series(0.0, index=data.index)

        # Gather each sub-strategy's signals
        sig_list = [s.generate_signals(data) for s in self.sub_strategies]

        if self.voting:
            return self._majority_vote(sig_list, data)
        else:
            # meta-model approach => omitted for brevity
            return self._sum_signals(sig_list, data)

    def _majority_vote(self, sig_list: List[pd.Series], data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0.0, index=data.index)
        df = pd.DataFrame(sig_list).T  # shape=(nrows, n_strategies)
        df.index = data.index
        for idx in data.index:
            row = df.loc[idx].values  # e.g. [-1,0,1]
            unique, counts = np.unique(row, return_counts=True)
            best = max(zip(unique, counts), key=lambda x: x[1])[0]
            signals.loc[idx] = best
        return signals

    def _sum_signals(self, sig_list: List[pd.Series], data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0.0, index=data.index)
        df = pd.DataFrame(sig_list).T
        df.index = data.index
        row_sums = df.sum(axis=1)
        signals[row_sums>0] = 1
        signals[row_sums<0] = -1
        return signals
