"""
Neural network based trading strategy.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# If you have a base class:
from .strategy import TradingStrategy
# If you have a data validation function:
from .utils import validate_data

class NeuralNetworkStrategy(TradingStrategy):
    """
    A neural network-based trading strategy that outputs signals in {-1,0,1}.
    """

    def __init__(self, params: Dict = None):
        super().__init__("nn_strategy")
        default_params = {
            "seq_length": 30,
            "num_features": 5,
            "n_hidden": 64,
            "epochs": 5,
            "batch_size": 32,
            "learning_rate": 1e-3
        }
        self.params = {**default_params, **(params or {})}
        self.model = None

    def build_model(self):
        """
        Build a simple LSTM-based Keras model for classification.
        """
        seq_len  = self.params["seq_length"]
        num_feat = self.params["num_features"]
        n_hidden = self.params["n_hidden"]

        model = Sequential()
        model.add(LSTM(n_hidden, input_shape=(seq_len, num_feat)))
        model.add(Dense(3, activation="softmax"))  # 3 classes => Sell,Hold,Buy
        opt = Adam(learning_rate=self.params["learning_rate"])
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        self.model = model

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """
        Train the neural network. Return a dict with final metrics.
        """
        if self.model is None:
            self.build_model()

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.params["epochs"],
            batch_size=self.params["batch_size"],
            verbose=1
        )
        final_val_acc = history.history["val_accuracy"][-1]
        return {"final_val_acc": float(final_val_acc)}

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate signals in {-1, 0, 1}.
        Must return a pandas Series with data.index, never a dict.
        """
        # Validate data => must remain a DataFrame
        try:
            data = validate_data(data)
        except ValueError as e:
            print("[NN Strategy] Validation error:", e)
            # Return zeros if invalid
            return pd.Series(0.0, index=getattr(data, "index", pd.RangeIndex(0)))

        if data.empty:
            print("[NN Strategy] Data is empty; returning zeros.")
            return pd.Series(0.0, index=data.index)

        if self.model is None:
            print("[NN Strategy] Model not trained; returning zeros.")
            return pd.Series(0.0, index=data.index)

        # Prepare inference data => shape (samples, seq_len, num_features)
        X = self._prepare_inference_data(data)
        if X is None or len(X)==0:
            print("[NN Strategy] Inference data empty; returning zeros.")
            return pd.Series(0.0, index=data.index)

        # Predict => shape (samples, 3)
        preds = self.model.predict(X)
        class_idx = preds.argmax(axis=1)  # in {0,1,2}

        # Map 0->-1, 1->0, 2->1
        idx_map = {0: -1, 1: 0, 2: 1}
        mapped = [idx_map[c] for c in class_idx]

        # Align signals with data.index
        signals = pd.Series(0.0, index=data.index)
        offset = self.params["seq_length"] - 1
        n_preds = len(mapped)
        pred_index = data.index[offset : offset + n_preds]

        partial_signals = pd.Series(mapped, index=pred_index)
        signals.update(partial_signals)

        return signals

    def _prepare_inference_data(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Convert data -> shape (samples, seq_len, num_features).
        Example columns: ["Close","High","Low","Open","Volume"].
        """
        seq_len = self.params["seq_length"]
        feat_cols = ["Close","High","Low","Open","Volume"]
        if any(c not in data.columns for c in feat_cols):
            print("[NN Strategy] Missing columns for inference.")
            return None

        arr = data[feat_cols].values  # shape (N,5)
        if len(arr) < seq_len:
            return None

        windows = []
        for i in range(seq_len, len(arr)+1):
            window = arr[i-seq_len : i]
            windows.append(window)
        return np.array(windows, dtype=np.float32)
