"""
timeseries_nn_strategy.py
A "real" LSTM-based time-series strategy with a minimal Keras example.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from .strategy import TradingStrategy

class TimeSeriesNNStrategy(TradingStrategy):
    """
    A more 'real' LSTM-based approach:
      - We define train(...) that fits a Keras model on windowed data,
        e.g. classification in {0,1,2} => Sell/Hold/Buy
      - We define generate_signals(...) that does sliding-window inference.
    """

    def __init__(self, name="time_series_nn", seq_length=30, num_features=5, n_hidden=64):
        super().__init__(name)
        self.seq_length = seq_length
        self.num_features = num_features
        self.n_hidden = n_hidden
        self.model = None

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              epochs=5,
              batch_size=32,
              lr=1e-3):
        """
        X_* shape = (samples, seq_length, num_features)
        y_* shape = (samples,) or (samples,3) if one-hot

        We'll assume y is an integer in {0,1,2} => we do a classification approach
        with one-hot encoding inside the code.
        """
        # One-hot if needed:
        if y_train.ndim==1:
            # make it (samples,3)
            y_train_oh = np.zeros((y_train.shape[0], 3), dtype=np.float32)
            for i,cls in enumerate(y_train):
                y_train_oh[i, cls] = 1.0
            y_val_oh = np.zeros((y_val.shape[0], 3), dtype=np.float32)
            for i,cls in enumerate(y_val):
                y_val_oh[i, cls] = 1.0
        else:
            y_train_oh = y_train
            y_val_oh = y_val

        self.model = Sequential()
        self.model.add(LSTM(self.n_hidden, input_shape=(self.seq_length, self.num_features)))
        self.model.add(Dense(3, activation="softmax"))
        opt = Adam(learning_rate=lr)
        self.model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

        es = EarlyStopping(patience=3, restore_best_weights=True)

        self.model.fit(
            X_train, y_train_oh,
            validation_data=(X_val, y_val_oh),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es],
            verbose=1
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        1) Build sliding-window input => shape (samples, seq_length, num_features)
        2) Model.predict => (samples, 3)
        3) Argmax => {0,1,2} => map to [-1, 0, 1]
        """
        signals = pd.Series(0.0, index=data.index)

        if self.model is None:
            print("[TimeSeriesNN] No model found, returning zeros.")
            return signals

        X = self._prepare_inference_data(data)
        if X is None or len(X)==0:
            return signals

        preds = self.model.predict(X)
        classes = preds.argmax(axis=1)  # in {0,1,2}
        # Map 0->-1, 1->0, 2->1
        idx_map = {0:-1, 1:0, 2:1}
        mapped = [idx_map[c] for c in classes]

        offset = self.seq_length - 1
        n_preds = len(mapped)
        pred_index = data.index[offset : offset+n_preds]
        partial_signals = pd.Series(mapped, index=pred_index, dtype=float)
        signals.update(partial_signals)

        return signals

    def _prepare_inference_data(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Build windows from the last seq_length rows each time.
        We'll assume columns: "Close","High","Low","Open","Volume" => total=5
        """
        req_cols = ["Close","High","Low","Open","Volume"]
        if any(c not in data.columns for c in req_cols):
            print("[TimeSeriesNN] Missing some required columns.")
            return None

        arr = data[req_cols].values  # shape (N,5)
        if len(arr) < self.seq_length:
            return None

        out = []
        for i in range(self.seq_length, len(arr)+1):
            window = arr[i-self.seq_length : i]
            out.append(window)
        return np.array(out, dtype=np.float32)
