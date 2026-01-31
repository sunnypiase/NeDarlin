from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def compute_atr(df: pd.DataFrame, window: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - prev_close).abs()
    low_close = (df["low"] - prev_close).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=1).mean()
    return atr


def compute_features(df: pd.DataFrame, atr_window: int, std_window: int) -> Tuple[pd.DataFrame, List[str]]:
    features = pd.DataFrame(index=df.index)

    close = df["close"]
    volume = df["volume"]

    returns = close.pct_change().fillna(0.0)
    log_returns = np.log(close).diff().fillna(0.0)
    volume_change = volume.pct_change().fillna(0.0)

    features["returns"] = returns
    features["log_returns"] = log_returns
    features["volume_change"] = volume_change

    for window in (5, 20, 60):
        features[f"ret_mean_{window}"] = returns.rolling(window=window, min_periods=1).mean()
        features[f"ret_std_{window}"] = returns.rolling(window=window, min_periods=1).std().fillna(0.0)
        features[f"vol_mean_{window}"] = volume.rolling(window=window, min_periods=1).mean()
        features[f"vol_std_{window}"] = volume.rolling(window=window, min_periods=1).std().fillna(0.0)

    features["price_vs_sma_20"] = close / close.rolling(window=20, min_periods=1).mean() - 1.0
    features["price_vs_sma_60"] = close / close.rolling(window=60, min_periods=1).mean() - 1.0

    features["volatility_20"] = returns.rolling(window=20, min_periods=1).std().fillna(0.0)

    features["atr"] = compute_atr(df, atr_window)
    features["std_close"] = close.rolling(window=std_window, min_periods=1).std().fillna(0.0)

    features = features.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return features, list(features.columns)


def prepare_input(candles_5m: pd.DataFrame, n: int, atr_window: int, std_window: int) -> np.ndarray:
    features, _ = compute_features(candles_5m, atr_window=atr_window, std_window=std_window)
    if len(features) < n:
        raise ValueError(f"Not enough rows to build window of {n}")
    window = features.iloc[-n:]
    return window.to_numpy(dtype=np.float32)
