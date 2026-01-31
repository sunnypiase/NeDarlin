from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabelConfig:
    horizon_bars: int
    min_move_pct: float
    min_rr: float


def compute_path_labels(
    df: pd.DataFrame,
    atr: pd.Series,
    std_close: pd.Series,
    config: LabelConfig,
) -> Tuple[pd.DataFrame, np.ndarray]:
    n = len(df)
    labels = pd.DataFrame(index=df.index)
    labels["label_sl_coef"] = 0.0
    labels["label_tp_coef"] = 0.0
    labels["label_rr"] = 0.0
    labels["label_trade"] = 0

    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    atr_values = atr.to_numpy()
    std_values = std_close.to_numpy()

    for i in range(n):
        end = min(n, i + 1 + config.horizon_bars)
        if i + 1 >= end:
            continue

        p0 = close[i]
        min_move_abs = p0 * config.min_move_pct
        future_high = high[i + 1 : end]
        future_low = low[i + 1 : end]

        running_min_low = np.minimum.accumulate(future_low)
        tp_dist = future_high - p0
        sl_dist = np.maximum(min_move_abs, p0 - running_min_low)
        sl_price = p0 - sl_dist

        valid = tp_dist >= min_move_abs
        valid &= future_low > sl_price  # SL-first rule: invalid if low hits SL

        rr = np.full_like(tp_dist, -np.inf, dtype=np.float64)
        rr[valid] = tp_dist[valid] / sl_dist[valid]

        best_idx = int(np.argmax(rr))
        best_rr = float(rr[best_idx])
        if not np.isfinite(best_rr) or best_rr < config.min_rr:
            continue

        best_tp = float(tp_dist[best_idx])
        best_sl = float(sl_dist[best_idx])

        scale = max(float(atr_values[i]), float(std_values[i]), 1e-8)
        coeff_scale = scale * config.min_move_pct

        labels.iloc[i, labels.columns.get_loc("label_sl_coef")] = best_sl / coeff_scale
        labels.iloc[i, labels.columns.get_loc("label_tp_coef")] = best_tp / coeff_scale
        labels.iloc[i, labels.columns.get_loc("label_rr")] = best_rr
        labels.iloc[i, labels.columns.get_loc("label_trade")] = 1

    mask_trade = labels["label_trade"].to_numpy().astype(bool)
    return labels, mask_trade
