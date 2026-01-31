# Experiment 1 Description

## Goal
Create a production-ready, leakage-safe pipeline and labeling scheme to train a deep learning model that predicts SL/TP coefficients (not absolute prices) that yield R/R >= 1.0 within a 1-week horizon.

## Data
- Source: `Data/*.csv` (all symbols), full available history.
- Base interval: 1m OHLCV.
- Aggregation: resample to 5m candles before feature generation and labeling.

## Input preparation (production-ready)
- Window size: 400 candles (or configurable `n`), where the last candle is the current candle.
- Function contract: `prepare_input(candles_5m: DataFrame, n: int = 400) -> features`.
- Stationary-only features (no global scaling). Examples:
  - Returns and log-returns, rolling distributions/quantiles.
  - Volume profiles, volume deltas, rolling stats.
  - Percent changes, rolling mean/variance, volatility proxies.
  - ATR/STD and derived ratios.
- Real-time readiness: only use information available up to the current candle; no future-peeking transforms.

## Leakage controls
- Time-based splits only (train/val/test by time).
- Normalization is window-local or rolling and computed strictly from past data.
- No cross-window statistics that include future candles.

## Trade/label logic
- Horizon: 1 week max holding (5m bars within 7 days).
- Labeling: for each current candle, search a grid of SL/TP pairs within the next-week window to maximize R/R.
- Trade eligibility: if best R/R < 1.0, label as no-trade.
- Minimum move threshold: require move >= `0.0500% * 2.5` to cover commission and slippage.
- SL-first rule: if SL and TP are both inside the same candle, count as SL (conservative).
- If time constraint expires (1 week) without TP/SL, close at time limit and treat as no-trade or loss based on realized R/R policy (to be fixed in code).

## Expected value target
- For each candle, compute the max achievable R/R within the next week.
- Choose the TP/SL pair that maximizes R/R, subject to the minimum move threshold.
- Output labels: relative coefficients for SL and TP, scaled by ATR/STD and multiplied by `0.0500% * 2.5` to enforce valid minimum predictions.

## Model plan
- Framework: PyTorch.
- Architecture: actor-critic (policy predicts SL/TP coefficients, critic estimates reward).
- Actor outputs: relative SL/TP coefficients (not price levels).
- Critic reward: based on realized R/R from the label grid search.

## Traceability
- Use TensorFlow for experiment tracking and parameter logging (metrics, config, seeds, data splits).
- Record: data hash, feature list, window size, labeling grid, min move threshold, and training hyperparameters.

## Non-negotiables
- No data leakage.
- Input preparation must be production-ready and compatible with real-time streaming.
