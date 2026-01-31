# Hypothesis

- Statement: A leakage-safe, stationary-feature pipeline on 5m data with 400-candle windows can train a deep RL model to predict valid SL/TP coefficients (relative to ATR/STD) that yield tradable R/R (>= 1.0) within a 1-week horizon across all symbols.
- Motivation: Use production-ready input prep and path-dependent labeling to align backtests with real execution constraints.
- Expected impact: A reliable signal for opening trades with SL/TP levels that clear minimum cost-adjusted move thresholds.
- Success metrics: Percentage of labeled trades with realized R/R >= 1.0; average realized R/R; hit rate of SL/TP; stability across symbols and time splits.
- Data scope (symbols, time range): All symbols in `Data/*.csv`, full available history; 1m candles aggregated to 5m.
- Method outline: Aggregate to 5m, build stationary features from rolling windows (no scaling leakage), create 400-candle inputs with last candle as current, label via 1-week horizon grid search for best R/R, train actor-critic to predict relative SL/TP coefficients.
- Assumptions: 5m aggregation preserves relevant microstructure; ATR/STD provide stable scaling; 1-week horizon is sufficient for R/R discovery; OHLC path approximation with conservative SL-first rule.
- Risks / confounders: Hidden leakage from normalization, non-stationary regimes, low-liquidity periods, symbol-specific behavior, OHLC path ambiguity.
- Stop criteria: Evidence of data leakage; model fails to exceed R/R >= 1.0 baseline; inconsistent results across time splits; instability in training.
