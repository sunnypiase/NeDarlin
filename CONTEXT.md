# Project Context

This file tracks high-level context and important changes. Update it whenever
material project decisions, data changes, or behavior shifts occur.

## Purpose
- Single source of truth for project intent and direction.
- Quick onboarding for future work sessions.

## Project Overview
- Domain:
- Goal:
- Key outputs:

## Data Sources
- Data location(s): `Data/*.csv` (APTUSDT, BTCUSDT, DOGEUSDT, ETHUSDT, IPUSDT, XLMUSDT)
- Data freshness/refresh cadence: 1-minute OHLCV candles; latest close_time 2025-11-30 23:59:59 UTC
- Notes on data quality: consistent 12-column schema; no empty cells detected; 1-minute interval with no gaps/duplicates; open_time monotonic; close_time - open_time = 59,999 ms; OHLC bounds valid; no negative volume; IPUSDT starts later (2025-02-13 17:00:00 UTC)

## Decisions
- Decision:
  - Rationale:
  - Date:
  - Alternatives considered:

## Recent Changes (append new items)
- 2026-01-31:
  - Summary: Added experiment structure and templates for hypotheses, code, and results.
  - Affected files: `Experiments/`, `README.md`, `.cursor/rules/experiments-structure.mdc`
  - Reason: Standardize experiment workflow and documentation.
  - Follow-ups: Create the first experiment folder from the template when ready.
- 2026-01-31:
  - Summary: Implemented Experiment 1 pipeline (data IO, features, labeling, model, train/eval).
  - Affected files: `Experiments/exp-20260131-exp1/code/`, `Experiments/exp-20260131-exp1/results.md`, `CONTEXT.md`
  - Reason: Provide a runnable, leakage-safe experiment implementation.
  - Follow-ups: Install dependencies and run training/evaluation.
- 2026-01-31:
  - Summary: Hardened CSV loader against mixed types and timestamp overflows.
  - Affected files: `Experiments/exp-20260131-exp1/code/data_io.py`, `CONTEXT.md`
  - Reason: Prevent read failures from mixed-type columns and invalid timestamps.
  - Follow-ups: Re-run smoke test.
- 2026-01-31:
  - Summary: Added per-symbol feature/label caching to speed up prep.
  - Affected files: `Experiments/exp-20260131-exp1/code/dataset.py`, `Experiments/exp-20260131-exp1/code/config.py`, `Experiments/exp-20260131-exp1/code/README.md`
  - Reason: Avoid recomputing labels/features on repeated runs.
  - Follow-ups: Verify cache reuse on next run.
- 2026-01-31:
  - Summary: Added tqdm progress bars, timing summaries, and optional parallel/GPU execution.
  - Affected files: `Experiments/exp-20260131-exp1/code/`, `pyproject.toml`, `CONTEXT.md`
  - Reason: Improve visibility into long-running steps and speed up prep/training.
  - Follow-ups: Re-run smoke test and short training to confirm performance.
- 2026-01-31:
  - Summary: Suppressed TensorFlow oneDNN warning in training script.
  - Affected files: `Experiments/exp-20260131-exp1/code/train.py`, `CONTEXT.md`
  - Reason: Reduce noisy startup logs.
  - Follow-ups: Re-run training to confirm warning is gone.
- 2026-01-31:
  - Summary: Increased default batch size and added CLI override for training.
  - Affected files: `Experiments/exp-20260131-exp1/code/config.py`, `Experiments/exp-20260131-exp1/code/train.py`, `Experiments/exp-20260131-exp1/code/README.md`, `CONTEXT.md`
  - Reason: Speed up training throughput.
  - Follow-ups: Monitor GPU/CPU memory usage and adjust batch size if needed.
- 2026-01-31:
  - Summary: Profiled CSV datasets in `Data/` (row counts, schema, time ranges).
  - Affected files: `Data/` CSVs, `CONTEXT.md`
  - Reason: Establish baseline dataset coverage and quality notes.
  - Follow-ups: If needed, confirm candle interval via time deltas and document refresh source.
- 2026-01-31:
  - Summary: Verified data quality checks (interval continuity, OHLC bounds, volume sign, close_time deltas).
  - Affected files: `Data/` CSVs, `CONTEXT.md`
  - Reason: Validate dataset integrity before analysis.
  - Follow-ups: Document data ingestion/source when available.
- 2026-01-31:
  - Summary: Added Experiment 1 documentation and requirements (hypothesis, description, scaffolding).
  - Affected files: `Experiments/exp-20260131-exp1/`, `CONTEXT.md`
  - Reason: Define experiment scope, labeling rules, and production-ready input prep.
  - Follow-ups: Implement data prep, labeling, and model training code.
