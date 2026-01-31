# Code

- Entry point: `train.py`
- How to run:
  - Train: `python Experiments/exp-20260131-exp1/code/train.py`
  - Train with larger batch: `python Experiments/exp-20260131-exp1/code/train.py --batch-size 2048`
  - Train with grad accumulation: `python Experiments/exp-20260131-exp1/code/train.py --batch-size 2048 --grad-accum-steps 4`
  - Eval: `python Experiments/exp-20260131-exp1/code/eval.py`
  - Smoke test: `python Experiments/exp-20260131-exp1/code/smoke_test.py`
- Notes:
  - Uses all CSVs in `Data/` and resamples to 5-minute candles.
  - If `Data/*.csv` are Git LFS pointers, run `git lfs pull` before training.
  - TensorFlow summary logs saved under `Experiments/exp-20260131-exp1/artifacts/tf_logs`.
  - Feature/label cache saved under `Experiments/exp-20260131-exp1/artifacts/cache`.
  - tqdm progress bars show load/label prep and training/eval batches.
  - GPU is used automatically if CUDA is available.
  - Labels are log1p-transformed for training stability (no cap).
  - Early stopping monitors `val/loss` (patience=3, min_delta=0).
  - Model outputs SL/TP coefficients plus a trade probability head (threshold 0.6).
  - Training includes BCE trade loss and L2 penalty on predicted coefficients (1e-3).
