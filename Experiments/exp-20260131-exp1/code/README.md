# Code

- Entry point: `train.py`
- How to run:
  - Train: `python Experiments/exp-20260131-exp1/code/train.py`
  - Train with larger batch: `python Experiments/exp-20260131-exp1/code/train.py --batch-size 256`
  - Eval: `python Experiments/exp-20260131-exp1/code/eval.py`
  - Smoke test: `python Experiments/exp-20260131-exp1/code/smoke_test.py`
- Notes:
  - Uses all CSVs in `Data/` and resamples to 5-minute candles.
  - TensorFlow summary logs saved under `Experiments/exp-20260131-exp1/artifacts/tf_logs`.
  - Feature/label cache saved under `Experiments/exp-20260131-exp1/artifacts/cache`.
  - tqdm progress bars show load/label prep and training/eval batches.
  - GPU is used automatically if CUDA is available.
