from __future__ import annotations

import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import build_config
from data_io import load_all_symbols
from dataset import build_datasets
from labeling import LabelConfig
from model import ActorCritic
from timing import StageTimer


def run_smoke_test() -> None:
    config = build_config()
    timer = StageTimer()
    symbol_frames = load_all_symbols(
        config.data_dir,
        parallel=config.parallel_prep,
        max_workers=config.max_workers,
        timer=timer,
    )
    first_symbol = next(iter(symbol_frames))
    symbol_frames = {first_symbol: symbol_frames[first_symbol]}

    label_config = LabelConfig(
        horizon_bars=int((config.horizon_days * 24 * 60) / config.resample_minutes),
        min_move_pct=config.min_move_pct,
        min_rr=1.0,
    )

    train_set, _, _, feature_names = build_datasets(
        symbol_frames,
        window_size=config.window_size,
        atr_window=config.atr_window,
        std_window=config.std_window,
        label_config=label_config,
        label_log_transform=config.label_log_transform,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        cache_dir=config.cache_dir,
        parallel_prep=config.parallel_prep,
        max_workers=config.max_workers,
        timer=timer,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    loader = DataLoader(train_set, batch_size=8, shuffle=True, pin_memory=pin_memory)
    model = ActorCritic(input_size=len(feature_names)).to(device)

    timer.start("smoke_forward")
    for features, labels, rr, trade in tqdm(loader, total=1, desc="Smoke batch", unit="batch"):
        features = features.to(device, non_blocking=pin_memory)
        labels = labels.to(device, non_blocking=pin_memory)
        rr = rr.to(device, non_blocking=pin_memory)
        trade = trade.to(device, non_blocking=pin_memory)
        pred_coeffs, value = model(features)
        break
    timer.stop("smoke_forward")
    print("Smoke test batch")
    print("features:", features.shape)
    print("labels:", labels.shape)
    print("rr:", rr.shape)
    print("trade:", trade.shape)
    print("pred_coeffs:", pred_coeffs.shape)
    print("value:", value.shape)
    print("Timing summary")
    for line in timer.summary_lines():
        print(line)


if __name__ == "__main__":
    run_smoke_test()
