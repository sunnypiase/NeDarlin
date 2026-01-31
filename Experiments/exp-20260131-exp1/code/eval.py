from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import build_config
from data_io import load_all_symbols
from dataset import build_datasets
from labeling import LabelConfig
from model import ActorCritic
from timing import StageTimer


def evaluate() -> None:
    config = build_config()
    timer = StageTimer()
    symbol_frames = load_all_symbols(
        config.data_dir,
        parallel=config.parallel_prep,
        max_workers=config.max_workers,
        timer=timer,
    )
    label_config = LabelConfig(
        horizon_bars=int((config.horizon_days * 24 * 60) / config.resample_minutes),
        min_move_pct=config.min_move_pct,
        min_rr=1.0,
    )

    _, _, test_set, feature_names = build_datasets(
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

    loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )
    model = ActorCritic(input_size=len(feature_names)).to(device)
    model_path = config.artifacts_dir / "model.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    pred_coeffs_all = []
    true_coeffs_all = []
    rr_all = []
    trade_all = []
    trade_logit_all = []

    timer.start("eval_forward")
    with torch.no_grad():
        for features, labels, rr, trade in tqdm(loader, desc="Eval", unit="batch"):
            features = features.to(device, non_blocking=pin_memory)
            labels = labels.to(device, non_blocking=pin_memory)
            rr = rr.to(device, non_blocking=pin_memory)
            trade = trade.to(device, non_blocking=pin_memory)
            pred_coeffs, trade_logit, _ = model(features)
            pred_coeffs_all.append(pred_coeffs.cpu().numpy())
            true_coeffs_all.append(labels.cpu().numpy())
            rr_all.append(rr.cpu().numpy())
            trade_all.append(trade.cpu().numpy())
            trade_logit_all.append(trade_logit.cpu().numpy())
    timer.stop("eval_forward")

    pred_coeffs = np.concatenate(pred_coeffs_all, axis=0)
    true_coeffs = np.concatenate(true_coeffs_all, axis=0)

    if config.label_log_transform:
        pred_coeffs = np.expm1(pred_coeffs)
        true_coeffs = np.expm1(true_coeffs)
    rr = np.concatenate(rr_all, axis=0)
    trade = np.concatenate(trade_all, axis=0)
    trade_logit = np.concatenate(trade_logit_all, axis=0)

    trade_mask = trade > 0.5
    mse = float(np.mean((pred_coeffs - true_coeffs) ** 2))
    mse_trade = float(np.mean((pred_coeffs[trade_mask] - true_coeffs[trade_mask]) ** 2)) if trade_mask.any() else 0.0

    pred_trade = (1.0 / (1.0 + np.exp(-trade_logit))) >= config.trade_threshold
    trade_rate = float(trade.mean())
    pred_trade_rate = float(pred_trade.mean())
    precision = float((pred_trade & trade_mask).sum() / max(pred_trade.sum(), 1))
    recall = float((pred_trade & trade_mask).sum() / max(trade_mask.sum(), 1))

    print("Test metrics")
    print(f"mse_all: {mse:.6f}")
    print(f"mse_trade: {mse_trade:.6f}")
    print(f"trade_rate_true: {trade_rate:.4f}")
    print(f"trade_rate_pred: {pred_trade_rate:.4f}")
    print(f"precision: {precision:.4f}")
    print(f"recall: {recall:.4f}")
    print(f"rr_mean_true: {rr.mean():.4f}")
    print("Timing summary")
    for line in timer.summary_lines():
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    evaluate()


if __name__ == "__main__":
    main()
