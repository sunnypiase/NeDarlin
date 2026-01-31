from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import build_config
from data_io import compute_data_hash, list_data_files, load_all_symbols
from dataset import build_datasets
from labeling import LabelConfig
from model import ActorCritic
from timing import StageTimer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(num_epochs: int | None = None, batch_size: int | None = None) -> None:
    config = build_config()
    set_seed(config.seed)
    timer = StageTimer()

    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    config.tf_log_dir.mkdir(parents=True, exist_ok=True)

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

    train_set, val_set, test_set, feature_names = build_datasets(
        symbol_frames,
        window_size=config.window_size,
        atr_window=config.atr_window,
        std_window=config.std_window,
        label_config=label_config,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        cache_dir=config.cache_dir,
        parallel_prep=config.parallel_prep,
        max_workers=config.max_workers,
        timer=timer,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    effective_batch = batch_size if batch_size is not None else config.batch_size

    train_loader = DataLoader(
        train_set,
        batch_size=effective_batch,
        shuffle=True,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=effective_batch,
        shuffle=False,
        pin_memory=pin_memory,
    )

    model = ActorCritic(input_size=len(feature_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    writer = tf.summary.create_file_writer(str(config.tf_log_dir))
    data_hash = compute_data_hash(list_data_files(config.data_dir))

    with writer.as_default():
        tf.summary.text("data_hash", data_hash, step=0)
        tf.summary.text("feature_names", ", ".join(feature_names), step=0)

    step = 0
    epochs = num_epochs if num_epochs is not None else config.num_epochs
    timer.start("train_total")
    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Train epoch {epoch + 1}/{epochs}", unit="batch"):
            features, labels, rr, trade = batch
            features = features.to(device, non_blocking=pin_memory)
            labels = labels.to(device, non_blocking=pin_memory)
            rr = rr.to(device, non_blocking=pin_memory)
            trade = trade.to(device, non_blocking=pin_memory)
            pred_coeffs, value = model(features)

            trade_mask = trade > 0.5
            if trade_mask.any():
                actor_loss = torch.mean((pred_coeffs[trade_mask] - labels[trade_mask]) ** 2)
            else:
                actor_loss = torch.tensor(0.0)

            critic_loss = torch.mean((value - rr) ** 2)
            loss = actor_loss + config.critic_weight * critic_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with writer.as_default():
                tf.summary.scalar("train/loss", loss.item(), step=step)
                tf.summary.scalar("train/actor_loss", actor_loss.item(), step=step)
                tf.summary.scalar("train/critic_loss", critic_loss.item(), step=step)
                tf.summary.scalar("train/trade_rate", trade.float().mean().item(), step=step)

            step += 1

        val_metrics = evaluate(model, val_loader, device, pin_memory)
        with writer.as_default():
            tf.summary.scalar("val/loss", val_metrics["loss"], step=step)
            tf.summary.scalar("val/actor_loss", val_metrics["actor_loss"], step=step)
            tf.summary.scalar("val/critic_loss", val_metrics["critic_loss"], step=step)
            tf.summary.scalar("val/trade_rate", val_metrics["trade_rate"], step=step)

    timer.stop("train_total")
    torch.save(model.state_dict(), config.artifacts_dir / "model.pt")
    print("Timing summary")
    for line in timer.summary_lines():
        print(line)


def evaluate(model: ActorCritic, loader: DataLoader, device: torch.device, pin_memory: bool) -> dict:
    model.eval()
    total_loss = 0.0
    total_actor = 0.0
    total_critic = 0.0
    total_trade = 0.0
    count = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", unit="batch"):
            features, labels, rr, trade = batch
            features = features.to(device, non_blocking=pin_memory)
            labels = labels.to(device, non_blocking=pin_memory)
            rr = rr.to(device, non_blocking=pin_memory)
            trade = trade.to(device, non_blocking=pin_memory)
            pred_coeffs, value = model(features)
            trade_mask = trade > 0.5
            if trade_mask.any():
                actor_loss = torch.mean((pred_coeffs[trade_mask] - labels[trade_mask]) ** 2)
            else:
                actor_loss = torch.tensor(0.0)
            critic_loss = torch.mean((value - rr) ** 2)
            loss = actor_loss + critic_loss
            total_loss += loss.item()
            total_actor += actor_loss.item()
            total_critic += critic_loss.item()
            total_trade += trade.float().mean().item()
            count += 1

    if count == 0:
        return {"loss": 0.0, "actor_loss": 0.0, "critic_loss": 0.0, "trade_rate": 0.0}
    return {
        "loss": total_loss / count,
        "actor_loss": total_actor / count,
        "critic_loss": total_critic / count,
        "trade_rate": total_trade / count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    train(num_epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
