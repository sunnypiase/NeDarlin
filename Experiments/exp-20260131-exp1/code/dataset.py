from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset
from tqdm import tqdm

from features import compute_features
from labeling import LabelConfig, compute_path_labels
from timing import StageTimer


@dataclass(frozen=True)
class SymbolData:
    features: np.ndarray
    labels: np.ndarray
    rr: np.ndarray
    trade_mask: np.ndarray
    timestamps: np.ndarray
    feature_names: List[str]


class WindowDataset(Dataset):
    def __init__(self, symbol_data: SymbolData, indices: Sequence[int], window_size: int) -> None:
        self.symbol_data = symbol_data
        self.indices = list(indices)
        self.window_size = window_size

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        end_idx = self.indices[idx]
        start_idx = end_idx - self.window_size + 1
        features = self.symbol_data.features[start_idx : end_idx + 1]
        label = self.symbol_data.labels[end_idx]
        rr = self.symbol_data.rr[end_idx]
        trade = self.symbol_data.trade_mask[end_idx]

        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(rr, dtype=torch.float32),
            torch.tensor(trade, dtype=torch.float32),
        )


def _split_indices(n: int, train_ratio: float, val_ratio: float) -> Tuple[range, range, range]:
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return range(0, train_end), range(train_end, val_end), range(val_end, n)


def build_symbol_data(
    df: pd.DataFrame,
    window_size: int,
    atr_window: int,
    std_window: int,
    label_config: LabelConfig,
) -> SymbolData:
    features_df, feature_names = compute_features(df, atr_window=atr_window, std_window=std_window)
    labels_df, trade_mask = compute_path_labels(
        df,
        atr=features_df["atr"],
        std_close=features_df["std_close"],
        config=label_config,
    )

    features = features_df.to_numpy(dtype=np.float32)
    labels = labels_df[["label_sl_coef", "label_tp_coef"]].to_numpy(dtype=np.float32)
    rr = labels_df["label_rr"].to_numpy(dtype=np.float32)
    timestamps = df.index.to_numpy()

    return SymbolData(
        features=features,
        labels=labels,
        rr=rr,
        trade_mask=trade_mask.astype(np.float32),
        timestamps=timestamps,
        feature_names=feature_names,
    )


def _cache_key(
    symbol: str,
    window_size: int,
    atr_window: int,
    std_window: int,
    label_config: LabelConfig,
    label_log_transform: bool,
    data_stamp: str,
) -> str:
    return (
        f"{symbol}_w{window_size}_atr{atr_window}_std{std_window}_"
        f"h{label_config.horizon_bars}_mm{label_config.min_move_pct}_rr{label_config.min_rr}_"
        f"log{int(label_log_transform)}_{data_stamp}"
    )


def _data_stamp(df: pd.DataFrame) -> str:
    if df.empty:
        return "empty"
    return f"{len(df)}_{df.index[0].value}_{df.index[-1].value}"


def build_symbol_data_cached(
    symbol: str,
    df: pd.DataFrame,
    window_size: int,
    atr_window: int,
    std_window: int,
    label_config: LabelConfig,
    label_log_transform: bool,
    cache_dir: Path | None = None,
) -> SymbolData:
    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = _cache_key(
            symbol,
            window_size,
            atr_window,
            std_window,
            label_config,
            label_log_transform,
            _data_stamp(df),
        )
        cache_path = cache_dir / f"{key}.npz"
        if cache_path.exists():
            cached = np.load(cache_path, allow_pickle=True)
            return SymbolData(
                features=cached["features"],
                labels=cached["labels"],
                rr=cached["rr"],
                trade_mask=cached["trade_mask"],
                timestamps=cached["timestamps"],
                feature_names=list(cached["feature_names"]),
            )

    symbol_data = build_symbol_data(
        df,
        window_size=window_size,
        atr_window=atr_window,
        std_window=std_window,
        label_config=label_config,
    )

    labels = symbol_data.labels
    if label_log_transform:
        labels = np.log1p(np.maximum(labels, 0.0))

    symbol_data = SymbolData(
        features=symbol_data.features,
        labels=labels.astype(np.float32),
        rr=symbol_data.rr,
        trade_mask=symbol_data.trade_mask,
        timestamps=symbol_data.timestamps,
        feature_names=symbol_data.feature_names,
    )

    if cache_path is not None:
        np.savez_compressed(
            cache_path,
            features=symbol_data.features,
            labels=symbol_data.labels,
            rr=symbol_data.rr,
            trade_mask=symbol_data.trade_mask,
            timestamps=symbol_data.timestamps,
            feature_names=np.array(symbol_data.feature_names),
        )

    return symbol_data


def _build_symbol_data_worker(
    symbol: str,
    df: pd.DataFrame,
    window_size: int,
    atr_window: int,
    std_window: int,
    label_config: LabelConfig,
    label_log_transform: bool,
    cache_dir: str | None,
) -> tuple[str, SymbolData]:
    cache_path = Path(cache_dir) if cache_dir is not None else None
    symbol_data = build_symbol_data_cached(
        symbol,
        df,
        window_size=window_size,
        atr_window=atr_window,
        std_window=std_window,
        label_config=label_config,
        label_log_transform=label_log_transform,
        cache_dir=cache_path,
    )
    return symbol, symbol_data


def build_datasets(
    symbol_frames: Dict[str, pd.DataFrame],
    window_size: int,
    atr_window: int,
    std_window: int,
    label_config: LabelConfig,
    label_log_transform: bool,
    train_ratio: float,
    val_ratio: float,
    cache_dir: Path | None = None,
    parallel_prep: bool = False,
    max_workers: int | None = None,
    timer: StageTimer | None = None,
) -> Tuple[ConcatDataset, ConcatDataset, ConcatDataset, List[str]]:
    train_sets: List[Dataset] = []
    val_sets: List[Dataset] = []
    test_sets: List[Dataset] = []
    feature_names: List[str] = []

    if timer is not None:
        timer.start("feature_label_prep")

    symbol_items = list(symbol_frames.items())
    symbol_results: List[tuple[str, SymbolData]] = []

    if parallel_prep:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _build_symbol_data_worker,
                    symbol,
                    df,
                    window_size,
                    atr_window,
                    std_window,
                    label_config,
                    label_log_transform,
                    str(cache_dir) if cache_dir is not None else None,
                )
                for symbol, df in symbol_items
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Prep features/labels", unit="symbol"):
                symbol_results.append(future.result())
    else:
        for symbol, df in tqdm(symbol_items, desc="Prep features/labels", unit="symbol"):
            symbol_results.append(
                _build_symbol_data_worker(
                    symbol,
                    df,
                    window_size,
                    atr_window,
                    std_window,
                    label_config,
                    label_log_transform,
                    str(cache_dir) if cache_dir is not None else None,
                )
            )

    label_samples: List[np.ndarray] = []
    for _, symbol_data in symbol_results:
        feature_names = symbol_data.feature_names
        label_samples.append(symbol_data.labels)

        n = len(symbol_data.features)
        train_idx, val_idx, test_idx = _split_indices(n, train_ratio, val_ratio)
        split_ranges = {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        }

        for split_name, idx_range in split_ranges.items():
            valid_indices = [i for i in idx_range if i >= window_size - 1]
            dataset = WindowDataset(symbol_data, valid_indices, window_size)
            if split_name == "train":
                train_sets.append(dataset)
            elif split_name == "val":
                val_sets.append(dataset)
            else:
                test_sets.append(dataset)

    if timer is not None:
        timer.stop("feature_label_prep")

    if label_samples:
        labels_all = np.concatenate(label_samples, axis=0)
        stats = {
            "min": float(np.min(labels_all)),
            "median": float(np.median(labels_all)),
            "p95": float(np.percentile(labels_all, 95)),
            "p99": float(np.percentile(labels_all, 99)),
            "max": float(np.max(labels_all)),
        }
        print("Label stats (post-transform)")
        for key, value in stats.items():
            print(f"{key}: {value:.6f}")

    return ConcatDataset(train_sets), ConcatDataset(val_sets), ConcatDataset(test_sets), feature_names
