from __future__ import annotations

import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd
from tqdm import tqdm

from timing import StageTimer


REQUIRED_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]


def load_symbol_csv(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        first_line = handle.readline().strip()
    if first_line.startswith("version https://git-lfs.github.com/spec/v1"):
        raise ValueError(f"{path.name} appears to be a Git LFS pointer, not data.")

    df = pd.read_csv(path, low_memory=False)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {missing}")
    df = df[REQUIRED_COLUMNS].copy()
    numeric_cols = [col for col in df.columns if col != "ignore"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True, errors="coerce")
    df = df.dropna(subset=["open_time", "open", "high", "low", "close", "volume"])
    df = df.sort_values("open_time").set_index("open_time")
    return df


def resample_to_5m(df: pd.DataFrame) -> pd.DataFrame:
    agg_map = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "close_time": "last",
        "quote_volume": "sum",
        "count": "sum",
        "taker_buy_volume": "sum",
        "taker_buy_quote_volume": "sum",
        "ignore": "last",
    }
    resampled = df.resample("5min", label="right", closed="right").agg(agg_map)
    resampled = resampled.dropna(subset=["open", "high", "low", "close", "volume"])
    return resampled


def _load_symbol_from_path(path_str: str) -> tuple[str, pd.DataFrame]:
    path = Path(path_str)
    symbol = path.stem
    df = load_symbol_csv(path)
    df_5m = resample_to_5m(df)
    df_5m["symbol"] = symbol
    return symbol, df_5m


def load_all_symbols(
    data_dir: Path,
    parallel: bool = False,
    max_workers: int | None = None,
    timer: StageTimer | None = None,
) -> Dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    csv_paths = sorted(data_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    symbol_data: Dict[str, pd.DataFrame] = {}
    skipped: list[str] = []
    if timer is not None:
        timer.start("data_load_resample")

    def add_symbol(symbol: str, df_5m: pd.DataFrame) -> None:
        symbol_data[symbol] = df_5m

    if parallel:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_load_symbol_from_path, str(path)) for path in csv_paths]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Load+resample", unit="symbol"):
                try:
                    symbol, df_5m = future.result()
                    add_symbol(symbol, df_5m)
                except ValueError as exc:
                    skipped.append(str(exc))
                    print(f"Skipping file due to error: {exc}")
    else:
        for path in tqdm(csv_paths, desc="Load+resample", unit="symbol"):
            try:
                symbol, df_5m = _load_symbol_from_path(str(path))
                add_symbol(symbol, df_5m)
            except ValueError as exc:
                skipped.append(str(exc))
                print(f"Skipping file due to error: {exc}")

    if timer is not None:
        timer.stop("data_load_resample")

    if not symbol_data:
        unique_errors = "\n".join(sorted(set(skipped)))
        raise ValueError(
            "No valid CSV files loaded. If files are Git LFS pointers, run `git lfs pull`.\n"
            f"Errors:\n{unique_errors}"
        )

    return symbol_data


def compute_data_hash(files: Iterable[Path]) -> str:
    hasher = hashlib.sha256()
    for path in sorted(Path(p).resolve() for p in files):
        stat = path.stat()
        payload = f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}".encode("utf-8")
        hasher.update(payload)
    return hasher.hexdigest()


def list_data_files(data_dir: Path) -> Tuple[Path, ...]:
    return tuple(sorted(Path(data_dir).glob("*.csv")))
