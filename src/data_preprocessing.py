"""Data loading and preprocessing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

URL_COLUMN_CANDIDATES = ("url", "link", "website", "domain", "address")
LABEL_COLUMN_CANDIDATES = ("label", "class", "target", "result", "is_phishing", "phishing")

PHISHING_LABELS = {"1", "phishing", "malicious", "fraud", "true", "yes", "bad", "spam"}
LEGIT_LABELS = {"0", "legitimate", "benign", "false", "no", "good", "safe", "ham", "normal"}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_dataset_path(data_path: str | Path | None = None, data_dir: str | Path | None = None) -> Path:
    if data_path is not None:
        path = Path(data_path)
        if not path.is_absolute():
            path = _project_root() / path
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        return path

    root = Path(data_dir) if data_dir else _project_root() / "data" / "raw"
    if not root.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {root}")

    csv_files = list(root.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV dataset found under: {root}")

    def priority(path: Path) -> tuple[int, int, str]:
        name = path.name.lower()
        score = 0
        if "phish" in name:
            score += 3
        if "url" in name:
            score += 2
        if "dataset" in name:
            score += 1
        return (-score, len(name), name)

    csv_files.sort(key=priority)
    return csv_files[0]


def _find_column(columns: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    original = list(columns)
    lookup = {column.lower(): column for column in original}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]

    for column in original:
        lowered = column.lower()
        if any(candidate in lowered for candidate in candidates):
            return column
    return None


def normalize_label(value: object) -> int | None:
    """Maps mixed labels to binary classes: legitimate=0, phishing=1."""
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return 1 if int(value) == 1 else 0 if int(value) == 0 else None
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        return 1 if int(value) == 1 else 0 if int(value) == 0 else None

    lowered = str(value).strip().lower()
    if lowered in PHISHING_LABELS or "phish" in lowered:
        return 1
    if lowered in LEGIT_LABELS or "legit" in lowered:
        return 0
    return None


def load_raw_dataset(
    data_path: str | Path | None = None,
    data_dir: str | Path | None = None,
    return_path: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, Path]:
    """Loads, validates, and cleans a raw phishing dataset."""
    resolved_path = _resolve_dataset_path(data_path=data_path, data_dir=data_dir)
    raw = pd.read_csv(resolved_path, on_bad_lines="skip")

    url_column = _find_column(raw.columns, URL_COLUMN_CANDIDATES)
    label_column = _find_column(raw.columns, LABEL_COLUMN_CANDIDATES)
    if url_column is None or label_column is None:
        raise KeyError(
            "Dataset must include URL and label columns. "
            f"Detected columns: {list(raw.columns)}"
        )

    cleaned = raw[[url_column, label_column]].copy()
    cleaned.columns = ["url", "label"]
    cleaned["url"] = cleaned["url"].astype(str).str.strip()
    cleaned = cleaned[cleaned["url"] != ""]
    cleaned["label"] = cleaned["label"].map(normalize_label)
    cleaned = cleaned.dropna(subset=["label"])
    cleaned["label"] = cleaned["label"].astype(int)
    cleaned = cleaned.drop_duplicates(subset=["url"]).reset_index(drop=True)

    if cleaned.empty:
        raise ValueError(f"Dataset became empty after cleaning: {resolved_path}")
    if cleaned["label"].nunique() < 2:
        raise ValueError("Dataset must contain both legitimate and phishing rows.")

    if return_path:
        return cleaned, resolved_path
    return cleaned
