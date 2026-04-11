"""Trip-level side features aligned with create_mutliple_sequences aggregates."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_booker_device_vocabs(train_trips: pd.DataFrame) -> tuple[dict[str, int], dict[str, int], int, int]:
    countries = sorted(train_trips["booker_country"].dropna().astype(str).unique().tolist())
    devices = sorted(train_trips["device_class"].dropna().astype(str).unique().tolist())
    booker_to_idx = {c: i + 1 for i, c in enumerate(countries)}
    device_to_idx = {d: i + 1 for i, d in enumerate(devices)}
    return booker_to_idx, device_to_idx, len(countries), len(devices)


def row_to_context_indices(
    row: pd.Series, booker_to_idx: dict[str, int], device_to_idx: dict[str, int]
) -> tuple[int, int, int, int]:
    booker = booker_to_idx.get(str(row["booker_country"]), 0) if pd.notna(row["booker_country"]) else 0
    device = device_to_idx.get(str(row["device_class"]), 0) if pd.notna(row["device_class"]) else 0

    m = row["checkin_month"]
    if pd.isna(m):
        month_idx = 0
    else:
        mi = int(m)
        month_idx = mi if 1 <= mi <= 12 else 0

    durs = row["stay_duration"]
    if isinstance(durs, (list, np.ndarray)) and len(durs) > 0:
        mean_stay = int(round(float(np.mean(durs))))
    else:
        mean_stay = 1
    mean_stay = max(1, min(30, mean_stay))
    stay_idx = mean_stay

    return booker, device, month_idx, stay_idx
