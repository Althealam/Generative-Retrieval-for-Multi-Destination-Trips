"""Trip-level side features aligned with trip aggregates (booker, device, stay, etc.)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_booker_device_vocabs(train_trips: pd.DataFrame) -> tuple[dict[str, int], dict[str, int], int, int]:
    countries = sorted(train_trips["booker_country"].dropna().astype(str).unique().tolist())
    devices = sorted(train_trips["device_class"].dropna().astype(str).unique().tolist())
    booker_to_idx = {c: i + 1 for i, c in enumerate(countries)}
    device_to_idx = {d: i + 1 for i, d in enumerate(devices)}
    return booker_to_idx, device_to_idx, len(countries), len(devices)


def _bucket_trip_len(n: int) -> int:
    return max(1, min(30, int(n)))


def _bucket_repeat_ratio(r: float) -> int:
    return max(1, min(10, int(r * 10) + 1))


def _compute_same_country_streak(countries: list[str]) -> int:
    if not countries:
        return 1
    last = countries[-1]
    streak = 1
    for i in range(len(countries) - 2, -1, -1):
        if countries[i] == last:
            streak += 1
        else:
            break
    return max(1, min(30, streak))


def row_to_context_indices(
    row: pd.Series,
    booker_to_idx: dict[str, int],
    device_to_idx: dict[str, int],
    *,
    prefix_len: int | None = None,
) -> tuple[int, int, int, int, int, int, int, int, int]:
    booker = booker_to_idx.get(str(row["booker_country"]), 0) if pd.notna(row["booker_country"]) else 0
    device = device_to_idx.get(str(row["device_class"]), 0) if pd.notna(row["device_class"]) else 0

    m = row["checkin_month"]
    if pd.isna(m):
        month_idx = 0
    else:
        mi = int(m)
        month_idx = mi if 1 <= mi <= 12 else 0

    durs_raw = row["stay_duration"]
    if isinstance(durs_raw, (list, np.ndarray)):
        durs = list(durs_raw)
    else:
        durs = []

    cities_raw = row["city_id"] if isinstance(row["city_id"], (list, np.ndarray)) else []
    cities = list(cities_raw)

    countries_raw = row["hotel_country"] if isinstance(row.get("hotel_country"), list) else []
    countries = [str(c) for c in countries_raw if pd.notna(c)]

    # Prevent future leakage: use only prefix information for this sample.
    if prefix_len is None:
        prefix_len = len(cities)
    prefix_len = max(1, min(int(prefix_len), len(cities) if len(cities) > 0 else 1))

    cities_prefix = cities[:prefix_len]
    durs_prefix = durs[:prefix_len]
    countries_prefix = countries[:prefix_len]

    if len(durs_prefix) > 0:
        mean_stay = int(round(float(np.mean(durs_prefix))))
    else:
        mean_stay = 1
    mean_stay = max(1, min(30, mean_stay))
    stay_idx = mean_stay

    trip_len = len(cities_prefix)
    num_unique = len(set(cities_prefix)) if trip_len > 0 else 1
    repeat_ratio = 1.0 - (num_unique / trip_len) if trip_len > 0 else 0.0

    trip_len_idx = _bucket_trip_len(trip_len if trip_len > 0 else 1)
    num_unique_idx = _bucket_trip_len(num_unique)
    repeat_ratio_idx = _bucket_repeat_ratio(repeat_ratio)

    if len(durs_prefix) > 0:
        last_stay = int(durs_prefix[-1])
    else:
        last_stay = 1
    last_stay_idx = max(1, min(30, last_stay))

    same_country_streak_idx = _compute_same_country_streak(countries_prefix)

    return (
        booker,
        device,
        month_idx,
        stay_idx,
        trip_len_idx,
        num_unique_idx,
        repeat_ratio_idx,
        last_stay_idx,
        same_country_streak_idx,
    )
