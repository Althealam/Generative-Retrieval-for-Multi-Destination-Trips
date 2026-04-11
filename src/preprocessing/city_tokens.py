"""Build city-id token sequences (and optional context) before ``CitySequenceDataset``."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.datasets.tokens import UNK_TOKEN_ID
from src.preprocessing.trip_context import row_to_context_indices


@dataclass
class CitySequencePack:
    x: list[list[int]]
    y: list[int] | None
    ctx_booker: list[int]
    ctx_device: list[int]
    ctx_month: list[int]
    ctx_stay: list[int]


def build_city_vocab(train_set: pd.DataFrame) -> tuple[dict[int, int], dict[int, int]]:
    unique_cities = sorted(train_set["city_id"].unique().tolist())
    city_to_idx = {city_id: idx + 2 for idx, city_id in enumerate(unique_cities)}
    idx_to_city = {idx: city for city, idx in city_to_idx.items()}
    return city_to_idx, idx_to_city


def build_city_sequence_pack(
    trip_df: pd.DataFrame,
    city_to_idx: dict[int, int],
    *,
    is_test: bool,
    multi_step: bool,
    use_context: bool,
    booker_to_idx: dict[str, int] | None = None,
    device_to_idx: dict[str, int] | None = None,
) -> CitySequencePack:
    x_values: list[list[int]] = []
    y_values: list[int] | None = [] if not is_test else None
    cb: list[int] = []
    cd: list[int] = []
    cm: list[int] = []
    cs: list[int] = []

    for _, row in trip_df.iterrows():
        cities = row["city_id"]
        token_seq = [city_to_idx.get(city_id, UNK_TOKEN_ID) for city_id in cities]
        if use_context and booker_to_idx is not None and device_to_idx is not None:
            b, d, m, s = row_to_context_indices(row, booker_to_idx, device_to_idx)
        else:
            b = d = m = s = 0

        if is_test:
            x_values.append(token_seq[:-1])
            if use_context:
                cb.append(b)
                cd.append(d)
                cm.append(m)
                cs.append(s)
        elif y_values is not None:
            if multi_step:
                for t in range(1, len(token_seq)):
                    x_values.append(token_seq[:t])
                    y_values.append(token_seq[t])
                    if use_context:
                        cb.append(b)
                        cd.append(d)
                        cm.append(m)
                        cs.append(s)
            else:
                if len(token_seq) >= 2:
                    x_values.append(token_seq[:-1])
                    y_values.append(token_seq[-1])
                    if use_context:
                        cb.append(b)
                        cd.append(d)
                        cm.append(m)
                        cs.append(s)

    return CitySequencePack(
        x=x_values,
        y=y_values,
        ctx_booker=cb,
        ctx_device=cd,
        ctx_month=cm,
        ctx_stay=cs,
    )
