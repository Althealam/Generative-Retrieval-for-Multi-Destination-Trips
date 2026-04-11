"""City popularity from training rows (check-in level counts)."""

from collections import Counter

import pandas as pd


def top_city_ids_from_train(train_set: pd.DataFrame, k: int = 4) -> list[int]:
    counts = Counter(train_set["city_id"].tolist())
    return [city_id for city_id, _ in counts.most_common(k)]
