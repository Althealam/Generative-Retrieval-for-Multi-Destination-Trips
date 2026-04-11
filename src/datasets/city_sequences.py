from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.datasets.tokens import PAD_TOKEN_ID, UNK_TOKEN_ID


class CitySequenceDataset(Dataset):
    def __init__(
        self,
        x_values: list[list[int]],
        y_values: list[int] | None = None,
        *,
        ctx_booker: list[int] | None = None,
        ctx_device: list[int] | None = None,
        ctx_month: list[int] | None = None,
        ctx_stay: list[int] | None = None,
    ):
        self.x_values = [torch.tensor(x, dtype=torch.long) for x in x_values]
        self.y_values = torch.tensor(y_values, dtype=torch.long) if y_values is not None else None
        self.use_ctx = (
            ctx_booker is not None
            and ctx_device is not None
            and ctx_month is not None
            and ctx_stay is not None
            and len(ctx_booker) == len(x_values)
        )
        if self.use_ctx:
            self.ctx_booker = torch.tensor(ctx_booker, dtype=torch.long)
            self.ctx_device = torch.tensor(ctx_device, dtype=torch.long)
            self.ctx_month = torch.tensor(ctx_month, dtype=torch.long)
            self.ctx_stay = torch.tensor(ctx_stay, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.x_values)

    def __getitem__(self, idx: int):
        if self.y_values is None:
            if self.use_ctx:
                return (
                    self.x_values[idx],
                    self.ctx_booker[idx],
                    self.ctx_device[idx],
                    self.ctx_month[idx],
                    self.ctx_stay[idx],
                )
            return (self.x_values[idx],)
        if self.use_ctx:
            return (
                self.x_values[idx],
                self.y_values[idx],
                self.ctx_booker[idx],
                self.ctx_device[idx],
                self.ctx_month[idx],
                self.ctx_stay[idx],
            )
        return self.x_values[idx], self.y_values[idx]


def collate_city_batch(batch):
    first = batch[0]
    if not isinstance(first, tuple):
        return pad_sequence(batch, batch_first=True, padding_value=PAD_TOKEN_ID)

    n_fields = len(first)
    if n_fields == 1:
        xs = [b[0] for b in batch]
        return pad_sequence(xs, batch_first=True, padding_value=PAD_TOKEN_ID)
    if n_fields == 2:
        xs, ys = zip(*batch)
        xs_padded = pad_sequence(xs, batch_first=True, padding_value=PAD_TOKEN_ID)
        return xs_padded, torch.stack(ys)
    if n_fields == 5:
        xs, bs, ds, ms, ss = zip(*batch)
        xs_padded = pad_sequence(xs, batch_first=True, padding_value=PAD_TOKEN_ID)
        return xs_padded, torch.stack(bs), torch.stack(ds), torch.stack(ms), torch.stack(ss)
    if n_fields == 6:
        xs, ys, bs, ds, ms, ss = zip(*batch)
        xs_padded = pad_sequence(xs, batch_first=True, padding_value=PAD_TOKEN_ID)
        return xs_padded, torch.stack(ys), torch.stack(bs), torch.stack(ds), torch.stack(ms), torch.stack(ss)
    raise ValueError(f"Unexpected batch tuple length {n_fields}")


def build_city_dataloaders(
    train_x: list[list[int]],
    train_y: list[int],
    test_x: list[list[int]],
    batch_size: int = 256,
    *,
    train_ctx: tuple[list[int], list[int], list[int], list[int]] | None = None,
    test_ctx: tuple[list[int], list[int], list[int], list[int]] | None = None,
) -> tuple[DataLoader, DataLoader]:
    if train_ctx is not None and test_ctx is not None:
        tb, td, tm, ts = train_ctx
        eb, ed, em, es = test_ctx
        train_ds = CitySequenceDataset(
            train_x, train_y, ctx_booker=tb, ctx_device=td, ctx_month=tm, ctx_stay=ts
        )
        test_ds = CitySequenceDataset(test_x, ctx_booker=eb, ctx_device=ed, ctx_month=em, ctx_stay=es)
    else:
        train_ds = CitySequenceDataset(train_x, train_y)
        test_ds = CitySequenceDataset(test_x)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_city_batch
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_city_batch
    )
    return train_loader, test_loader
