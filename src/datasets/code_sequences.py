"""Code-token sequences for RQ-KMeans / RQVAE (quantized city codes).

Padding must match the model:
- RQ-KMeans K=32: codes 0..31, pad ``32``.
- RQVAE K=128: codes 0..127, pad ``codebook_size`` (128).
"""

from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

DEFAULT_CODE_PAD_TOKEN = 32


class CityCodeDataset(Dataset):
    def __init__(self, x_values: list[list[int]], y_values: list[list[int]] | None = None):
        self.x_values = [torch.tensor(x, dtype=torch.long) for x in x_values]
        self.y_values = torch.tensor(y_values, dtype=torch.long) if y_values is not None else None

    def __len__(self) -> int:
        return len(self.x_values)

    def __getitem__(self, idx: int):
        if self.y_values is not None:
            return self.x_values[idx], self.y_values[idx]
        return self.x_values[idx]


def _make_collate_code(pad_token: int):
    def collate_fn(batch):
        if isinstance(batch[0], tuple):
            xs, ys = zip(*batch)
            xs_padded = pad_sequence(xs, batch_first=True, padding_value=pad_token)
            return xs_padded, torch.stack(ys)
        return pad_sequence(batch, batch_first=True, padding_value=pad_token)

    return collate_fn


def build_dataloaders(
    train_x: list[list[int]],
    train_y: list[list[int]],
    test_x: list[list[int]],
    batch_size: int = 256,
    pad_token: int = DEFAULT_CODE_PAD_TOKEN,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = CityCodeDataset(train_x, train_y)
    test_dataset = CityCodeDataset(test_x)
    collate_fn = _make_collate_code(pad_token)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    return train_loader, test_loader
