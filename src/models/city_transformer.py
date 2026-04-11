import math

import torch
import torch.nn as nn

from src.models.positional import PositionalEncoding


class CityTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int = 0,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 256,
        *,
        use_trip_context: bool = False,
        n_booker_countries: int = 0,
        n_device_classes: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.use_trip_context = use_trip_context

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if use_trip_context:
            self.emb_booker = nn.Embedding(n_booker_countries + 1, 64, padding_idx=0)
            self.emb_device = nn.Embedding(n_device_classes + 1, 48, padding_idx=0)
            self.emb_month = nn.Embedding(13, 32)
            self.emb_stay = nn.Embedding(31, 48)
            ctx_dim = 64 + 48 + 32 + 48
            self.ctx_proj = nn.Linear(ctx_dim, d_model)
        else:
            self.emb_booker = self.emb_device = self.emb_month = self.emb_stay = None
            self.ctx_proj = None

        self.classifier = nn.Linear(d_model, vocab_size)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    def forward(
        self,
        x: torch.Tensor,
        booker_idx: torch.Tensor | None = None,
        device_idx: torch.Tensor | None = None,
        month_idx: torch.Tensor | None = None,
        stay_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        padding_mask = x.eq(self.pad_token_id)
        causal_mask = self._generate_causal_mask(x.size(1), x.device)

        h = self.embedding(x) * math.sqrt(self.d_model)
        h = self.pos_encoder(h)
        h = self.transformer(h, mask=causal_mask, src_key_padding_mask=padding_mask)
        last_hidden = h[:, -1, :]

        if self.use_trip_context and self.ctx_proj is not None:
            assert booker_idx is not None and device_idx is not None
            assert month_idx is not None and stay_idx is not None
            ctx = torch.cat(
                [
                    self.emb_booker(booker_idx),
                    self.emb_device(device_idx),
                    self.emb_month(month_idx),
                    self.emb_stay(stay_idx),
                ],
                dim=-1,
            )
            last_hidden = last_hidden + self.ctx_proj(ctx)

        return self.classifier(last_hidden)
