import math

import torch
import torch.nn as nn

from src.models.positional import PositionalEncoding


class RQKMeansTransformer(nn.Module):
    def __init__(
        self,
        num_codes: int = 33,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        max_len: int = 100,
        codebook_size: int = 32,
        pad_code: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_code = pad_code
        self.codebook_size = codebook_size

        self.embedding = nn.Embedding(num_codes, d_model, padding_idx=pad_code)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer_block = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

        self.fc_code1 = nn.Linear(d_model, codebook_size)
        self.fc_code2 = nn.Linear(d_model, codebook_size)

    def _generate_causal_mask(self, sz: int, device):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask.to(device)

    def forward(self, x):
        _, seq_len = x.size()
        device = x.device

        padding_mask = x == self.pad_code
        causal_mask = self._generate_causal_mask(seq_len, device)

        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        output = self.transformer_block(x, mask=causal_mask, src_key_padding_mask=padding_mask)

        last_hidden = output[:, -1, :]
        return self.fc_code1(last_hidden), self.fc_code2(last_hidden)
