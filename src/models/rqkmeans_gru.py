import torch.nn as nn


class RQKmeansPredictor(nn.Module):
    def __init__(self, num_codes: int = 33, embedding_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(num_codes, embedding_dim, padding_idx=32)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc_code1 = nn.Linear(hidden_dim, 32)
        self.fc_code2 = nn.Linear(hidden_dim, 32)

    def forward(self, x):
        embeds = self.embedding(x)
        _, hn = self.gru(embeds)
        last_hidden = hn[-1]
        return self.fc_code1(last_hidden), self.fc_code2(last_hidden)
