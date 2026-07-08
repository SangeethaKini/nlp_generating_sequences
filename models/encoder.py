# Kavya Itikala Gopichand 401540
# Encoder for sentence VAE

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, latent_size, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True)

        self.mu = nn.Linear(hidden_size, latent_size)
        self.logvar = nn.Linear(hidden_size, latent_size)

    def forward(self, x):
        x = self.embedding(x)

        out, (h, c) = self.lstm(x)

        h = h[-1]

        mu = self.mu(h)
        logvar = self.logvar(h)

        return mu, logvar

# Kavya Itikala Gopichand 401540
