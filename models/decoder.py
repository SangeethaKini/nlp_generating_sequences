# Sinthuja Santhiran 398419 start- Decoder for sentence VAE

import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, latent_size, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True)

        self.latent_to_hidden = nn.Linear(latent_size, hidden_size)
        self.latent_to_cell = nn.Linear(latent_size, hidden_size)

        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, z, x):
        x = self.embedding(x)

        h = torch.tanh(self.latent_to_hidden(z)).unsqueeze(0)
        c = torch.tanh(self.latent_to_cell(z)).unsqueeze(0)

        out, (h, c) = self.lstm(x, (h, c))

        prediction = self.output(out)

        return prediction

    def init_state(self, z):
    h = torch.tanh(self.latent_to_hidden(z)).unsqueeze(0)
    c = torch.tanh(self.latent_to_cell(z)).unsqueeze(0)
    return h, c
    
    def step(self, prev_token, state):
    h, c = state
    x = self.embedding(prev_token).unsqueeze(1)
    out, (h, c) = self.lstm(x, (h, c))
    logits = self.output(out.squeeze(1))
    return logits, (h, c)


    @torch.no_grad()
    def generate(self, z, bos_idx, eos_idx, max_len):
        self.eval()
        batch_size = z.size(0)

        h = torch.tanh(self.latent_to_hidden(z)).unsqueeze(0)
        c = torch.tanh(self.latent_to_cell(z)).unsqueeze(0)

        token = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=z.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=z.device)
        generated = []

        for _ in range(max_len):
            x = self.embedding(token)
            out, (h, c) = self.lstm(x, (h, c))
            logits = self.output(out.squeeze(1))
            token = logits.argmax(dim=-1)

            token = torch.where(finished, torch.full_like(token, eos_idx), token)
            generated.append(token)
            finished = finished | (token == eos_idx)

            token = token.unsqueeze(1)
            if finished.all():
                break

        return torch.stack(generated, dim=1)


def ids_to_sentences(token_ids, index_to_word_list, eos_idx):
    # token_ids: (batch_size, seq_len) tensor of predicted word indices, e.g. from Decoder.generate()
    # Converts each row back into a space-joined sentence, stopping at <eos>
    sentences = []
    for row in token_ids.tolist():
        words = []
        for idx in row:
            if idx == eos_idx:
                break
            words.append(index_to_word_list[idx])
        sentences.append(" ".join(words))
    return sentences
# Sinthuja Santhiran 398419 end
