from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

# EXACT tokens + order from preprocessing_data.py (Javeria)
PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]            # ids: pad=0, bos=1, eos=2, unk=3

# VocabAdapter: wraps the repo's (index_to_word_list, word_to_index_dict)

class VocabAdapter:
    """Thin convenience wrapper around the repo's two vocab structures.

    Build from the real preprocessing output:
        vocab = VocabAdapter(index_to_word_list, word_to_index_dict)
    or, for standalone testing:
        vocab = VocabAdapter.from_sentences(list_of_token_lists)
    """

    def __init__(self, index_to_word_list, word_to_index_dict):
        self.itos = index_to_word_list
        self.stoi = word_to_index_dict

    @classmethod
    def from_sentences(cls, sentences, vocab_size=10000):
        from collections import Counter
        counts = Counter(w for s in sentences for w in s)
        itos = list(SPECIALS) + [w for w, _ in counts.most_common(vocab_size)]
        stoi = {w: i for i, w in enumerate(itos)}
        return cls(itos, stoi)

    def __len__(self):
        return len(self.itos)

    @property
    def pad_id(self): return self.stoi[PAD]
    @property
    def bos_id(self): return self.stoi[BOS]
    @property
    def eos_id(self): return self.stoi[EOS]
    @property
    def unk_id(self): return self.stoi[UNK]

    # Kept as sos_id alias so generic generation code reads naturally.
    @property
    def sos_id(self): return self.bos_id

    def encode(self, words, add_bos_eos=True):
        ids = [self.stoi.get(w, self.unk_id) for w in words]
        if add_bos_eos:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids, strip_special=True):
        out = []
        for i in ids:
            i = int(i)
            tok = self.itos[i] if 0 <= i < len(self.itos) else UNK
            if strip_special and tok in (PAD, BOS):
                continue
            if strip_special and tok == EOS:
                break
            out.append(tok)
        return " ".join(out)

# build_batch: numericalise + pad a list of token-lists -> {"input": LongTensor}
# (This is the missing data->tensor step; hand to Person 1/5 for the DataLoader.)
def build_batch(sentences, vocab, max_len=17, device="cpu"):
    ids = [vocab.encode(s)[:max_len] for s in sentences]
    T = max(len(x) for x in ids)
    padded = [x + [vocab.pad_id] * (T - len(x)) for x in ids]
    return {"input": torch.tensor(padded, dtype=torch.long, device=device)}

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hid_dim=256, z_dim=32, pad_id=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(emb_dim, hid_dim, batch_first=True)
        self.to_mu = nn.Linear(hid_dim, z_dim)
        self.to_logvar = nn.Linear(hid_dim, z_dim)

    def forward(self, batch):
        x = batch["input"] if isinstance(batch, dict) else batch
        _, (h, _) = self.lstm(self.embed(x))
        h = h[-1]
        return self.to_mu(h), self.to_logvar(h)

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hid_dim=256, z_dim=32,
                 pad_id=0, unk_id=3):
        super().__init__()
        self.pad_id, self.unk_id = pad_id, unk_id
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.z_to_h = nn.Linear(z_dim, hid_dim)
        self.z_to_c = nn.Linear(z_dim, hid_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, batch_first=True)
        self.out = nn.Linear(hid_dim, vocab_size)

    def forward(self, z, batch, word_keep=1.0):
        tgt = batch["input"] if isinstance(batch, dict) else batch
        dec_in = tgt[:, :-1]
        if word_keep < 1.0:
            mask = torch.rand_like(dec_in.float()) < word_keep
            dec_in = torch.where(mask, dec_in, torch.full_like(dec_in, self.unk_id))
        out, _ = self.lstm(self.embed(dec_in), self.init_state(z))
        return self.out(out)

    def init_state(self, z):
        h0 = torch.tanh(self.z_to_h(z)).unsqueeze(0)
        c0 = torch.tanh(self.z_to_c(z)).unsqueeze(0)
        return h0, c0

    def step(self, prev_token, state):
        emb = self.embed(prev_token).unsqueeze(1)
        out, state = self.lstm(emb, state)
        return self.out(out.squeeze(1)), state


class SentenceVAE(nn.Module):
    def __init__(self, vocab, emb_dim=128, hid_dim=256, z_dim=32):
        super().__init__()
        self.vocab = vocab
        V = len(vocab)
        self.encoder = Encoder(V, emb_dim, hid_dim, z_dim, vocab.pad_id)
        self.decoder = Decoder(V, emb_dim, hid_dim, z_dim, vocab.pad_id, vocab.unk_id)
        self.z_dim = z_dim

    def forward(self, batch, word_keep=1.0):
        mu, logvar = self.encoder(batch)
        z = reparameterize(mu, logvar)
        logits = self.decoder(z, batch, word_keep=word_keep)
        return logits, mu, logvar, z

def vae_loss(logits, batch, mu, logvar, kl_weight=1.0, pad_id=0):
    tgt = (batch["input"] if isinstance(batch, dict) else batch)[:, 1:]
    B = tgt.size(0)
    recon = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
        ignore_index=pad_id, reduction="sum",
    ) / B
    kl = kl_divergence(mu, logvar).mean()
    return recon + kl_weight * kl, {"recon": recon.item(), "kl": kl.item()}
