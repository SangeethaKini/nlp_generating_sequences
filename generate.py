
from __future__ import annotations
import torch
import torch.nn.functional as F
# Low-level greedy decode from a latent vector z

@torch.no_grad()
def greedy_decode(decoder, z, vocab, max_len=30):
    """Deterministically decode a sentence from latent z. z: [B, z_dim]."""
    decoder.eval()
    B = z.size(0)
    device = z.device
    state = decoder.init_state(z)
    prev = torch.full((B,), vocab.sos_id, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    seqs = [[] for _ in range(B)]
    for _ in range(max_len):
        logits, state = decoder.step(prev, state)
        prev = logits.argmax(-1)
        for b in range(B):
            if not finished[b]:
                tok = int(prev[b])
                if tok == vocab.eos_id:
                    finished[b] = True
                else:
                    seqs[b].append(tok)
        if finished.all():
            break
    return [vocab.decode(s) for s in seqs]

# 1. Reconstruction: encode a real sentence, decode from posterior mean
#    (Table 7 "mean" row) and from posterior samples (Table 7 "samp" rows)
@torch.no_grad()
def reconstruct(encoder, decoder, reparameterize, batch, vocab,
                n_samples=0, max_len=30):
    encoder.eval(); decoder.eval()
    mu, logvar = encoder(batch)
    results = {"mean": greedy_decode(decoder, mu, vocab, max_len)}
    for s in range(n_samples):
        z = reparameterize(mu, logvar)
        results[f"samp_{s+1}"] = greedy_decode(decoder, z, vocab, max_len)
    return results

# 2. Sample from the Gaussian prior and greedily decode  (Table 5)
@torch.no_grad()
def sample_prior(decoder, vocab, z_dim, n=5, max_len=30, device="cpu"):
    z = torch.randn(n, z_dim, device=device)
    return greedy_decode(decoder, z, vocab, max_len)
# 3. Latent interpolation / homotopy between two sentences  (Tables 6 & 8)
#    z(t) = z1*(1-t) + z2*t  for t in [0,1]
@torch.no_grad()
def homotopy(encoder, decoder, batch_pair, vocab, steps=7, max_len=30):
    """batch_pair['input'] must contain exactly 2 sentences (the endpoints)."""
    encoder.eval(); decoder.eval()
    mu, _ = encoder(batch_pair)
    z1, z2 = mu[0], mu[1]
    ts = torch.linspace(0, 1, steps, device=mu.device)
    zs = torch.stack([z1 * (1 - t) + z2 * t for t in ts], dim=0)
    lines = greedy_decode(decoder, zs, vocab, max_len)
    return list(zip([round(float(t), 3) for t in ts], lines))
@torch.no_grad()
def impute(encoder, decoder, reparameterize, batch, vocab,
           frac=0.2, max_len=30):
    encoder.eval(); decoder.eval()
    mu, logvar = encoder(batch)
    z = reparameterize(mu, logvar)

    x = batch["input"] if isinstance(batch, dict) else batch  # [B, T]
    B, T = x.size()
    device = x.device
    outputs = []
    for b in range(B):
        # locate real (non-pad) length, excluding EOS
        row = x[b].tolist()
        length = len(row)
        if vocab.eos_id in row:
            length = row.index(vocab.eos_id)
        n_obs = max(1, int(length * (1 - frac)))
        prefix = row[:n_obs]                    # includes SOS at row[0]

        state = decoder.init_state(z[b:b + 1])
        # feed the observed prefix through the decoder to build state
        prev = torch.tensor([prefix[0]], device=device)
        for tok in prefix[1:]:
            _, state = decoder.step(prev, state)
            prev = torch.tensor([tok], device=device)
        # now generate the imputed tail
        tail = []
        for _ in range(max_len):
            logits, state = decoder.step(prev, state)
            nxt = int(logits.argmax(-1))
            if nxt == vocab.eos_id:
                break
            tail.append(nxt)
            prev = torch.tensor([nxt], device=device)
        outputs.append({
            "observed": vocab.decode(prefix),
            "imputed": vocab.decode(tail),
            "true": vocab.decode(row[n_obs:length]),
        })
    return outputs

# 5. Inputless-decoder generation (word_keep = 0)  (Section 4 / Table 5)
#    Here the decoder receives no previous words; everything must come from z.
#    We reuse greedy_decode but the decoder was *trained* with word_keep=0.
@torch.no_grad()
def inputless_sample(decoder, vocab, z_dim, n=5, max_len=30, device="cpu"):
    """Same call as sample_prior; the difference lives in how the decoder was
    trained. Provided as a named function for clarity in evaluate.py."""
    z = torch.randn(n, z_dim, device=device)
    return greedy_decode(decoder, z, vocab, max_len)
<<<<<<< HEAD

#Sri Vyshnvi Madala 393232
=======
>>>>>>> 3487052 (Add files via upload)
