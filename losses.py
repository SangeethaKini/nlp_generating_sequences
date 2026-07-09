# losses.py - Madhumitha Mahendiran (391379)
import torch
from torch import nn

# Cross-entropy loss for the predicted words. Padding tokens are ignored.
def reconstruction_loss(logits, targets, pad_idx):
    batch_size, seq_len, vocab_size = logits.size()

    logits_flat = logits.view(batch_size * seq_len, vocab_size)
    targets_flat = targets.view(batch_size * seq_len)

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
    return loss_fn(logits_flat, targets_flat)

 # Start with a small KL weight and increase it slowly.
def kl_annealing_weight(step, total_anneal_steps=10000):
    if total_anneal_steps <= 0:
        return 1.0

    return min(1.0, step / total_anneal_steps)

# Replace some input words with <unk> so the decoder uses z more.
def word_dropout(input_seq, dropout_prob, unk_idx, pad_idx=None):
    if dropout_prob <= 0.0:
        return input_seq

    mask = torch.rand_like(input_seq, dtype=torch.float) < dropout_prob

    if pad_idx is not None:
        pad_mask = (input_seq == pad_idx)
        mask = mask & (~pad_mask)

    dropped = input_seq.clone()
    dropped[mask] = unk_idx
    return dropped

# Total loss = reconstruction loss + KL weight * KL loss
def vae_loss(recon, kl, kl_weight):
    return recon + kl_weight * kl
# losses.py - Madhumitha Mahendiran (391379)
