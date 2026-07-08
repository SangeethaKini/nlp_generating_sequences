# losses.py - Madhumitha
import torch from torch import nn


def reconstruction_loss(logits, targets, pad_idx):
    """
    Reconstruction part of the VAE loss.
    This is just cross-entropy over the predicted words.
    We ignore padding tokens so they do not affect the loss.
    """
    batch_size, seq_len, vocab_size = logits.size()

    logits_flat = logits.view(batch_size * seq_len, vocab_size)
    targets_flat = targets.view(batch_size * seq_len)

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
    return loss_fn(logits_flat, targets_flat)


def kl_loss(mu, logvar):
    """
    KL divergence between q(z|x) and the standard normal prior.
    This is the regularization term used in the paper.
    """
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl.mean()


def kl_annealing_weight(step, total_anneal_steps=10000):
    """
    At the beginning we keep KL small, then slowly increase it.
    This follows the KL annealing idea from the paper.
    """
    if total_anneal_steps <= 0:
        return 1.0

    return min(1.0, step / total_anneal_steps)


def word_dropout(input_seq, dropout_prob, unk_idx, pad_idx=None):
    """
    Some decoder input words are replaced with <unk>.
    This makes the decoder rely more on the latent variable z.
    """
    if dropout_prob <= 0.0:
        return input_seq

    mask = torch.rand_like(input_seq, dtype=torch.float) < dropout_prob

    if pad_idx is not None:
        pad_mask = (input_seq == pad_idx)
        mask = mask & (~pad_mask)

    dropped = input_seq.clone()
    dropped[mask] = unk_idx
    return dropped


def vae_loss(logits, targets, mu, logvar, pad_idx, kl_weight):
    """
    Full VAE loss used in training.

    total loss = reconstruction loss + KL weight * KL loss
    """
    recon = reconstruction_loss(logits, targets, pad_idx)
    kl = kl_loss(mu, logvar)
    loss = recon + kl_weight * kl

    return loss, recon, kl
