#train.py
# Manisha Shekhar Mirje - 402237

import torch
import torch.nn.functional as F
import random

from model import kl_divergence

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
WORD_DROPOUT_PROB = 0.25

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "vae_checkpoint.pt"
# 

def make_batches(sentences, word_to_index, batch_size, pad_idx, bos_idx, eos_idx, unk_idx, shuffle=True):
    """
    Turns a list of tokenized sentences into padded (encoder_input, decoder_input, target) tensors.
    Kept inline here (instead of a separate dataset.py) so train.py is self-contained.
    """
    indices = list(range(len(sentences)))
    if shuffle:
        random.shuffle(indices)
 
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        encoded = []
        for i in batch_idx:
            ids = [word_to_index.get(t, unk_idx) for t in sentences[i]]
            encoder_input = ids + [eos_idx]        # w1 ... wn <eos>
            decoder_input = [bos_idx] + ids        # <bos> w1 ... wn
            target = ids + [eos_idx]               # w1 ... wn <eos>
            encoded.append((encoder_input, decoder_input, target))
 
        max_len = max(len(e[0]) for e in encoded)
 
        def pad(seq):
            return seq + [pad_idx] * (max_len - len(seq))
 
        encoder_batch = torch.tensor([pad(e[0]) for e in encoded], dtype=torch.long)
        decoder_batch = torch.tensor([pad(e[1]) for e in encoded], dtype=torch.long)
        target_batch = torch.tensor([pad(e[2]) for e in encoded], dtype=torch.long)
 
        yield encoder_batch, decoder_batch, target_batch


def apply_word_dropout(decoder_input, prob, unk_idx, pad_idx):
    """
    Word dropout weakens the decoder by replacing some input words with <unk>
    This encourages the decoder to use the latent vector z instead of relying only on previous words.
    Encoder inputs and target sentences are not changed.
    """
    if prob <= 0:
        return decoder_input
    mask = torch.rand_like(decoder_input, dtype=torch.float) < prob
    mask[:, 0] = False   #keep <bos>;decoder needs to know where a sentence starts
    mask = mask & (decoder_input != pad_idx)  #no padding to dropping, is ignored
    dropped = decoder_input.clone()
    dropped[mask] = unk_idx
    return dropped


def kl_weight_schedule(epoch, total_epochs):
    """
    KL annealing delays the KL penalty so the decoder learns to use the latent vector
    The KL weight gradually increases from 0 to 1 during training. This helps prevent posterior collapse
    """
    # increase KL weight from 0 to 1 over first half of training
    if total_epochs <= 0:
        return 1.0
    weight = epoch / (total_epochs * 0.5)
    return float(min(1.0, max(0.0, weight)))


def train(model, train_sentences, word_to_index, index_to_word, pad_idx, unk_idx, bos_idx, eos_idx):
    
    model = model.to(DEVICE)
    # optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    vocab_size = len(index_to_word)

    #Training loop - Epoch loop
    for epoch in range(EPOCHS):
        model.train()

        # Use one KL weight per epoch instead of updating it every training step
        kl_weight = kl_weight_schedule(epoch, EPOCHS)

        total_recon, total_kl, n_batches = 0.0, 0.0, 0

        for encoder_input, decoder_input, target in make_batches(train_sentences, word_to_index, BATCH_SIZE, pad_idx, bos_idx, eos_idx, unk_idx,):
            encoder_input = encoder_input.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            target = target.to(DEVICE)

            #Apply word dropout only to decoder input, not encoder input or targets
            decoder_input = apply_word_dropout(decoder_input, WORD_DROPOUT_PROB, unk_idx, pad_idx)

            # Forward pass 
            # returns reconstructed output, μ and logσ²
            logits, mu, logvar = model(encoder_input, decoder_input)

            #reconstruction term: log p(x|z), as cross-entropy over the vocab
            # ignore_index=pad_idx so padding positions don't contribute to the loss
            recon_loss = F.cross_entropy(logits.reshape(-1, vocab_size), target.reshape(-1), ignore_index=pad_idx)

            #KL term
            kl_loss = kl_divergence(mu, logvar)

            #negative ELBO we actually minimize
            loss = recon_loss + kl_weight * kl_loss

            # backpropagation
            optimizer.zero_grad()
            loss.backward()

            # Prevent very large gradient updates during training - clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            n_batches += 1

        # Log reconstruction loss, KL loss, and KL weight separately
        print(f"Epoch {epoch + 1}/{EPOCHS} | recon {total_recon / n_batches:.3f} | kl {total_kl / n_batches:.3f} | kl_weight {kl_weight:.2f}")

    # save checkpoint;save the model and vocabulary for sentence generation
    torch.save({"model_state": model.state_dict(), "word_to_index": word_to_index, "index_to_word": index_to_word}, CHECKPOINT_PATH)
    print(f"Saved checkpoint to {CHECKPOINT_PATH}")

    print("Model saved.")
