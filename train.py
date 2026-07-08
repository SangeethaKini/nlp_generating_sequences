#train.py

import torch
import torch.nn.functional as F

from dataset import make_batches
from model import kl_divergence

# Hyperparameters
# Sizes are much smaller than the paper's (the paper uses bigger LSTMs / hidden sizes and train on more data)
# with below values training finishes in a reasonable time on BookCorpus subset
EMBED_DIM = 128
HIDDEN_DIM = 256
LATENT_DIM = 32 #size of z - the paper's "continuous space"
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
WORD_DROPOUT_PROB = 0.25

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "vae_checkpoint.pt"
# 

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
    kl_weight = 1.0, epoch / (total_epochs * 0.5)
    return min(kl_weight)


def train(model, train_sentences, word_to_index, index_to_word, pad_idx, unk_idx):
    
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

        for encoder_input, decoder_input, target in make_batches(train_sentences, word_to_index, BATCH_SIZE, pad_idx,
                "<bos>",
                "<eos>",
                "<unk>",
        ):
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
