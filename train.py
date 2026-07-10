#train.py
# Manisha Shekhar Mirje - 402237

import torch
import random

from models.vae import reparameterize
from models.encoder import Encoder
from models.decoder import Decoder
from losses import (kl_annealing_weight, word_dropout, vae_loss)
from model_interface import vae_loss

def make_batches(sentences, word_to_index, batch_size, pad_idx, bos_idx, eos_idx, unk_idx, shuffle=True):
    """
    Turns a list of tokenized sentences into padded (encoder_input, decoder_input, target) tensors
    """
    indices = list(range(len(sentences)))
    if shuffle:
        random.shuffle(indices)
 
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        encoded = []
        for i in batch_idx:
            #convert tokens -> ids
            ids = [word_to_index.get(t, unk_idx) for t in sentences[i]]
            encoder_input = ids + [eos_idx]   # encoder input :w1 ... wn <eos>
            decoder_input = [bos_idx] + ids  #decoder input: <bos> w1 ... wn
            target = ids + [eos_idx]   #target: w1 ... wn <eos>
            encoded.append((encoder_input, decoder_input, target))
 
        # pad all sequences to same length
        max_len = max(len(e[0]) for e in encoded)
 
        def pad(seq):
            return seq + [pad_idx] * (max_len - len(seq))
 
        encoder_batch = torch.tensor([pad(e[0]) for e in encoded], dtype=torch.long)
        decoder_batch = torch.tensor([pad(e[1]) for e in encoded], dtype=torch.long)
        target_batch = torch.tensor([pad(e[2]) for e in encoded], dtype=torch.long)
 
        yield encoder_batch, decoder_batch, target_batch

def train(model_enc, model_dec, train_sentences, word_to_index, index_to_word, 
          pad_idx, unk_idx, bos_idx, eos_idx, batch_size, epochs, lr, 
          word_dropout_prob, checkpoint_path, device):
    
    # optimizer 
    optimizer = torch.optim.Adam(list(model_enc.parameters()) + list(model_dec.parameters()), lr=lr)
    vocab_size = len(index_to_word)
        history = {"recon": [], "kl": [], "kl_weight": []}


    #Training loop - Epoch loop
    for epoch in range(epochs):
        model_enc.train()
        model_dec.train()

        # Use one KL weight per epoch instead of updating it every training step
        kl_weight = kl_annealing_weight(epoch, epochs)

        total_recon, total_kl, n_batches = 0.0, 0.0, 0

        #creating batches of sentences
        for encoder_input, decoder_input, target in make_batches(train_sentences, word_to_index, batch_size, 
                                                                 pad_idx, bos_idx, eos_idx, unk_idx):
            encoder_input = encoder_input.to(device)
            decoder_input = decoder_input.to(device)
            target = target.to(device)

            #Applying word dropout only to decoder input
            decoder_input = word_dropout(decoder_input, word_dropout_prob, unk_idx, pad_idx)

            # Forward pass
            # returns reconstructed output, μ and logσ²
            mu, logvar = model_enc(encoder_input)
            #reparameterization -> z
            z = reparameterize(mu, logvar)
            # decoder -> logits
            logits = model_dec(z, decoder_input)

            loss, recon_loss, kl_loss = vae_loss(logits, target, mu, logvar, pad_idx, kl_weight)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()

            # Prevent very large gradient updates during training - clip gradients
            torch.nn.utils.clip_grad_norm_(list(model_enc.parameters()) + list(model_dec.parameters()), 5.0)
            optimizer.step()

            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            n_batches += 1

        # Log reconstruction loss, KL loss, and KL weight separately
        print(f"Epoch {epoch + 1}/{epochs} | recon {total_recon / n_batches:.3f} | kl {total_kl / n_batches:.3f} | kl_weight {kl_weight:.2f}")
                history["recon"].append(total_recon / n_batches)
                    history["kl"].append(total_kl / n_batches)
                    history["kl_weight"].append(kl_weight)


    # save checkpoint;save the model and vocabulary for sentence generation
    torch.save({"encoder_state": model_enc.state_dict(), "decoder_state": model_dec.state_dict(), "word_to_index": word_to_index, "index_to_word": index_to_word}, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    return history


    #print("Model saved.")
