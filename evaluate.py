from __future__ import annotations
import os
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- INTEGRATION BLOCK -------------------
from models.encoder import Encoder
from models.decoder import Decoder
from models.vae import reparameterize, kl_divergence
from models.losses import vae_loss, kl_annealing_weight
<<<<<<< HEAD
<<<<<<< HEAD
from models.train import train
=======
>>>>>>> 09ea602 (Import kl_annealing_weight from models.losses)
=======
from models.train import train
>>>>>>> 4038d69 (Import train function from models.train)
from model_interface import (
    VocabAdapter, build_batch,
    PAD, BOS, EOS, UNK,
)
<<<<<<< HEAD
<<<<<<< HEAD


<<<<<<< HEAD
=======
>>>>>>> c3a17f9 (Refactor import statement and clean up whitespace)
=======

>>>>>>> 76cc3f2 (Import Encoder, Decoder, and VAE functions)
=======
>>>>>>> 4038d69 (Import train function from models.train)
import generate as G

SEED = 7
random.seed(SEED); torch.manual_seed(SEED)
OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)
# Load data: try the real repo pipeline, else fall back to a toy corpus.
def load_real_data(limit=2000):
    """Uses Javeria's preprocessing_data.py if it (and its deps) are available."""
    from preprocessing_data import load_dataset_from_hf, preprocess_dataset
    ds = load_dataset_from_hf()
    train, val, word_to_index, index_to_word = preprocess_dataset(ds)
    vocab = VocabAdapter(index_to_word, word_to_index)
    return train, val, vocab

def load_toy_data(n=600):
    subj = ["i", "he", "she", "they", "we"]
    verb = ["went", "walked", "ran", "came", "looked"]
    place = ["to the store", "to the kitchen", "into the room",
             "down the street", "to the door", "across the floor"]
    tail = ["quietly", "again", "at last", "in silence", "for a moment"]
    sents = [[random.choice(subj), random.choice(verb),
              *random.choice(place).split(), random.choice(tail)]
             for _ in range(n)]
    vocab = VocabAdapter.from_sentences(sents)
    return sents, sents[: n // 10], vocab

def get_data():
    try:
        train, val, vocab = load_real_data()
        print(f"[data] real Books Corpus subset | train={len(train)} val={len(val)}")
        return train, vocab
    except Exception as e:
        print(f"[data] real pipeline unavailable ({type(e).__name__}); using toy corpus")
        train, _, vocab = load_toy_data()
        return train, vocab

# Batching over a list of token-lists

def batches(sentences, vocab, bs=64, device="cpu"):
    data = list(sentences)
    random.shuffle(data)
    for i in range(0, len(data), bs):
        yield build_batch(data[i:i + bs], vocab, device=device)

# KL annealing (Person 6) - linear warmup, Section 3.1

def kl_weight_at(step, warmup=300):
    return min(1.0, step / warmup)

# Compact training loop (stands in for Person 5's train.py)

def train(model, sents, vocab, epochs=8, lr=2e-3, word_keep=0.5,
          anneal=True, device="cpu"):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"recon": [], "kl": [], "kl_weight": []}
    step = 0
    for _ in range(epochs):
        model.train()
        for batch in batches(sents, vocab, device=device):
            kw = kl_weight_at(step) if anneal else 1.0
            logits, mu, logvar, _ = model(batch, word_keep=word_keep)
            loss, parts = vae_loss(logits, batch, mu, logvar,
                                   kl_weight=kw, pad_id=vocab.pad_id)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            history["recon"].append(parts["recon"])
            history["kl"].append(parts["kl"])
            history["kl_weight"].append(kw)
            step += 1
    return history


def plot_losses(history, path):
    steps = range(len(history["recon"]))
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(steps, history["recon"], label="reconstruction (cross-entropy)",
             color="tab:blue")
    ax1.plot(steps, history["kl"], label="KL divergence", color="tab:red")
    ax1.set_xlabel("training step"); ax1.set_ylabel("loss (nats)")
    ax2 = ax1.twinx()
    ax2.plot(steps, history["kl_weight"], "--", color="tab:green",
             label="KL weight (anneal)")
    ax2.set_ylabel("KL weight"); ax2.set_ylim(-0.05, 1.05)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=8)
    ax1.set_title("Vyshnavi: Sentence-VAE training (KL annealing + word dropout)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def banner(t):
    print("\n" + "=" * 68 + f"\n {t}\n" + "=" * 68)


def probe_batch(vocab, device):
    """Two real-looking short sentences to probe reconstruction/homotopy/impute."""
    s1 = ["she", "went", "to", "the", "door", "quietly"]
    s2 = ["they", "ran", "down", "the", "street", "again"]
    return build_batch([s1, s2], vocab, device=device), (s1, s2)


def main():
    device = "cpu"
    sents, vocab = get_data()
    print(f"[vocab] size={len(vocab)}  pad={vocab.pad_id} bos={vocab.bos_id} "
          f"eos={vocab.eos_id} unk={vocab.unk_id}")

    banner("Training standard Sentence-VAE (word_keep=0.5, KL annealing)")
    model = SentenceVAE(vocab, emb_dim=64, hid_dim=128, z_dim=16).to(device)
    hist = train(model, sents, vocab, epochs=8, word_keep=0.5, device=device)
    plot_path = os.path.join(OUT, "loss_curves.png")
    plot_losses(hist, plot_path)
    print(f"final recon={hist['recon'][-1]:.3f}  final KL={hist['kl'][-1]:.3f}")
    print(f"saved loss plot -> {plot_path}")

    probe, _ = probe_batch(vocab, device)

    banner("Experiment 1 - Reconstruction (Table 7)")
    rec = G.reconstruct(model.encoder, model.decoder, reparameterize, probe,
                        vocab, n_samples=2)
    for k, v in rec.items():
        print(f"  [{k:7}] {v}")

    banner("Experiment 2 - Samples from the Gaussian prior (Table 5)")
    for s in G.sample_prior(model.decoder, vocab, model.z_dim, n=5, device=device):
        print("  •", s)

    banner("Experiment 3 - Missing-word imputation, final ~20% (Table 3)")
    for r in G.impute(model.encoder, model.decoder, reparameterize, probe,
                      vocab, frac=0.34):
        print(f"  observed: {r['observed']}")
        print(f"     true : {r['true']}")
        print(f"    imputed: {r['imputed']}\n")

    banner("Experiment 4 - Latent interpolation / homotopy (Tables 6 & 8)")
    for t, line in G.homotopy(model.encoder, model.decoder, probe, vocab, steps=7):
        print(f"  t={t:<4} {line}")

    banner("Experiment 5 - Inputless decoder, word_keep=0 (Section 4 / Table 5)")
    model_il = SentenceVAE(vocab, emb_dim=64, hid_dim=128, z_dim=16).to(device)
    hist_il = train(model_il, sents, vocab, epochs=8, word_keep=0.0, device=device)
    print(f"  inputless final KL={hist_il['kl'][-1]:.3f} "
          f"(expect HIGHER than standard {hist['kl'][-1]:.3f}: decoder forced to use z)")
    for s in G.inputless_sample(model_il.decoder, vocab, model_il.z_dim, n=5,
                                device=device):
        print("  •", s)

    banner("Done")

if __name__ == "__main__":
    main()
<<<<<<< HEAD
<<<<<<< HEAD
# Sri Vyshnavi Madala 393232
=======

#Sri vyshnavi Madala 393232
>>>>>>> 3758a89 (Add author comment to evaluate.py)
=======
# Sri Vyshnavi Madala 393232
>>>>>>> 6776516 (Add files via upload)
