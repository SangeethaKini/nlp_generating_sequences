#Model_Interface.py
#Sri Vyshnavi Madala 393232

from __future__ import annotations
import torch

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
