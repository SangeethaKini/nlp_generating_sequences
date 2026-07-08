from preprocessing_data import (load_dataset_from_hf,preprocess_dataset, PADDING_TOKEN, UNKNOWN_TOKEN)

import torch
from train import train
from model import (Encoder, Decoder, SentenceVAE)
# replace above line as per the encoder, decoder and vae files and functions

EMBED_DIM = 128
HIDDEN_DIM = 256
LATENT_DIM = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":

    print("Loading dataset...")
    #loading the dataset
    ds = load_dataset_from_hf()

    #Preprocessing the dataset
    preprocess_dataset(ds)
    train_sentences, validation_sentences, word_to_index_dict, index_to_word_list = preprocess_dataset(ds)

    vocab_size = len(index_to_word_list)
    pad_idx = word_to_index_dict[PADDING_TOKEN]
    bos_idx = word_to_index_dict[BEGINNING_TOKEN]
    eos_idx = word_to_index_dict[END_TOKEN]
    unk_idx = word_to_index_dict[UNKNOWN_TOKEN]

    encoder = Encoder(vocab_size, EMBED_DIM, HIDDEN_DIM, LATENT_DIM, pad_idx)
    decoder = Decoder(vocab_size, EMBED_DIM, HIDDEN_DIM, LATENT_DIM, pad_idx)
    model = SentenceVAE(encoder, decoder).to(DEVICE)

    train(model=model, train_sentences=train_sentences, word_to_index=word_to_index_dict, index_to_word=index_to_word_list, 
          pad_idx=pad_idx, unk_idx=unk_idx, eos_idx=eos_idx, unk_idx=unk_idx)
