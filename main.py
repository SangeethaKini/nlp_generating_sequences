import torch
from preprocessing_data import (load_dataset_from_hf, preprocess_dataset, PADDING_TOKEN, UNKNOWN_TOKEN, BEGINNING_TOKEN, END_TOKEN)
from train import train
from models.decoder import Decoder, ids_to_sentences
from models.vae import reparameterize, kl_divergence
from models.encoder import Encoder

#Hyperparameters
EMBED_DIM = 128
HIDDEN_DIM = 256
LATENT_DIM = 32

#Training parameters
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
WORD_DROPOUT_PROB = 0.25
CHECKPOINT_PATH = "vae_checkpoint.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBED_DIM, HIDDEN_DIM, LATENT_DIM, PAD_IDX = 128, 256, 32, 0

if __name__ == "__main__":

    print("Loading dataset...")
    #loading the dataset
    ds = load_dataset_from_hf()

    #tokenize + building vocabulary + split
    preprocess_dataset(ds)
    train_sentences, validation_sentences, word_to_index_dict, index_to_word_list = preprocess_dataset(ds)
    print(f"Number of training sentences: {len(train_sentences)}")
    vocab_size = len(index_to_word_list)

    #special token indices
    pad_idx = word_to_index_dict[PADDING_TOKEN]
    bos_idx = word_to_index_dict[BEGINNING_TOKEN]
    eos_idx = word_to_index_dict[END_TOKEN]
    unk_idx = word_to_index_dict[UNKNOWN_TOKEN]

    # encoder and decoder
    print("Initializing encoder:")
    encoder = Encoder(vocab_size, emb_size=EMBED_DIM, hidden_size=HIDDEN_DIM, latent_size=LATENT_DIM, pad_idx=pad_idx).to(DEVICE)
    print("Initializing Decoder...")
    decoder = Decoder(vocab_size, emb_size=EMBED_DIM, hidden_size=HIDDEN_DIM, latent_size=LATENT_DIM, pad_idx=pad_idx).to(DEVICE)

    print("Starting Training...")
    train(model_enc=encoder, model_dec=decoder, train_sentences=train_sentences, word_to_index=word_to_index_dict, index_to_word=index_to_word_list, 
          pad_idx=pad_idx, unk_idx=unk_idx, eos_idx=eos_idx, bos_idx=bos_idx, 
          batch_size= BATCH_SIZE, epochs=EPOCHS, lr=LEARNING_RATE, word_dropout_prob=WORD_DROPOUT_PROB, checkpoint_path=CHECKPOINT_PATH, device=DEVICE)
    
    print("Training completed. Model saved to", CHECKPOINT_PATH)
