import torch
from models.decoder import Decoder, ids_to_sentences
from models.vae import reparameterize
from preprocessing_data import (load_dataset_from_hf, preprocess_dataset, PADDING_TOKEN, UNKNOWN_TOKEN, BEGINNING_TOKEN, END_TOKEN)
from models.encoder import Encoder
from train import train

#EMBED_DIM, HIDDEN_DIM, LATENT_DIM, PAD_IDX = 128, 256, 32, 0
# Hyperparameters
EMBED_DIM = 128
HIDDEN_DIM = 256
LATENT_DIM = 32

# Training parameters
VOCAB_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
WORD_DROPOUT_PROB = 0.25
CHECKPOINT_PATH = "vae_checkpoint.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":

#Javeria Raja 344234 - start
    #loading the dataset
    ds = load_dataset_from_hf()

    #Preprocessing the dataset
    preprocess_dataset(ds)
    train_sentences, validation_sentences, word_to_index_dict, index_to_word_list = preprocess_dataset(ds)

#       #special token indices
#     pad_idx = word_to_index_dict[PADDING_TOKEN]
#     bos_idx = word_to_index_dict[BEGINNING_TOKEN]
#     eos_idx = word_to_index_dict[END_TOKEN]
#     unk_idx = word_to_index_dict[UNKNOWN_TOKEN]
    # print(f"Sample Train Sentence (Ready for encoding): {train_sentences[0]}")
    # print(f"Vocabulary maps successfully built ({len(index_to_word_list)} tokens).")

    #train sentences and validation sentences created ready for encoding and decoding,  word_to_index_dict for encoding and index_to_word_list for decoding created
    
    # Sinthuja Santhiran 398419 - start
    # Call the decoder
    decoder = Decoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, LATENT_DIM, PAD_IDX)

    generated = decoder.generate(z, bos_idx=BOS_IDX, eos_idx=EOS_IDX, max_len=15)
    decoded_sentences = ids_to_sentences(generated, index_to_word_list, eos_idx)

    print("Input sentence:  ", " ".join(sample_sentence))
    print("Decoded sentence:", decoded_sentences[0])
    print("Decoder built successfully. Vocab size:", VOCAB_SIZE)
    # Sinthuja Santhiran 398419 - end

#Javeria Raja 344234 - end

#Sangeetha Kamalaksha Kini 392545
    VOCAB_SIZE = len(word_to_index_dict)
    #Call encoder here
    encoder = Encoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, LATENT_DIM, PAD_IDX)

    #Get token for every sentence in tain_sentences and pass it as a tensor to encoder
    token_ids = [word_to_index_dict.get(token, word_to_index_dict["<unk>"]) for sentence in train_sentences for token in sentence]
    input_tensor = torch.tensor([token_ids], dtype=torch.long)

    m, logvar = encoder(input_tensor)
    z = reparameterize(m, logvar)
    #print("Encoded latent vector z shape:", z.shape)  # Should be (1, LATENT_DIM)
#Sangeetha Kamalaksha Kini 392525

    
#     print("Starting Training...")
#     train(model_enc=encoder, model_dec=decoder, train_sentences=train_sentences, word_to_index=word_to_index_dict, index_to_word=index_to_word_list, 
#           pad_idx=pad_idx, unk_idx=unk_idx, eos_idx=eos_idx, bos_idx=bos_idx, 
#           batch_size= VOCAB_SIZE, epochs=EPOCHS, lr=LEARNING_RATE, word_dropout_prob=WORD_DROPOUT_PROB, checkpoint_path=CHECKPOINT_PATH, device=DEVICE)
#     print("Training completed. Model saved to", CHECKPOINT_PATH)
#     checkpoint = torch.load("vae_checkpoint.pt", map_location="cpu")