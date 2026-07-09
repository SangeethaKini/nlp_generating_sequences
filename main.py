import torch

from models.vae import reparameterize
from preprocessing_data import load_dataset_from_hf,preprocess_dataset
from models.encoder import Encoder


EMBED_DIM, HIDDEN_DIM, LATENT_DIM, PAD_IDX = 128, 256, 32, 0

if __name__ == "__main__":

#Javeria Raja 344234 - start
    #loading the dataset
    ds = load_dataset_from_hf()

    #Preprocessing the dataset
    preprocess_dataset(ds)
    train_sentences, validation_sentences, word_to_index_dict, index_to_word_list = preprocess_dataset(ds)

    # print(f"Sample Train Sentence (Ready for encoding): {train_sentences[0]}")
    # print(f"Vocabulary maps successfully built ({len(index_to_word_list)} tokens).")

    #train sentences and validation sentences created ready for encoding and decoding,  word_to_index_dict for encoding and index_to_word_list for decoding created

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