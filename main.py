from models.decoder import Decoder, ids_to_sentences
from preprocessing_data import load_dataset_from_hf,preprocess_dataset


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
    
    # Sinthuja Santhiran 398419 - start
    # Call the decoder
    decoder = Decoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, LATENT_DIM, PAD_IDX)

    generated = decoder.generate(z, bos_idx=BOS_IDX, eos_idx=EOS_IDX, max_len=15)
    decoded_sentences = ids_to_sentences(generated, index_to_word_list, EOS_IDX)

    print("Input sentence:  ", " ".join(sample_sentence))
    print("Decoded sentence:", decoded_sentences[0])
    print("Decoder built successfully. Vocab size:", VOCAB_SIZE)
    # Sinthuja Santhiran 398419 - end

#Javeria Raja 344234 - end