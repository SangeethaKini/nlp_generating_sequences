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

#Javeria Raja 344234 - end

#Sangeetha Kamalaksha Kini 392545 

        #Call encoder here 

        #

#Sangeetha Kamalaksha Kini 392525