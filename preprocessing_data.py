# Javeria Raja - 344324 - file preprocessing_data.py

from datasets import Dataset,load_dataset
import random, spacy
from collections import Counter

nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")
PADDING_TOKEN, BEGINNING_TOKEN, END_TOKEN, UNKNOWN_TOKEN = "<pad>", "<bos>", "<eos>", "<unk>"

def load_dataset_from_hf():
    streamed_ds = load_dataset("mteb/imdb",  split="train", streaming=True).take(2000)
    ds = Dataset.from_list(list(streamed_ds))
    return ds

def sentences_array_from_dataset(dataset):
    #extracts sentences and returns them as array in lower case using spacy's sentencizer
    extracted_text_from_dataset = [ex["text"] for ex in dataset]
    sentences = []
    for text in extracted_text_from_dataset:
        for s in nlp(text).sents:
            sentences.append(s.text.strip().lower())
    # print (f"Total sentences extracted: {len(sentences)}")
    # print (sentences)
    return sentences

def tokenize_and_filter_sentences_by_length(sentences, min_length=5, max_length=15):
    tokenized_sentences = [[t.text for t in doc] for doc in nlp.pipe(sentences, batch_size=1000)]
    filtered_sentences = [s for s in tokenized_sentences if min_length <= len(s) <= max_length]
    return filtered_sentences
    
def splitting_sentences_into_train_and_validation(sentences, validation_split=0.1):
    random.Random(0).shuffle(sentences)
    split_range = int(len(sentences) * validation_split)
    validation_sentences, train_sentences = sentences[:split_range], sentences[split_range:]
    return train_sentences, validation_sentences

def build_vocabulary(sentences, vocab_size=10000):
    counts = Counter(word for sentence in sentences for word in sentence)
    index_to_word_list = [PADDING_TOKEN, BEGINNING_TOKEN, END_TOKEN, UNKNOWN_TOKEN] + [word for word, _ in counts.most_common(vocab_size)]
    word_to_index_dict = {word: i for i, word in enumerate(index_to_word_list)}
    return index_to_word_list, word_to_index_dict

def preprocess_dataset(dataset):

    sentences = sentences_array_from_dataset(dataset)
    filtered_sentences = tokenize_and_filter_sentences_by_length(sentences, min_length=5, max_length=15)
    train_sentences, validation_sentences = splitting_sentences_into_train_and_validation(filtered_sentences, validation_split=0.1)
    index_to_word_list, word_to_index_dict = build_vocabulary(train_sentences + validation_sentences, vocab_size=10000)

    return train_sentences, validation_sentences, word_to_index_dict, index_to_word_list



