from nltk.stem.porter import *
import pickle
import re
import nltk
import numpy as np
import gensim
from gensim.models import FastText

class TextPreprocessor:

    def __init__(self):
        self.nltk_stop_words_table = ['is', 'it', 'the', 'are', 'will', 'of', 'are', 'you', 'do', 'think', 'does',
                                      'there', 'should', 'does', 'to']
        self.default_stemmer = PorterStemmer()
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        self.default_lemmatizer = nltk.stem.WordNetLemmatizer()

    def remove_punctuation(self, text):
        text = re.sub(r"[^\w\s]", "", text)
        return text

    def tokenize(self, text):
        text = text.strip()
        tokens = text.split(' ')
        return tokens

    def remove_stopwords(self, tokens, stop_words_table):
        # stop_words_table = self.nltk_stop_words_table
        tokens = [w for w in tokens if not w in stop_words_table]
        return tokens

    def stemming(self, tokens, stemmer):
        tokens = [stemmer.stem(w) for w in tokens]
        return tokens

    def lemmatize(self, tokens, lemmatizer):
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
        return tokens

    def preprocess_pipeline(self, text_list):
        preprocessed_text_list = []
        for text in text_list:
            text = self.remove_punctuation(text)
            tokens = self.tokenize(text)
            tokens = self.remove_stopwords(tokens, self.nltk_stop_words_table)
            tokens = self.stemming(tokens, self.default_stemmer)
            tokens = self.lemmatize(tokens, self.default_lemmatizer)
            preprocessed_text_list.append(tokens)

        return preprocessed_text_list


def create_word_tool(sentences):
    # Create word set
    word_set = set()
    for sentence in sentences:
        for word in sentence:
            word_set.add(word)

    word_set.remove('')
    word_list = list(word_set)
    word_list.sort()

    # Add special feature
    word_list.insert(0, '[PAD]')
    word_list.insert(1, '[UNKNOWN]')

    # Make word dictionary
    word_index = {}
    for i, word in enumerate(word_list):
        word_index[word] = i

    return word_list, word_index

def create_embedding_table(word_list, model):
    emb_dim = model.vector_size

    emb_table = []
    for i, word in enumerate(word_list):
        embedding_item = np.array([])
        if word not in model:
            embedding_item = [0] * emb_dim
        else:
            embedding_item = model[word]

        emb_table.append(embedding_item)

    emb_table = np.array(emb_table)

    return emb_table, emb_dim


def make_embedding(q_train,q_val):
    textpreprocessor = TextPreprocessor()
    q_train_pre = textpreprocessor.preprocess_pipeline(q_train)
    q_val_pre = textpreprocessor.preprocess_pipeline(q_val)

    embedding_sentences_list = q_train_pre + q_val_pre
    embedding_sentences_list

    size = [len(sent) for sent in embedding_sentences_list]
    max(size)



    ft_cbow_model = FastText(sentences=embedding_sentences_list, size=100, window=5, min_count=5, workers=2, sg=0)

    word_list, word_index = create_word_tool(embedding_sentences_list)

    emb_table, emb_dim = create_embedding_table(word_list, model=ft_cbow_model)

    import pickle
    with open("./word.pkl", 'wb') as f:
        pickle.dump({
            "word_index": word_index,
            "word_num": len(word_list),
            "emb_table": emb_table
        }, f)