import nltk
from nltk.stem.porter import *
import numpy as np
import tensorflow as tf

class ImagePreprocessor:
    def __init__(self):
        pass

    def preprocess_pipeline(self, img_files, img_height=224, img_width=224):
        # imgs = []
        # for img_file in img_files:
        #   # 1. Read image file
        #   img = tf.io.read_file(img_file)
        #   # 2. Decode the image
        #   img = tf.image.decode_jpeg(img, channels=3)
        #   # 3. Convert to float32 in [0, 1] range
        #   img = tf.image.convert_image_dtype(img, tf.float32)
        #   # 4. Resize to the desired size
        #   img = tf.image.resize(img, [img_height, img_width])
        #   imgs.append(img)
        # return tf.convert_to_tensor(imgs)
        return img_files


class TextPreprocessor:

    def __init__(self):
        self.nltk_stop_words_table = ['is', 'it', 'the', 'are', 'will', 'of',
                                      'are', 'you', 'do', 'think', 'does', 'there', 'should', 'does', 'to']
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
        #stop_words_table = self.nltk_stop_words_table
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


class VQAPreprocessor:
    def __init__(self, text_processor, image_processor, word_index, word_emb_dim=21):
        self.text_p = text_processor
        self.image_p = image_processor
        self.word_emb_dim = word_emb_dim
        self.word_index = word_index
        self.label_dic = {'yes': 1, 'no': 0}

    def process_samples(self, img_files, q, anno=None, test=False):
        imgs = self.image_p.preprocess_pipeline(img_files)
        q = self.text_p.preprocess_pipeline(q)
        encode_q = self.encode_and_add_padding(q)

        if not test:
            return imgs, encode_q, anno
        else:
            return imgs, encode_q

    def encode_and_add_padding(self, sentences):
        seq_length = self.word_emb_dim
        word_index = self.word_index

        sent_encoded = []
        for sent in sentences:
            temp_encoded = [
                word_index[word] if word in word_index else word_index['[UNKNOWN]'] for word in sent]
            if len(temp_encoded) < seq_length:
                temp_encoded += [word_index['[PAD]']] * \
                    (seq_length - len(temp_encoded))
            sent_encoded.append(temp_encoded)

        return tf.convert_to_tensor(np.array(sent_encoded))
