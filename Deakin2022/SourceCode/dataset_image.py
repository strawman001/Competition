import tensorflow as tf
from tensorflow import keras
import json
import base64
import os
from sklearn.utils import shuffle
import pickle
import nltk
import numpy as np

answers = ['yes', 'no']
num_answers = len(answers)

img_width = 224
img_height = 224
image_size = (img_height, img_width)

#Different filter method
def filter_questions(questions, annotations, answers, imgs_path):
    # Make sure the questions and annotations are alligned
    questions['questions'] = sorted(questions['questions'], key=lambda x: x['question_id'])
    annotations['annotations'] = sorted(annotations['annotations'], key=lambda x: x['question_id'])
    q_out = []
    anno_out = []
    imgs_out = []
    q_ids = []
    question_ids_set = set()
    # Filter annotations
    for annotation in annotations['annotations']:
        if annotation['multiple_choice_answer'] in answers:
            question_ids_set.add(annotation['question_id'])
            q_ids.append(annotation['question_id'])
            anno_out.append(answers.index(annotation['multiple_choice_answer']))
    # Filter images and questions
    for q in questions['questions']:
        if q['question_id'] in question_ids_set:
            # Preprocessing the question
            q_text = q['question'].lower()
            q_text = q_text.replace('?', ' ? ')
            q_text = q_text.replace('.', ' . ')
            q_text = q_text.replace(',', ' . ')
            q_text = q_text.replace('!', ' . ').strip()
            q_out.append(q_text)
            file_name = str(q['image_id'])
            while len(file_name) != 12:
                file_name = '0' + file_name
            file_name = imgs_path + questions['data_type'] + '_' + questions['data_subtype'] + '_' + file_name + '.png'
            imgs_out.append(file_name)
    return imgs_out, q_out, anno_out, q_ids

def filter_questions_coco(questions, annotations, answers, imgs_path, image_name_template = 'COCO_train2014_%012d.jpg',all=True, size=50000):
    # Make sure the questions and annotations are alligned
    questions['questions'] = sorted(questions['questions'], key=lambda x: x['question_id'])
    annotations['annotations'] = sorted(annotations['annotations'], key=lambda x: x['question_id'])
    q_out = []
    anno_out = []
    imgs_out = []
    q_ids = []
    question_ids_set = set()
    # Filter annotations
    for annotation in annotations['annotations']:
        if annotation['multiple_choice_answer'] in answers:
            question_ids_set.add(annotation['question_id'])
            q_ids.append(annotation['question_id'])
            anno_out.append(answers.index(annotation['multiple_choice_answer']))
    # Filter images and questions
    for q in questions['questions']:
        if q['question_id'] in question_ids_set:
            # Preprocessing the question
            q_text = q['question'].lower()
            q_text = q_text.replace('?', ' ? ')
            q_text = q_text.replace('.', ' . ')
            q_text = q_text.replace(',', ' . ')
            q_text = q_text.replace('!', ' . ').strip()
            q_out.append(q_text)
            # file_name = str(q['image_id'])
            # while len(file_name) != 12:
            #     file_name = '0' + file_name
            file_name = imgs_path + image_name_template % q['image_id']
            # file_name = imgs_path + questions['data_type'] + '_' + questions['data_subtype'] + '_' + file_name + '.png'
            imgs_out.append(file_name)
    imgs_out, q_out, anno_out, q_ids = shuffle(imgs_out, q_out, anno_out, q_ids, random_state=0)
    if all:
      return imgs_out, q_out, anno_out, q_ids
    else:
      return imgs_out[:size], q_out[:size], anno_out[:size], q_ids[:size]

#Different encode method
def encode_single_sample(img_file, q, anno):
    img = tf.io.read_file(img_file)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [img_height, img_width])

    return (img, q), anno

def encode_single_sample_features(img_offset, q, anno):
    f = open("/root/.keras/datasets/train2014_obj36.tsv",mode="r")
    f.seek(img_offset)
    img_info = f.readline()
    f.close()
    #assert img_info.startswith('COCO') and img_info.endswith('\n'), 'Offset is inappropriate'
    img_info = img_info.split('\t')
    feats = img_info[-1]
    feats = np.frombuffer(base64.b64decode(feats), dtype=np.float32)
    feats = feats.reshape(36,-1)
    feats.setflags(write=False)

    return feats, q, anno

def encode_single_sample_val_features(img_out, q, anno,val_feature_dic = val_feature_dic):
    #print(img_out.numpy().decode("utf-8"))
    feats = val_feature_dic[img_out.numpy().decode("utf-8")]
    return feats, q, anno

def make_sample(feates,q,anno):
    return (feates,q),anno

#Preprocess
class ImagePreprocessor:
    def __init__(self):
        pass

    def preprocess_pipeline(self, img_files, img_height=224, img_width=224):
        return img_files


from nltk.stem.porter import *
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
            return imgs, encode_q,

    def encode_and_add_padding(self, sentences):
        seq_length = self.word_emb_dim
        word_index = self.word_index

        sent_encoded = []
        for sent in sentences:
            temp_encoded = [word_index[word] if word in word_index else word_index['[UNKNOWN]'] for word in sent]
            if len(temp_encoded) < seq_length:
                temp_encoded += [word_index['[PAD]']] * (seq_length - len(temp_encoded))
            sent_encoded.append(temp_encoded)

        return tf.convert_to_tensor(np.array(sent_encoded))



#Make datasets
def make_abstract_train_img_dataset():
    # Traning Images
    data_url = "http://206.12.93.90:8080/simpson2022_dataset/scene_img_abstract_v002_train2015.tar.gz"
    zip_path = keras.utils.get_file("scene_img_abstract_v002_train2015.tar.gz", data_url, extract=True)
    imgs_path_train = os.path.dirname(zip_path) + '/scene_img_abstract_v002_train2015/'

    # Traning Annotations
    data_url = "http://206.12.93.90:8080/simpson2022_dataset/abstract_v002_train2015_annotations.zip"
    zip_path = keras.utils.get_file("abstract_v002_train2015_annotations.zip", data_url,
                                    cache_subdir='datasets/abstract_v002_train2015_annotations/', extract=True)
    anno_train_file = os.path.dirname(zip_path) + '/abstract_v002_train2015_annotations.json'

    # Traning Questions
    data_url = "http://206.12.93.90:8080/simpson2022_dataset/OpenEnded_abstract_v002_train2015_questions.zip"
    zip_path = keras.utils.get_file("OpenEnded_abstract_v002_train2015_questions.zip", data_url,
                                    cache_subdir='datasets/OpenEnded_abstract_v002_train2015_questions/', extract=True)
    q_train_file = os.path.dirname(zip_path) + '/OpenEnded_abstract_v002_train2015_questions.json'

    q_train = json.load(open(q_train_file))
    anno_train = json.load(open(anno_train_file))

    imgs_train, q_train, anno_train, q_ids_train = filter_questions(q_train, anno_train, answers, imgs_path_train)
    imgs_train, q_train, anno_train, q_ids_train = shuffle(imgs_train, q_train, anno_train, q_ids_train, random_state=0)

    with open("/content/word.pkl", 'rb') as f:
        word_pkl = pickle.load(f)

    word_index = word_pkl["word_index"]
    word_num = word_pkl["word_num"]
    emb_table = word_pkl["emb_table"]

    preprocessor = VQAPreprocessor(TextPreprocessor(), ImagePreprocessor(), word_index)

    # We define the batch size
    batch_size = 32
    # Define the trainig dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (preprocessor.process_samples(imgs_train, q_train, anno_train))
    )
    train_dataset = (
        train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    return  train_dataset

def make_simpson_val_img_dataset():

    #  Validation Images
    data_url = "http://206.12.93.90:8080/simpson2022_dataset/simpsons_validation.tar.gz"
    zip_path = keras.utils.get_file("simpsons_validation.tar.gz", data_url, extract=True)
    imgs_path_val = os.path.dirname(zip_path) + '/simpsons_validation/'

    #  Validation Questions
    data_url = "http://206.12.93.90:8080/simpson2022_dataset/questions_validation.zip"
    zip_path = keras.utils.get_file("questions_validation.zip", data_url,
                                    cache_subdir='datasets/questions_validation/', extract=True)
    q_val_file = os.path.dirname(zip_path) + '/questions_validation.json'

    data_url = "http://206.12.93.90:8080/simpson2022_dataset/annotations_validation.zip"
    zip_path = keras.utils.get_file("annotations_validation.zip", data_url,
                                    cache_subdir='datasets/annotations_validation/', extract=True)
    anno_val_file = os.path.dirname(zip_path) + '/annotations_validation.json'

    q_val = json.load(open(q_val_file))
    anno_val = json.load(open(anno_val_file))

    imgs_val, q_val, anno_val, q_ids_val = filter_questions(q_val, anno_val, answers, imgs_path_val)

    with open("/content/word.pkl", 'rb') as f:
        word_pkl = pickle.load(f)

    word_index = word_pkl["word_index"]
    word_num = word_pkl["word_num"]
    emb_table = word_pkl["emb_table"]

    preprocessor = VQAPreprocessor(TextPreprocessor(), ImagePreprocessor(), word_index)


    # Define the validation dataset
    batch_size = 1
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (preprocessor.process_samples(imgs_val, q_val, anno_val))
    )
    val_dataset = (
        val_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    return val_dataset

def make_coco_train_img_dataset():
    # Traning Images
    data_url = "http://images.cocodataset.org/zips/train2014.zip"
    zip_path = keras.utils.get_file("train2014.zip", data_url, extract=True)
    imgs_path_train = os.path.dirname(zip_path) + '/train2014/'

    # Traning Questions
    data_url = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip"
    zip_path = keras.utils.get_file("v2_Questions_Train_mscoco.zip", data_url,
                                    cache_subdir='datasets/train_Questions/', extract=True)
    q_train_file = os.path.dirname(zip_path) + '/v2_OpenEnded_mscoco_train2014_questions.json'

    # Traning Annotations
    data_url = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip"
    zip_path = keras.utils.get_file("v2_Annotations_Train_mscoco.zip", data_url,
                                    cache_subdir='datasets/train_Annotations/', extract=True)
    anno_train_file = os.path.dirname(zip_path) + '/v2_mscoco_train2014_annotations.json'

    q_train = json.load(open(q_train_file))
    anno_train = json.load(open(anno_train_file))

    imgs_train, q_train, anno_train, q_ids_train = filter_questions_coco(q_train, anno_train, answers, imgs_path_train)
    imgs_train, q_train, anno_train, q_ids_train = shuffle(imgs_train, q_train, anno_train, q_ids_train, random_state=0)

    with open("/content/word.pkl", 'rb') as f:
        word_pkl = pickle.load(f)

    word_index = word_pkl["word_index"]
    word_num = word_pkl["word_num"]
    emb_table = word_pkl["emb_table"]

    preprocessor = VQAPreprocessor(TextPreprocessor(), ImagePreprocessor(), word_index)

    # We define the batch size
    batch_size = 32
    # Define the trainig dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (preprocessor.process_samples(imgs_train, q_train, anno_train))
    )
    train_dataset = (
        train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    return train_dataset
