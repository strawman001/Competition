import tensorflow as tf
from tensorflow import keras
import numpy as np


class BottomUpVisualHead(tf.keras.layers.Layer):
    def __init__(self, num_feature=20, visual_feature_size=[32, 32]):
        super(BottomUpVisualHead, self).__init__()
        self.num_feature = num_feature
        self.visual_feature_size = visual_feature_size

        self.resnet_conv = tf.keras.applications.resnet50.ResNet50(
            include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling="avg")

    def build(self, input_shape):
        pass

    def call(self, x):

        img = x
        part_array = self.sliceimg(img)
        reshape_array = part_array[0]
        for i in range(x.shape[0]-1):
            reshape_array = tf.concat((reshape_array, part_array[i+1]), axis=0)
        features = self.resnet_conv(reshape_array)
        reshape_features = features[:9][None, :, :]
        for i in range(x.shape[0]-1):
            reshape_features = tf.concat(
                (reshape_features, features[(i+1)*9:(i+2)*9][None, :, :]), axis=0)
        return reshape_features

    def sliceimg(self, img, slice_size=[3, 3], img_size=(224, 224)):
        img = tf.image.resize(img, img_size)
        xth = slice_size[0]
        yth = slice_size[1]

        width = int((img.shape[1]-1)/slice_size[0])
        height = int((img.shape[2]-1)/slice_size[1])

        init = False
        part_array = []
        for i in range(xth):
            for j in range(yth):
                if i == xth-1 and j == yth-1:
                    part = img[:, int(i*width):, int(j*height):, :]
                elif i == xth-1:
                    part = img[:, int(i*width):, int(j*height):int((j+1)*height), :]
                elif j == yth-1:
                    part = img[:, int(i*width):((i+1)*width),int(j*height):, :]
                else:
                    part = img[:, int(i*width):int((i+1)*width),int(j*height):int((j+1)*height), :]

                part = tf.image.resize(part, self.visual_feature_size)[:, None, :, :, :]
                if init:
                    part_array = tf.concat((part_array, part), axis=1)
                else:
                    part_array = part
                    init = True

        return part_array


class QuestionEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, num_hidden, vocab_size, emb_dim, embedding_init=False, embedding_weights=None, dropout=0.3):
        super(QuestionEmbeddingLayer, self).__init__()
        if not embedding_init:
            self.embedding = tf.keras.layers.Embedding(
                vocab_size, emb_dim, mask_zero=True)
        else:
            self.embedding = tf.keras.layers.Embedding(vocab_size, emb_dim,
                                                       embeddings_initializer=tf.keras.initializers.Constant(embedding_weights), mask_zero=True)
        self.rnn = tf.keras.layers.GRU(
            num_hidden, activation='tanh', dropout=dropout)

    def build(self, input_shape):
        pass

    def call(self, x):
        x = self.embedding(x)
        x = self.rnn(x)
        return x


class LVAttention(tf.keras.layers.Layer):
    def __init__(self, v_dim, q_dim, num_hidden, dropout=0.3):
        super(LVAttention, self).__init__()
        self.v_dense = tf.keras.layers.Dense(num_hidden,
                                            input_shape=[None, None, v_dim], activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())
        self.q_dense = tf.keras.layers.Dense(num_hidden,
                                            input_shape=[None, q_dim], activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())
        self.dropout = tf.keras.layers.Dropout(rate=0.3)
        self.linear = tf.keras.layers.Dense(1, input_shape=[None, None, num_hidden], kernel_initializer=tf.keras.initializers.HeNormal())

    def build(self, input_shape):
        pass

    def call(self, x):
        v, q = x

        logits = self.logit(v, q)
        w = tf.nn.softmax(logits, axis=1)

        return w

    def logit(self, v, q):
        batch_size, k, v_dim = v.shape

        v_proj = self.v_dense(v)
        q_proj = tf.repeat(tf.expand_dims(
            self.q_dense(q), axis=1), repeats=[k], axis=1)

        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)

        return logits


class SimpleClassifier(tf.keras.layers.Layer):
    def __init__(self, num_hidden, num_class):
        super(SimpleClassifier, self).__init__()
        self.num_hidden = num_hidden
        self.num_class = num_class

    def build(self, input_shape):
        self.dense1 = tf.keras.layers.Dense(
            self.num_hidden, input_shape=input_shape, activation='relu')
        self.dense2 = tf.keras.layers.Dense(
            self.num_class, input_shape=(input_shape[0], self.num_hidden))

    def call(self, x):
        x = self.dense1(x)
        output = self.dense2(x)
        prob = tf.nn.softmax(output)
        return prob


class BuTdModel(tf.keras.Model):
    def __init__(self, voc_size, embedding_weights, q_dim=512, v_dim=2048, attention_hidden=1024, joint_dim=512):
        super(BuTdModel, self).__init__()
        self.visual_head = BottomUpVisualHead()
        self.question_head = QuestionEmbeddingLayer(
            512, voc_size, 100, embedding_init=True, embedding_weights=embedding_weights)
        self.attention = LVAttention(v_dim, q_dim, attention_hidden)
        self.v_proj = tf.keras.layers.Dense(joint_dim, input_shape=[None, v_dim], activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())
        self.q_proj = tf.keras.layers.Dense(joint_dim, input_shape=[None, q_dim], activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())
        self.classifier = SimpleClassifier(256, 2)

    def build(self, input_shape):
        pass

    def call(self, x):
        v, q = x
        v = self.visual_head(v)

        q_emb = self.question_head(q)  # [batch, q_dim]
        att = self.attention((v, q_emb))
        v_emb = tf.reduce_sum((att * v), axis=1)  # [batch, v_dim]

        q_repr = self.q_proj(q_emb)
        v_repr = self.v_proj(v_emb)
        joint_repr = q_repr * v_repr

        logits = self.classifier(joint_repr)
        return logits


class BuTdModel_Feature(tf.keras.Model):
    def __init__(self, voc_size, embedding_weights, q_dim=512, v_dim=2048, attention_hidden=1024, joint_dim=512):
        super(BuTdModel_Feature, self).__init__()
        self.question_head = QuestionEmbeddingLayer(512,voc_size,100,embedding_init=True,embedding_weights=embedding_weights)
        self.attention = LVAttention(v_dim, q_dim, attention_hidden)
        self.v_proj = tf.keras.layers.Dense(joint_dim,input_shape=[None,v_dim],activation='relu',kernel_initializer=tf.keras.initializers.HeNormal())
        self.q_proj = tf.keras.layers.Dense(joint_dim,input_shape=[None,q_dim],activation='relu',kernel_initializer=tf.keras.initializers.HeNormal())
        self.classifier = SimpleClassifier(256,2)

    def build(self, input_shape):
        pass


    def call(self, x):
        v,q = x

        q_emb = self.question_head(q) # [batch, q_dim]
        att = self.attention((v, q_emb))

        v_emb = tf.reduce_sum((att * v),axis=1) # [batch, v_dim]

        q_repr = self.q_proj(q_emb)
        v_repr = self.v_proj(v_emb)
        joint_repr = q_repr * v_repr
        #print(joint_repr)
        logits = self.classifier(joint_repr)
        return logits

class BuTdModel_Warp(tf.keras.Model):
    def __init__(self, model_feature):
        super(BuTdModel_Warp, self).__init__()
        self.visual_head = BottomUpVisualHead()
        self.model_feature = model_feature

    def build(self, input_shape):
        pass


    def call(self, x):
        v,q = x
        v = self.visual_head(v)
        x = (v,q)
        logits = self.model_feature(x)

        return logits


