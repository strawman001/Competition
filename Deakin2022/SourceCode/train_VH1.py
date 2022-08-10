import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.saving import hdf5_format
import matplotlib.pyplot as plt
import json
import base64
import os
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.utils import plot_model
import random
import h5py
from sklearn.metrics import classification_report
import pickle
import re
import nltk

import pandas as pd
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps
import time
import gensim

from .model_VH1 import *
from .dataset_image import *


if __name__ == "__main__":
    with open("./word.pkl", 'rb') as f:
        word_pkl = pickle.load(f)

    word_index = word_pkl["word_index"]
    word_num = word_pkl["word_num"]
    emb_table = word_pkl["emb_table"]

    model = BuTdModel(word_num, emb_table)

    model.compile(keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],
                  run_eagerly=True)

    from datetime import datetime
    import os

    root_logdir = "./logs/"
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(root_logdir, run_id)


    def scheduler(epoch, lr):
        if epoch == 10:
            return lr * 0.1
        else:
            return lr


    checkpoint_path = "./model/model-1-{epoch:04d}.ckpt"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                           save_weights_only=True,
                                           verbose=1),
        tf.keras.callbacks.LearningRateScheduler(scheduler),
        tf.keras.callbacks.TensorBoard(
            log_dir=logdir,
            histogram_freq=1
        )
    ]

    train_dataset = make_coco_train_img_dataset()
    val_dataset = make_simpson_val_img_dataset()

    history = model.fit(
        train_dataset,
        epochs=20,
        validation_data=val_dataset,
        callbacks=callbacks
    )

    print(history)