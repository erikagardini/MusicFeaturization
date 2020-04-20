#References:
#[1] Bahuleyan, H.
    # Music Genre Classification using Machine Learning Techniques.
        # arXiv:1804.01149
#[2] https://github.com/HareeshBahuleyan/music-genre-classification

import numpy as np
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from keras import layers
from keras import regularizers
from keras import models
from keras.layers import Dense
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
from keras import applications
from keras import backend as K

INPUT_DIR = '../datasets/train_test_val/'
IMG_DIR = "../datasets/spectrogram_images_class_files/sub_genres_no_doubles/"
IMG_HEIGHT = 216
IMG_WIDTH = 216
NUM_CLASSES = 6
NUM_EPOCHS = 50
BATCH_SIZE = 100
L2_LAMBDA = 0.001

label_dict = {'classical': 0,
              'baroque': 1,
              'opera': 2,
              'medieval': 3,
              'jazz': 4,
              'rock': 5
              }

one_hot = OneHotEncoder(n_values=NUM_CLASSES)


def load_input():

    with open(INPUT_DIR + 'train_files.txt', 'r') as file:
        lines = file.readlines()
        train_files = []
        for line in lines:
            train_files.append(line[0:-1])
    file.close()

    with open(INPUT_DIR + 'test_files.txt', 'r') as file:
        lines = file.readlines()
        test_files	= []
        for line in lines:
            test_files.append(line[0:-1])
    file.close()

    with open(INPUT_DIR + 'val_files.txt', 'r') as file:
        lines = file.readlines()
        val_files	= []
        for line in lines:
            val_files.append(line[0:-1])
    file.close()

    with open(INPUT_DIR + 'train_labels.txt', 'r') as file:
        train_labels = file.readlines()
    file.close()

    with open(INPUT_DIR + 'test_labels.txt', 'r') as file:
        test_labels = file.readlines()
    file.close()

    with open(INPUT_DIR + 'val_labels.txt', 'r') as file:
        val_labels = file.readlines()
    file.close()

    train_labels = np.array(train_labels).astype(int)
    test_labels = np.array(test_labels).astype(int)
    val_labels = np.array(val_labels).astype(int)

    return [train_files, train_labels, test_files, test_labels, val_files, val_labels]


def get_model():
    conv_base = applications.VGG16(include_top=False, weights='imagenet', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(512, name='dense_1', kernel_regularizer=regularizers.l2(L2_LAMBDA)))
    model.add(layers.Dropout(rate=0.3, name='dropout_1'))  # Can try varying dropout rates
    model.add(layers.Activation(activation='relu', name='activation_1'))
    model.add(Dense(NUM_CLASSES, name='dense_output'))
    get_custom_objects().update({'custom_activation': Activation(softmax)})
    model.add(Activation(softmax))
    conv_base.trainable = False

    return model


def load_batch(file_list):
    img_array = []
    idx_array = []
    label_array = []

    for file_ in file_list:
        im = Image.open(IMG_DIR + file_)
        im = im.resize((IMG_WIDTH, IMG_HEIGHT), Image.ANTIALIAS)
        img_array.append(np.array(im))

        vals = file_[:-4].split('.')
        idx_array.append(vals[1])
        label_array.append([label_dict[vals[0]]])

    label_array = one_hot.fit_transform(label_array).toarray()
    img_array = np.array(img_array) / 255.0  # Normalize RGB

    return img_array, np.array(label_array), np.array(idx_array)


def batch_generator(files, BATCH_SIZE):
    L = len(files)

    # this line is just to make the generator infinite, keras needs that
    while True:

        batch_start = 0
        batch_end = BATCH_SIZE

        while batch_start < L:
            limit = min(batch_end, L)
            file_list = files[batch_start: limit]
            batch_img_array, batch_label_array, batch_idx_array = load_batch(file_list)

            yield (batch_img_array, batch_label_array)  # a tuple with two numpy arrays with batch_size samples

            batch_start += BATCH_SIZE
            batch_end += BATCH_SIZE


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = K.exp(x - K.max(x))
    return e_x / K.sum(e_x)