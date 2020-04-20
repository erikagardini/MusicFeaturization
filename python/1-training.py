"""Train the model

References:
[1] Bahuleyan, H.
    Music Genre Classification using Machine Learning Techniques.
    arXiv:1804.01149
[2] https://github.com/HareeshBahuleyan/music-genre-classification"""

import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from keras import callbacks
import python.utils as utils

'''os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

sess = tf.Session(config=config)'''

RES_DIR = "../models/"
IMG_HEIGHT = 216
IMG_WIDTH = 216
NUM_CLASSES = 6
NUM_EPOCHS = 20
BATCH_SIZE = 100
L2_LAMBDA = 0.001

one_hot = OneHotEncoder(n_values=NUM_CLASSES)

[train_files, train_labels, test_files, test_labels, val_files, val_labels] = utils.load_input()

label_array = []
for el in train_labels:
    label_array.append(el)
for el in test_labels:
    label_array.append(el)
for el in val_labels:
    label_array.append(el)

cl_weight = compute_class_weight(class_weight='balanced', classes=np.unique(label_array), y=label_array)

model = utils.get_model()

filepath= RES_DIR + "epoch_{epoch:02d}_{val_categorical_accuracy:.4f}.h5"
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_categorical_accuracy', verbose=0, save_best_only=False)
callbacks_list = [checkpoint]

metrics = ['categorical_accuracy']
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)

STEPS_PER_EPOCH = len(train_files)//BATCH_SIZE
VAL_STEPS = len(val_files)//BATCH_SIZE

history = model.fit_generator(generator=utils.batch_generator(train_files, BATCH_SIZE),
                              epochs=NUM_EPOCHS,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              class_weight=cl_weight,
                              validation_data=utils.batch_generator(val_files, BATCH_SIZE),
                              validation_steps=VAL_STEPS,
                              callbacks=callbacks_list)

# Save scores on train and validation sets
with open(RES_DIR + 'history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
