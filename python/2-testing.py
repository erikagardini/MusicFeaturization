"""Test the model

References:
[1] Bahuleyan, H.
    Music Genre Classification using Machine Learning Techniques.
    arXiv:1804.01149
[2] https://github.com/HareeshBahuleyan/music-genre-classification"""

import os
import csv
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from keras.models import Model
import python.utils as utils

'''os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

sess = tf.Session(config=config)'''

IMG_HEIGHT = 216
IMG_WIDTH = 216
NUM_CLASSES = 6
BATCH_SIZE = 100
L2_LAMBDA = 0.001

one_hot = OneHotEncoder(n_values=NUM_CLASSES)

[train_files, train_labels, test_files, test_labels, val_files, val_labels] = utils.load_input()
TEST_STEPS = len(test_files)//BATCH_SIZE

model = utils.get_model()

os.system("cat ../models/splitted_best_model.z* > ../models/best_model.zip")
os.system("unzip ../models/best_model.zip -d ../models")
model.load_weights(filepath='../models/best_model/epoch_06_0.8451.h5')

metrics = ['categorical_accuracy']
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)

# Make Prediction on Test set
pred_probs = model.predict_generator(generator=utils.batch_generator(test_files, BATCH_SIZE), steps=TEST_STEPS)
pred = np.argmax(pred_probs, axis=-1)

path_prediction = "../results/prediction"
path_prediction_probabilities = "../results/prediction_probabilities"
np.save(path_prediction, pred)
np.save(path_prediction_probabilities, pred_probs)

#Save penultimate layer (music_dataset)
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer("activation_1").output)
prediction = intermediate_layer_model.predict_generator(generator = utils.batch_generator(test_files, BATCH_SIZE), steps=TEST_STEPS)

line = []
lines = []
for i in range(0, prediction.shape[0]):
    line.append(test_files[i])
    for el in prediction[i]:
        line.append(el)
    print(line)
    lines.append(line)
    line = []

with open('../results/music_dataset_testing.csv', 'a') as csvFile:
    writer = csv.writer(csvFile)
    for el in lines:
        writer.writerow(el)
csvFile.close()


