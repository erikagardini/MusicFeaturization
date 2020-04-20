"""Save the list of files belonging to the training set, the testing set and
the validation set together with the corresponding labels."""

import os
from sklearn.model_selection import train_test_split

IMG_DIR = "../datasets/spectrogram_images_class_files/sub_genres_no_doubles/"
DEST_DIR = "../datasets/train_test_val/"
label_dict = {'classical': 0,
              'baroque': 1,
              'opera': 2,
              'medieval': 3,
              'jazz': 4,
              'rock': 5,
              }

all_files = os.listdir(IMG_DIR)
os.mkdir(DEST_DIR)
sub_files = []

label_array = []
for file_ in all_files:
    vals = file_[:-4].split('.')
    if label_dict.get(vals[0]) != None:
        sub_files.append(file_)
        label_array.append(label_dict[vals[0]])

# Train-val-test split of files
train_files, test_files, train_labels, test_labels = train_test_split(sub_files, label_array,
                                                                      random_state = 10, test_size = 0.2)
train_files, val_files, train_labels, val_labels = train_test_split(train_files, train_labels,
                                                                    random_state = 10, test_size = 0.1)

with open(DEST_DIR + 'train_files.txt', 'w') as file:
    for el in train_files:
        file.write(el)
        file.write("\n")
file.close()

with open(DEST_DIR + 'test_files.txt', 'w') as file:
    for el in test_files:
        file.write(el)
        file.write("\n")
file.close()

with open(DEST_DIR + 'val_files.txt', 'w') as file:
    for el in val_files:
        file.write(el)
        file.write("\n")
file.close()

with open(DEST_DIR + 'train_labels.txt', 'w') as file:
    for el in train_labels:
        file.write(str(el))
        file.write("\n")
file.close()

with open(DEST_DIR + 'test_labels.txt', 'w') as file:
    for el in test_labels:
        file.write(str(el))
        file.write("\n")
file.close()

with open(DEST_DIR + 'val_labels.txt', 'w') as file:
    for el in val_labels:
        file.write(str(el))
        file.write("\n")
file.close()