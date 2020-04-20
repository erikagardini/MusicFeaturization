"""Get the subset of spectrograms belonging to the following genres:

- classical
- baroque
- opera
- medieval
- jazz
-rock

If a spectogram has more than one genre, the duplicated are removed according
to the following priority order:

- baroque
- medieval
- opera
- classical
- jazz
- rock

Eg. if spect1 belongs to medieval and classical, only medieval is kept."""

import os
from shutil import copyfile

IMG_DIR = "../datasets/spectrogram_images_class_files/genres/"
IMG_DEST = "../datasets/spectrogram_images_class_files/sub_genres_no_doubles/"

label_dict = {'classical': 0,
              'baroque': 1,
              'opera': 2,
              'medieval': 3,
              'jazz': 4,
              'rock': 5,
              }

def save_files(files, names, class_name):
    count = 0
    for file in files:
        vals = file[:-4].split('.')
        if label_dict.get(vals[0]) != None:
            if vals[0] == class_name and vals[1] not in names:
                names.append(vals[1])
                copyfile(IMG_DIR + file, IMG_DEST + file)
                count = count + 1
    return names, count

all_files = os.listdir(IMG_DIR)
os.mkdir(IMG_DEST)

names = []
[names, count] = save_files(all_files, names, "baroque")
[names, count] = save_files(all_files, names, "medieval")
[names, count] = save_files(all_files, names, "opera")
[names, count] = save_files(all_files, names, "classical")
[names, count] = save_files(all_files, names, "jazz")
[names, count] = save_files(all_files, names, "rock")
