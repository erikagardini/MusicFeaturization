"""Using the annotation file, assign the corresponding genre/instrument to each spectrogram.
If a song has more than one genre/instrument, the spectrogram is duplicated."""

import os
import matplotlib
import pandas as pd
matplotlib.use('agg')
from shutil import copyfile
from tqdm import tqdm


IMG_DIR = '../datasets/spectrogram_images_mp3_files/'
SPECTROGRAM_CLASS_DIR = '../datasets/spectrogram_images_class_files/'
SPECTROGRAM_GENRES_SUB_DIR = "../datasets/spectrogram_images_class_files/genres/"
SPECTROGRAM_INSTRUMENT_SUB_DIR = "../datasets/spectrogram_images_class_files/instrument/"

spectrograms_sub_dir = os.listdir(IMG_DIR)

def search_genre(file_name):
    file_name = '"' + file_name + '.mp3"'
    dataset = pd.read_csv("../datasets/annotations_final.csv")
    dataset_array = dataset.values
    dataset_cols = dataset.columns

    cols_strings = dataset_cols.astype(str)
    cols_splits = cols_strings[0].split("\t")

    class_ = []

    for el in dataset_array:
        res = el.astype(str)
        splits = res[0].split("\t")

        if file_name == splits[-1]:
            for i in range(1, len(splits) - 1):
                num = splits[i].split('"')
                if int(num[1]) == 1:
                    class_to_append = cols_splits[i].split('"')
                    class_.append(class_to_append[1])

            return splits[0], class_

    return -1, -1

def select_spectrograms():
    os.mkdir(SPECTROGRAM_CLASS_DIR)
    os.mkdir(SPECTROGRAM_GENRES_SUB_DIR)
    os.mkdir(SPECTROGRAM_INSTRUMENT_SUB_DIR)

    for sub_dir in spectrograms_sub_dir:
        if sub_dir != ".DS_Store":
            files = os.listdir(IMG_DIR + sub_dir)
            for f in tqdm(files):
                name_file_split = f.split(".")
                [id, class_] = search_genre(sub_dir + "/" + name_file_split[0])
                if id != -1 and class_ != -1:
                    save_genres(sub_dir, f, id, class_)


def save_genres(sub_dir, file_name, id, class_):
    genres = "hard_rock,classical,funcky,classical,soft_rock,jazz,spacey,folk,new_age,funk,middle_eastern,medieval," \
             "classic,electronic,chanting,opera,country,electro,reggae,tribal,dark,irish,electronica,operatic,arabic," \
             "trance,drone,heavy_metal,disco,deep,jungle,pop,celtic,orchestral,eastern,punk,blues,indian,rock,dance," \
             "jazzy,techno,house,oriental,rap,metal,hip_hop,choir,baroque"
    genres_split = genres.split(",")
    instrument = "bongos,harpsichord,sitar,clarinet,woodwind,horns,guitar,banjo,violins,synth,trumpet,percussion," \
                 "drum,bass,harpsicord,drums,organ,acoustic_guitar,electric_guitar,classical_guitar,violin," \
                 "horn,synthesizer,bells,harp,lute,oboe,viola,piano,flutes,sax,piano_solo,guitars,cello,flute,fiddle"
    instrument_split = instrument.split(",")

    for cl in class_:
        result = format_class(cl)
        copyfile(IMG_DIR + sub_dir + "/" + file_name, SPECTROGRAM_CLASS_DIR + result + "." + id + "_" +
                 file_name)
        if search(genres_split, result) == 1:
            copyfile(IMG_DIR + sub_dir + "/" + file_name, SPECTROGRAM_GENRES_SUB_DIR + result + "." + id + "_" +
                     file_name)
        if search(instrument_split, result) == 1:
            copyfile(IMG_DIR + sub_dir + "/" + file_name, SPECTROGRAM_INSTRUMENT_SUB_DIR + result + "." + id + "_" +
                     file_name)


def search(array, to_search):
    for el in array:
        if el == to_search:
            return 1
    return 0


def format_class(cl):
    result = ""
    res = cl.split(" ")
    #print(res)
    if len(res) > 1:
        for i in range(0, len(res) - 1):
            result = result + res[i] + "_"
        result = result + res[-1]
    else:
        result = res[0]

    return result

#MAIN

select_spectrograms()