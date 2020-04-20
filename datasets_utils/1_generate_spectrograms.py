"""Get the spectrograms from the mp3 files.

References:
[1] Bahuleyan, H.
    Music Genre Classification using Machine Learning Techniques.
    arXiv:1804.01149
[2] https://github.com/HareeshBahuleyan/music-genre-classification"""

import os
import matplotlib

matplotlib.use('agg')
from matplotlib import cm
from tqdm import tqdm
import pylab
import librosa
from librosa import display
import numpy as np

MP3_DIR = '../datasets/mp3_files/'
IMG_DIR = '../datasets/spectrogram_images_mp3_files/'
mp3_sub_dir = os.listdir(MP3_DIR)

os.mkdir(IMG_DIR)
for sub_dir in mp3_sub_dir:
    #print(sub_dir)
    if sub_dir != ".DS_Store":
        files = os.listdir(MP3_DIR + sub_dir)
        os.mkdir(IMG_DIR + sub_dir)
        #print(files)
        for f in tqdm(files):
            try:
                # Read mp3-file
                y, sr = librosa.load(MP3_DIR + sub_dir + "/" + f, sr=22050)  # Use the default sampling rate of 22,050 Hz

                # Pre-emphasis filter
                pre_emphasis = 0.97
                y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

                # Compute spectrogram
                M = librosa.feature.melspectrogram(y, sr,
                                                   fmax=sr / 2,  # Maximum frequency to be used on the on the MEL scale
                                                   n_fft=2048,
                                                   hop_length=512,
                                                   n_mels=96,  # As per the Google Large-scale audio CNN paper
                                                   power=2)  # Power = 2 refers to squared amplitude

                # Power in DB
                log_power = librosa.power_to_db(M, ref=np.max)  # Covert to dB (log) scale

                # Plotting the spectrogram and save as JPG without axes (just the image)
                pylab.figure(figsize=(3, 3))
                pylab.axis('off')
                pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edge
                librosa.display.specshow(log_power, cmap=cm.jet)
                pylab.savefig(IMG_DIR + sub_dir + "/" + f[:-4] + '.jpg', bbox_inches=None, pad_inches=0)
                pylab.close()

            except Exception as e:
                print(f, e)
                pass