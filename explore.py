import os

import matplotlib
import IPython.display as ipd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
import librosa
import librosa.display

import utils
import tensorflow as tf

import csv

import glob


# Directory where mp3 are stored.
AUDIO_DIR = '/home/dan/fma_small/'


def get_audio_path(audio_dir, track_id):
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(audio_dir, tid_str[:3], tid_str + '.mp3')


# https://nbviewer.jupyter.org/github/mdeff/fma/blob/outputs/usage.ipynb

plt.rcParams['figure.figsize'] = (17, 5)

# https://www.analyticsvidhya.com/blog/2019/01/build-image-classification-model-10-minutes/
# https://colab.research.google.com/notebooks/gpu.ipynb#scrollTo=BlmQIFSLZDdc

def filter_csv():
    # Load metadata and features.
    tracks = utils.load('tracks.csv')

    small = tracks['set', 'subset'] <= 'small'

    genres = tracks.track['genre_top']
    sets = tracks.set

    a = pd.concat([genres, sets], axis=1)
    a.to_csv('my_file.csv')


for root, directories, filenames in os.walk(AUDIO_DIR):
    for filename in filenames:
        filepath = os.path.join(root, filename)
        if filepath.endswith(".png"):
            try:
                print(filepath)
            except Exception as e:
                print(filepath)
                print(e)



# train = tracks['set', 'split'] == 'training'
# val = tracks['set', 'split'] == 'validation'
# test = tracks['set', 'split'] == 'test'


# https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html#selection