import numpy as np
import torch
import sys
import pylab
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
from os import listdir
from os.path import isfile, join

from collections import Counter
from sklearn.preprocessing import LabelEncoder

from librosa.core import load
from librosa.feature import melspectrogram
from librosa import power_to_db

from model import genreNet
from config import MODELPATH
from config import GENRES

import warnings
warnings.filterwarnings("ignore")

def main(argv):
    plot = False
    verbose = True

    if len(argv) < 1:
        print("Usage: python3 get_genre.py audiopath [-plot]")
        exit()
    if len(argv) == 2 and argv[1] == '-plot' :
        plot = True
    if len(argv) == 2 and argv[1] == '-no-print' :
        verbose = False

    if (os.path.isfile(argv[0])):
        get_genre(argv[0], plot, print)
    if (os.path.isdir(argv[0])):
        onlyfiles = [f for f in listdir(argv[0]) if isfile(join(argv[0], f))]

        res = list()
        for audiofile in onlyfiles:
            res.append(get_genre(join(argv[0], audiofile), plot, verbose))

        res = [i for g in res for i in g]
        stats = dict()
        for genre in res:
            print(genre)
            if genre[0] not in stats.keys():
                stats[genre[0]] = 0
            stats[genre[0]] += genre[1]
        total = 0
        for key, val in stats.items():
            total += val
        for key, val in stats.items():
            stats[key] = (val / total) * 100
            print("%10s: \t%.2f\t%%" % (key, stats[key]))


def get_genre(audio_path, plot, verbose):


    le = LabelEncoder().fit(GENRES)
    # ------------------------------- #
    ## LOAD TRAINED GENRENET MODEL
    net         = genreNet()
    net.load_state_dict(torch.load(MODELPATH, map_location='cpu'))
    # ------------------------------- #
    ## LOAD AUDIO
    y, sr       = load(audio_path, mono=True, sr=22050)
    # ------------------------------- #
    ## GET CHUNKS OF AUDIO SPECTROGRAMS
    S           = melspectrogram(y, sr).T
    S           = S[:-1 * (S.shape[0] % 128)]
    num_chunk   = S.shape[0] / 128
    data_chunks = np.split(S, num_chunk)

    ## Plot the spectrogram
    if plot:
        plt.figure(figsize=(10, 4))
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-frequency spectrogram')
        plt.tight_layout()
        plt.show()
    # ------------------------------- #

    ## CLASSIFY SPECTROGRAMS
    genres = np.array([])
    for i, data in enumerate(data_chunks):

        data    = torch.FloatTensor(data).view(1, 1, 128, 128)
        preds   = net(data)
        pred_val, pred_index    = preds.max(1)
        pred_index              = pred_index.data.numpy()[0]
        pred_val                = np.exp(pred_val.data.numpy()[0])
        pred_genre              = le.inverse_transform([pred_index])
        if pred_val >= 0.5:
            genres = np.append(genres, pred_genre)
    # ------------------------------- #
    s           = float(sum([v for k,v in dict(Counter(genres)).items()]))
    pos_genre   = sorted([(k, v/s*100 ) for k,v in dict(Counter(genres)).items()], key=lambda x:x[1], reverse=True)
    if verbose:
        print(audio_path)
        for genre, pos in pos_genre:
            print("%10s: \t%.2f\t%%" % (genre, pos))
    return pos_genre

if __name__ == '__main__':
    main(sys.argv[1:])
