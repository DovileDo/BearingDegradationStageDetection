import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.fft import rfft, rfftfreq
from scipy import signal
from sklearn.decomposition import PCA
import sklearn
import os


def correct_labels(seq):
    l = []
    for i in range(4):
        l.append(np.sum(np.where(seq == i)) / len(np.where(seq == i)[0]))
    new = np.copy(seq)
    for i in range(4):
        idx = np.argmin(l)
        new[np.where(seq == idx)] = i
        l[idx] = float("inf")
    return new


def get_PCAlabels(Xt):
    xh_norm = sklearn.preprocessing.normalize(Xt[:, :641])
    xv_norm = sklearn.preprocessing.normalize(Xt[:, 641:])
    pcah = PCA(n_components=40)
    pcah.fit(xh_norm)
    pcav = PCA(n_components=40)
    pcav.fit(xv_norm)
    reducedh = pcah.transform(xh_norm)
    reducedv = pcav.transform(xv_norm)
    reduced = np.concatenate((reducedh, reducedv), axis=1)
    kmeans = KMeans(n_clusters=4, random_state=0).fit(reduced)
    labels = kmeans.labels_

    return correct_labels(labels)
