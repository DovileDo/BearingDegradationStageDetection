import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import matplotlib as mpl
import os

import sys
sys.path.append('../data')

from TransformToFrequencyDomain import ToFrequency

sys.path.append('../models')

from AutoEncoder import get_AElabels
from PCAlabeling import get_PCAlabels

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False


def accuracy(Y_true, predictions):
    return f1_score(np.argmax(Y_true, axis=1), np.argmax(predictions, axis=1), average=None)

Manual_labels = pd.read_csv('../../data/labels/train_ManualLabels.csv')
AE_labacc = []
PCA_labacc = []
start = 0

rootdir = '../../data/MergedTrain_files/'
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        print(file)
        df = pd.read_csv(rootdir+ '/' +file)
        X_freq = ToFrequency(df)

        #labeling accuracy
        X_truncated = X_freq[:int(len(X_freq)*0.8),:]
        AElabels = get_AElabels(X_truncated, X_freq)
        end = start + len(AElabels)
        Manual = Manual_labels[start:end]
        AE_acc = accuracy(keras.utils.to_categorical(Manual.values, num_classes=4), keras.utils.to_categorical(AElabels, num_classes=4))
        AE_labacc.append(AE_acc*100)
        
        PCAlabels = get_PCAlabels(X_freq)
        PCA_acc = accuracy(keras.utils.to_categorical(Manual.values, num_classes=4), keras.utils.to_categorical(PCAlabels, num_classes=4))
        PCA_labacc.append(PCA_acc*100)
        start = len(AElabels)

    AE_labacc = np.array(AE_labacc).reshape(6, 4)
    PCA_labacc = np.array(PCA_labacc).reshape(6, 4)       
    tlabels = ['Healthy', 'Stage 1', 'Stage 2', 'Stage 3']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,3))
    fig.subplots_adjust(hspace=0.2)
    ax1.boxplot([AE_labacc[:,0], AE_labacc[:,1], AE_labacc[:,2], AE_labacc[:,3]], labels=tlabels)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Labeling accuracy")
    ax2.boxplot([PCA_labacc[:,0], PCA_labacc[:,1], PCA_labacc[:,2], PCA_labacc[:,3]], labels=tlabels)
    ax2.set_ylim(0, 100)
    plt.savefig('../../reports/figures/labelingAcc.png', bbox_inches='tight', dpi=100)
    plt.close()