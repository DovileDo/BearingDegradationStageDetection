import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.fft import rfft, rfftfreq
from scipy import signal
import sklearn
import os

# AutoEncoder architecture

input_freq = keras.Input(shape=(641,))
# "encoded" is the encoded representation of the input
encoded = keras.layers.Dense(256, activation='relu')(input_freq)
encoded = keras.layers.Dense(128, activation='relu')(encoded)
encoded = keras.layers.Dense(64, activation='relu')(encoded)
encoded = keras.layers.Dense(32, activation='relu')(encoded)
encoded = keras.layers.Dense(8, activation='relu')(encoded)

# "decoded" is the lossy reconstruction of the input
decoded = keras.layers.Dense(8, activation='relu')(encoded)
decoded = keras.layers.Dense(32, activation='relu')(decoded)
decoded = keras.layers.Dense(64, activation='relu')(decoded)
decoded = keras.layers.Dense(128, activation='relu')(decoded)
decoded = keras.layers.Dense(641, activation='linear')(decoded)

# This model maps an input to its reconstruction
autoencoder = keras.Model(input_freq, decoded)
# This model maps an input to its encoded representation
encoder = keras.Model(input_freq, encoded)
autoencoder.compile(optimizer='adam', loss='mae')

def anomaly_score(X_true, X_pred):

    if X_true.shape != X_pred.shape:
        raise ValueError(f"arrays must be of the same shape. X_true is {X_true.shape} while X_pred is {X_pred.shape}")

    score =  np.abs(X_true - X_pred).mean(axis=1)                

    return score

def calc_anomaly_treshold(x):

    threshold = np.mean(x) + 3 * np.std(x)

    return threshold
    
def correct_labels(Hl, Al):
    labelsb = Al
    labelsb = np.where(labelsb == 1, 3, 2)
    avg = []
    for i in range(3):
        avg.append(np.sum(np.where(Hl==i))/len(np.where(Hl==i)[0]))
    new = np.copy(Hl)
    for i in range(3):
        idx = np.argmin(avg)
        new[np.where(Hl==idx)] = i
        avg[idx] = float('inf')
    labelsb[:Hl.shape[0]] = new     
    return labelsb

def get_AElabels(Xt, Xf):
    autoencoder.compile(optimizer='adam', loss='mae')
    autoencoder.fit(Xt[:,:641], Xt[:,:641], epochs=100, batch_size=64, shuffle=True, verbose=False)
    
    #Horizontal vibration encoding
    Hencoded_f = encoder.predict(Xt[:,:641])
    
    Hdecoded_f = autoencoder.predict(Xf[:,:641])
    score = anomaly_score(Xf[:,:641], Hdecoded_f)
    anomaly_threshold = calc_anomaly_treshold(anomaly_score(Xt[:,:641], autoencoder.predict(Xt[:,:641])))
    
    autoencoder.compile(optimizer='adam', loss='mae')
    autoencoder.fit(Xt[:,641:], Xt[:,641:], epochs=100, batch_size=64, shuffle=True, verbose=False)
    
    #Vertical vibration encoding
    Vencoded_f = encoder.predict(Xt[:,641:])    
    
    Encoding = np.concatenate((Hencoded_f, Vencoded_f), axis=1)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(Encoding)
    kMeanslabels = kmeans.labels_
    
    Alabels = np.where(score > anomaly_threshold, 1, 0)

    return correct_labels(kMeanslabels, Alabels)