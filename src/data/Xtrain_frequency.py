import numpy as np
import pandas as pd
import time
from scipy.fft import rfft, rfftfreq
from scipy import signal
import os
from TransformToFrequencyDomain import ToFrequency

# Data from merged files to np arrays ready for training

X = np.empty((0,1282), float)

rootdir = '../../data/MergedTrain_files/'
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        df = pd.read_csv(rootdir+ '/' +file)
        x = ToFrequency(df)
        X = np.append(X, x, axis=0)            
np.savetxt("../../data/Xtrain.csv", X, delimiter=",")