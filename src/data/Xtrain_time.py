import numpy as np
import pandas as pd
import os
from scipy.stats import kurtosis
import scipy.stats as stats
from scipy.signal import find_peaks
from scipy import signal
from TransformToTimeDomain import ToTime

# Data from merged files to np arrays ready for training

X = np.empty((0, 26), float)

rootdir = "../../data/MergedTrain_files"
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        print(file)
        df = pd.read_csv(rootdir + "/" + file)
        features = ToTime(df)
        X = np.append(X, features.values, axis=0)

pd.DataFrame(X).to_csv("../../data/Xtrain_time.csv", index=False)
