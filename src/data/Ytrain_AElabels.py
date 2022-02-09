import numpy as np
import pandas as pd
import os
from TransformToFrequencyDomain import ToFrequency

import sys
sys.path.insert(1, '/mnt/c/Users/doju/OneDrive - ITU/BearingDegradationStageDetection/src/models')

from AutoEncoder import get_AElabels

# Folder with merged data files
rootdir = '../../data/MergedTrain_files/'

X = np.empty((0,1282), float)
Y = []
    
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        print(file)
        df = pd.read_csv(rootdir+ '/' +file)#, sep='\t')
        X = ToFrequency(df)
        X_truncated = X[:int(len(X)*0.8),:]
        labels = get_labels(X_truncated, X)
        Y.extend(labels.tolist())

pd.DataFrame(Y).to_csv('../../data/labels/AElabels.csv', index=False)