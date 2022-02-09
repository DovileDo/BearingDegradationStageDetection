import numpy as np
import pandas as pd
import os
from TransformToFrequencyDomain import ToFrequency

import sys
sys.path.insert(1, '/mnt/c/Users/doju/OneDrive - ITU/BearingDegradationStageDetection/src/models')

from PCAlabeling import get_PCAlabels

# Folder with merged data files
rootdir = '../../data/MergedTrain_files/'

X = np.empty((0,1282), float)
Y = []
    
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        print(file)
        df = pd.read_csv(rootdir+ '/' +file)#, sep='\t')
        X = ToFrequency(df)
        labels = get_PCAlabels(X)
        Y.extend(labels.tolist())
  
pd.DataFrame(Y).to_csv('../../data/labels/PCAlabels.csv', index=False)