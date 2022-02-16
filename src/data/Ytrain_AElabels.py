import numpy as np
import pandas as pd
import os
from TransformToFrequencyDomain import ToFrequency
from pathlib import Path
import sys

sys.path.append('../models')

from AutoEncoder import get_AElabels


if __name__ == '__main__':


    # Folder with merged data files
    rootdir = '../../data/MergedTrain_files/'

    Y = []
        
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            print(file)
            df = pd.read_csv(rootdir+ '/' +file)
            X = ToFrequency(df)
            X_truncated = X[:int(len(X)*0.8),:]
            labels = get_AElabels(X_truncated, X)
            Y.extend(labels.tolist())

    pd.DataFrame(Y).to_csv('../../data/labels/train_AElabels.csv', index=False)