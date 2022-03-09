import numpy as np
import pandas as pd
import os
from TransformToFrequencyDomain import ToFrequency

import sys

sys.path.append("../models")

from PCAlabeling import get_PCAlabels

if __name__ == "__main__":

    # Folder with merged data files
    rootdir = "../../data/MergedTrain_files/"

    Y = []

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            print(file)
            df = pd.read_csv(rootdir + "/" + file)  # , sep='\t')
            X = ToFrequency(df)
            labels = get_PCAlabels(X)
            Y.extend(labels.tolist())

    pd.DataFrame(Y).to_csv("../../data/labels/train_PCAlabels.csv", index=False)
