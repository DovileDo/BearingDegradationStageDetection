import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib as mpl
from random import randint
from NNclassifier import create_model

# read in train data
data = pd.read_csv("../../data/Xtrain.csv", header=None)
X = data.values
data = pd.read_csv("../../data/Xtrain_time.csv")
Xtime = data.values
data = pd.read_csv("../../data/labels/train_AElabels.csv")
Y = keras.utils.to_categorical(data.values, num_classes=4)
data = pd.read_csv("../../data/labels/train_PCAlabels.csv")
Ypca = keras.utils.to_categorical(data.values, num_classes=4)
Yrand = [randint(0, 3) for p in range(0, len(Y))]
Yrand = keras.utils.to_categorical(Yrand, num_classes=4)

AEclassifier = create_model()
AEhistory = AEclassifier.fit(
    [X[:, :641], X[:, 641:], Xtime], Y, batch_size=512, epochs=15, shuffle=True
)

AEclassifier.save_weights("../../models/AE/AEclassifier")

PCAclassifier = create_model()
PCAhistory = PCAclassifier.fit(
    [X[:, :641], X[:, 641:], Xtime], Ypca, batch_size=512, epochs=100, shuffle=True
)

PCAclassifier.save_weights("../../models/PCA/PCAclassifier")

RANDclassifier = create_model()
RANDhistory = RANDclassifier.fit(
    [X[:, :641], X[:, 641:], Xtime], Yrand, batch_size=512, epochs=100, shuffle=True
)

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False

plt.plot(AEhistory.history["acc"], label="AElabels")
plt.plot(PCAhistory.history["acc"], label="PCAlabels")
plt.plot(RANDhistory.history["acc"], label="RANDOMlabels")
plt.xlabel("Epoch")
plt.ylabel("Trainig accuracy")
plt.legend(loc="best")
plt.savefig("../../reports/figures/trainingAcc.png", dpi=100)
