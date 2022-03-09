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

sys.path.append("../data")

from TransformToFrequencyDomain import ToFrequency
from TransformToTimeDomain import ToTime

sys.path.append("../models")

from NNclassifier import create_model
from AutoEncoder import get_AElabels
from PCAlabeling import get_PCAlabels

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False


def movingAvg(array):
    window_size = 5
    return convolve2d(array, np.ones((window_size, 1)), "valid") / window_size


def testpost(predictions, PCA=False):
    smooth = movingAvg(predictions)
    plt.figure(figsize=(8, 2), dpi=80)
    plt.plot(smooth[:, 0], label="Healthy bearing")
    plt.plot(smooth[:, 1], label="Fault stage 1")
    plt.plot(smooth[:, 2], label="Fault stage 2")
    plt.plot(smooth[:, 3], label="Fault stage 3")
    plt.legend(bbox_to_anchor=(1, 1))
    plt.xlabel("Observation #")
    plt.ylabel("Posterior probability")
    if PCA:
        plt.savefig(
            "../../reports/figures/posterior/PCALb" + file + ".png",
            bbox_inches="tight",
            dpi=100,
        )
        plt.close()
    else:
        plt.savefig(
            "../../reports/figures/posterior/AELb" + file + ".png",
            bbox_inches="tight",
            dpi=100,
        )
        plt.close()


def accuracy(Y_true, predictions):
    return f1_score(
        np.argmax(Y_true, axis=1), np.argmax(predictions, axis=1), average=None
    )


def overlap(pred):
    overlaped = [100] * 3
    smooth = movingAvg(pred)
    Y_pred = np.argmax(smooth, axis=1)
    if 1 in Y_pred:
        region = Y_pred[
            np.min(np.where(Y_pred == 1)) : (np.max(np.where(Y_pred == 1))) + 1
        ]
        overlaped[0] = (len(np.where(region != 1)[0]) / len(region)) * 100
    if 2 in Y_pred:
        region = Y_pred[
            np.min(np.where(Y_pred == 2)) : (np.max(np.where(Y_pred == 2))) + 1
        ]
        overlaped[1] = (len(np.where(region != 2)[0]) / len(region)) * 100
    if 3 in Y_pred:
        region = Y_pred[
            np.min(np.where(Y_pred == 3)) : (np.max(np.where(Y_pred == 3))) + 1
        ]
        overlaped[2] = (len(np.where(region != 3)[0]) / len(region)) * 100
    return overlaped


def fault(pred):
    smooth = movingAvg(pred)
    Y_pred = np.argmax(smooth, axis=1)
    if 2 not in Y_pred:
        fault = Y_pred[np.min(np.where(Y_pred == 3)) :]
        fault_point = np.min(np.where(Y_pred == 3))
    else:
        start2 = np.min(np.where(Y_pred == 2))
        start3 = np.min(np.where(Y_pred == 3))
        fault = Y_pred[np.min([start2, start3]) :]
        fault_point = np.min([start2, start3])
    return (
        (len(Y_pred[fault_point:]) / len(Y_pred)) * 100,
        (len(np.where(fault < 2)[0]) / len(fault)) * 100,
        len(Y_pred),
    )


AEmodel = create_model()
AEmodel.load_weights("../../models/AE/AEclassifier")

PCAmodel = create_model()
PCAmodel.load_weights("../../models/PCA/PCAclassifier")

AE_testacc = []
PCA_testacc = []
AE_Overlap = []
PCA_Overlap = []
AE_failPoint = []
PCA_failPoint = []
AE_HealthyAfterFail = []
PCA_HealthyAfterFail = []
AE_len = []

rootdir = "../../data/MergedTest_files/"
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        print(file)
        df = pd.read_csv(rootdir + "/" + file)
        X_freq = ToFrequency(df)
        X_time = ToTime(df)
        AEpred = AEmodel.predict([X_freq[:, :641], X_freq[:, 641:], X_time])
        PCApred = PCAmodel.predict([X_freq[:, :641], X_freq[:, 641:], X_time])

        # classifier predictions for test set
        testpost(AEpred)
        testpost(PCApred, PCA=True)

        # test accuracy
        X_truncated = X_freq[: int(len(X_freq) * 0.8), :]
        AElabels = get_AElabels(X_truncated, X_freq)
        AE_acc = accuracy(keras.utils.to_categorical(AElabels, num_classes=4), AEpred)
        AE_testacc.append(AE_acc)

        PCAlabels = get_PCAlabels(X_freq)
        PCA_acc = accuracy(
            keras.utils.to_categorical(PCAlabels, num_classes=4), PCApred
        )
        PCA_testacc.append(PCA_acc)

        # prediction overlap
        AE_Overlap.append(overlap(AEpred))
        PCA_Overlap.append(overlap(PCApred))

        # Fault point and back to predicting lower stage (healthy or stage 1 after fault)
        AE_failPoint.append(fault(AEpred)[0])
        PCA_failPoint.append(fault(PCApred)[0])
        AE_HealthyAfterFail.append(fault(AEpred)[1])
        PCA_HealthyAfterFail.append(fault(PCApred)[1])
        AE_len.append(fault(AEpred)[2])

    AE_testacc = np.array(AE_testacc).reshape(11, 4)
    PCA_testacc = np.array(PCA_testacc).reshape(11, 4)
    bplabels = ["Healthy", "Stage 1", "Stage 2", "Stage 3"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    fig.subplots_adjust(hspace=0.2)
    ax1.boxplot(
        [AE_testacc[:, 0], AE_testacc[:, 1], AE_testacc[:, 2], AE_testacc[:, 3]],
        labels=bplabels,
    )
    ax1.set_ylabel("Test accuracy")
    ax2.boxplot(
        [PCA_testacc[:, 0], PCA_testacc[:, 1], PCA_testacc[:, 2], PCA_testacc[:, 3]],
        labels=bplabels,
    )
    plt.savefig("../../reports/figures/testAcc.png", bbox_inches="tight", dpi=100)
    plt.close()

    AE_Overlap = np.array(AE_Overlap).reshape(11, 3)
    PCA_Overlap = np.array(PCA_Overlap).reshape(11, 3)
    bpoverlap = ["Stage 1", "Stage 2", "Stage 3"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    fig.subplots_adjust(hspace=0.2)
    ax1.boxplot(
        [AE_Overlap[:, 0], AE_Overlap[:, 1], AE_Overlap[:, 2]], labels=bpoverlap
    )
    ax1.set_ylabel("% overlaped by other stages")
    ax2.boxplot(
        [PCA_Overlap[:, 0], PCA_Overlap[:, 1], PCA_Overlap[:, 2]], labels=bpoverlap
    )
    plt.savefig("../../reports/figures/overlap.png", bbox_inches="tight", dpi=100)
    plt.close()

    AE_failPoint = np.array(AE_failPoint).reshape(11, 1)
    PCA_failPoint = np.array(PCA_failPoint).reshape(11, 1)
    AE_HealthyAfterFail = np.array(AE_HealthyAfterFail).reshape(11, 1)
    PCA_HealthyAfterFail = np.array(PCA_HealthyAfterFail).reshape(11, 1)
    AE_len = np.array(AE_len).reshape(11, 1)
    Fault_results = np.concatenate(
        (
            AE_HealthyAfterFail,
            PCA_HealthyAfterFail,
            AE_failPoint,
            PCA_failPoint,
            AE_len,
        ),
        axis=1,
    )
    pd.DataFrame(Fault_results).to_csv("../../reports/fault_results.csv")
