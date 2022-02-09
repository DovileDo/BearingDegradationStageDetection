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
from TransformToFrequecnyDomain import ToFrequency
from TransformToTimeDomain import ToTime

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False


#read in test data
"""
data = pd.read_csv('../../data/Xtest.csv', header=None)
X_test = data.values
data = pd.read_csv('../../data/Xtest_time.csv')
Xtesttime = data.values
data = pd.read_csv('../../data/test_AElabels.csv')
Ytest = keras.utils.to_categorical(data.values, num_classes=4)
data = pd.read_csv('../../data/test_PCAlabels.csv')
Ypcatest = keras.utils.to_categorical(data.values, num_classes=4)
"""

def create_model():

    H_input = keras.Input(shape=(641,), name='horizontal')
    V_input = keras.Input(shape=(641,), name='vertical')
    meta_input = keras.Input(shape=(26,), name='meta_input')
    initializer = tf.keras.initializers.RandomNormal(seed=42)

    H = keras.layers.Dense(256, activation='relu', kernel_initializer=initializer, kernel_regularizer='l2')(H_input)
    H = keras.layers.Dense(128, activation='relu', kernel_initializer=initializer, kernel_regularizer='l2')(H)
    H = keras.layers.Dense(64, activation='relu', kernel_initializer=initializer, kernel_regularizer='l2')(H)
    H = keras.layers.Dense(32, activation='relu', kernel_initializer=initializer, kernel_regularizer='l2')(H)
    H = keras.layers.Dense(4, activation='relu', kernel_initializer=initializer)(H)

    V = keras.layers.Dense(256, activation='relu', kernel_initializer=initializer, kernel_regularizer='l2')(V_input)
    V = keras.layers.Dense(128, activation='relu', kernel_initializer=initializer, kernel_regularizer='l2')(V)
    V = keras.layers.Dense(64, activation='relu', kernel_initializer=initializer, kernel_regularizer='l2')(V)
    V = keras.layers.Dense(32, activation='relu', kernel_initializer=initializer, kernel_regularizer='l2')(V)
    V = keras.layers.Dense(4, activation='relu', kernel_initializer=initializer)(V)

    F = keras.layers.Dense(16, activation='relu', kernel_initializer=initializer)(meta_input)
    F = keras.layers.Dense(4, activation='relu', kernel_initializer=initializer)(F)

    final = keras.layers.concatenate([H, V, F])
    final = keras.layers.Dense(64, activation='relu', kernel_initializer=initializer, kernel_regularizer='l2')(final)
    final = keras.layers.Dense(32, activation='relu', kernel_initializer=initializer, kernel_regularizer='l2')(final)
    final = keras.layers.Dense(4, activation='softmax', kernel_initializer=initializer)(final)
    
    model = keras.models.Model(inputs=[H_input, V_input, meta_input], outputs=[final])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
    
def testBearing(X_freq, X_time):
    predictions = model.predict([X_freq[,:641], X_freq[,641:], X_time[b1:b2,:]])
    smooth = movingAvg(predictions)
    Y_pred = np.argmax(smooth, axis=1)
    yint = [0,1,2,3]
    plt.yticks(yint)
    plt.plot(Y_pred, '.')
    return Y_pred

AEmodel = create_model()
AEmodel.load_weights('../../models/AEclassifier')

PCAmodel = create_model()
PCAmodel.load_weights('../../models/PCAclassifier')

rootdir = '../../data/MergedTest_files/'
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        X_freq = ToFrequency(file)
        X_time = ToTime(file)



bplabels = ['Healthy', 'Stage 1', 'Stage 2', 'Stage 3']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,3))
fig.subplots_adjust(hspace=0.2)
ax1.boxplot([xh, xs1, xs2, xs3], labels=bplabels)
ax1.set_ylabel("Test accuracy")
ax2.boxplot([pcaxh, pcaxs1, pcaxs2, pcaxs3], labels=bplabels)
plt.savefig('../../reports/figures/testAcc.png', dpi=100)