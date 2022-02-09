import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib as mpl

#read in train data
data = pd.read_csv('../../data/Xtrain.csv', header=None)
X = data.values
data = pd.read_csv('../../data/Xtrain_time.csv')
Xtime = data.values
data = pd.read_csv('../../data/labels/train_AElabels.csv')
Y = keras.utils.to_categorical(data.values, num_classes=4)
data = pd.read_csv('../../data/labels/train_PCAlabels.csv')
Ypca = keras.utils.to_categorical(data.values, num_classes=4)


#classifier architecture

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
AEhistory = model.fit([X[:,:641], X[:,641:], Xtime], Y, batch_size=512, epochs=15, shuffle=True)

model.save_weights('../../models/AEclassifier')

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
PCAhistory = model.fit([X[:,:641], X[:,641:], Xtime], Ypca, batch_size=512, epochs=100, shuffle=True,)

model.save_weights('../../models/PCAclassifier')

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

plt.plot(AEhistory.history['acc'], label='AElabels')
plt.plot(PCAhistory.history['acc'], label='PCAlabels')
plt.xlabel("Epoch")
plt.ylabel("Trainig accuracy")
plt.legend(loc='best')
plt.savefig('../../reports/figures/trainingAcc.png', dpi=100)


