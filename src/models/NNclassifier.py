import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib as mpl
from random import randint

# classifier architecture


def create_model():

    H_input = keras.Input(shape=(641,), name="horizontal")
    V_input = keras.Input(shape=(641,), name="vertical")
    meta_input = keras.Input(shape=(26,), name="meta_input")
    initializer = tf.keras.initializers.RandomNormal(seed=42)

    H = keras.layers.Dense(
        256, activation="relu", kernel_initializer=initializer, kernel_regularizer="l2"
    )(H_input)
    H = keras.layers.Dense(
        128, activation="relu", kernel_initializer=initializer, kernel_regularizer="l2"
    )(H)
    H = keras.layers.Dense(
        64, activation="relu", kernel_initializer=initializer, kernel_regularizer="l2"
    )(H)
    H = keras.layers.Dense(
        32, activation="relu", kernel_initializer=initializer, kernel_regularizer="l2"
    )(H)
    H = keras.layers.Dense(4, activation="relu", kernel_initializer=initializer)(H)

    V = keras.layers.Dense(
        256, activation="relu", kernel_initializer=initializer, kernel_regularizer="l2"
    )(V_input)
    V = keras.layers.Dense(
        128, activation="relu", kernel_initializer=initializer, kernel_regularizer="l2"
    )(V)
    V = keras.layers.Dense(
        64, activation="relu", kernel_initializer=initializer, kernel_regularizer="l2"
    )(V)
    V = keras.layers.Dense(
        32, activation="relu", kernel_initializer=initializer, kernel_regularizer="l2"
    )(V)
    V = keras.layers.Dense(4, activation="relu", kernel_initializer=initializer)(V)

    F = keras.layers.Dense(16, activation="relu", kernel_initializer=initializer)(
        meta_input
    )
    F = keras.layers.Dense(4, activation="relu", kernel_initializer=initializer)(F)

    final = keras.layers.concatenate([H, V, F])
    final = keras.layers.Dense(
        64, activation="relu", kernel_initializer=initializer, kernel_regularizer="l2"
    )(final)
    final = keras.layers.Dense(
        32, activation="relu", kernel_initializer=initializer, kernel_regularizer="l2"
    )(final)
    final = keras.layers.Dense(4, activation="softmax", kernel_initializer=initializer)(
        final
    )
    model = keras.models.Model(inputs=[H_input, V_input, meta_input], outputs=[final])

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model
