import pickle

import numpy
import pydot
import pylsl
import time
import scipy
import tf as tf
from keras.layers import MaxPooling1D, BatchNormalization
from numpy import float32
from pylsl import StreamInlet
import numpy as np

from tensorflow import keras
import keras
import pydotplus
import graphviz
from keras.utils.vis_utils import plot_model

import scipy.signal as sgn
from scipy.io import loadmat

import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten,Conv1D

# stream_info = pylsl.resolve_stream()
# print(stream_info)
# probes_duration=750
# #
# inlet = StreamInlet(stream_info[0], max_chunklen=probes_duration)
# inlet1 = StreamInlet(stream_info[1], max_chunklen=probes_duration)
# ---------------------------------
# chunk, t_ = inlet.pull_chunk(timeout=3)  # Chunk [750x31(?)]
# ---------------------------------
# chunk_arr = []
# chunk_res = []
# chunk_r = [0]
# with open('data.bin', mode='rb') as file:
#     chunk_arr = pickle.load(file)
# with open('data2.bin',mode='rb') as f:
#     chunk_res = pickle.load(f)
# # ---------------------------------
# for x in range(100):
#     chunk, t_ = inlet.pull_chunk(timeout=3)  # Chunk [750x31(?)]
#     chunk = chunk[:650]
#     chunk_arr.append(chunk)
#     chunk_res.append(chunk_r)
# with open('data.bin', mode='wb') as file:
#     pickle.dump(chunk_arr, file)
# with open('data2.bin', mode='wb') as file:
#     pickle.dump(chunk_res,file)
# ---------------------------------
# chunk_res = np.expand_dims(chunk_res, axis=0)
# ---------------------------------

# model = Sequential([
#     Conv1D(32, 30, activation='sigmoid',  padding='same', input_shape=(650, 30)),
#     BatchNormalization(),
#     MaxPooling1D(pool_size=2),
#     Conv1D(64, 30, activation='tanh', padding='same'),
#     BatchNormalization(),
#     MaxPooling1D(pool_size=2),
#     Conv1D(64, 30, activation='sigmoid'),
#     # BatchNormalization(),
#     MaxPooling1D(pool_size=2),
#     Flatten(),
#     Dense(128, activation='linear'),
#     # BatchNormalization(),
#     Dense(3, activation='softmax'),
# ])
# chunk_res = keras.utils.to_categorical(chunk_res,3)
# chunk_arr = np.expand_dims(chunk_arr, axis=0)
#
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# chunk, t_ = inlet.pull_chunk(timeout=3)
# chunk = chunk[:650]
# x = np.expand_dims(chunk, axis=0)


def neural_network_training():
    with open('data.bin', mode='rb') as file:
        chunk_arr = pickle.load(file)
    with open('data2.bin', mode='rb') as f:
        chunk_res = pickle.load(f)
    # model = Sequential([
    #     Conv1D(32, 30, activation='sigmoid', padding='same', input_shape=(650, 30)),
    #     BatchNormalization(),
    #     MaxPooling1D(pool_size=2),
    #     Conv1D(64, 30, activation='tanh', padding='same'),
    #     BatchNormalization(),
    #     MaxPooling1D(pool_size=2),
    #     Conv1D(64, 30, activation='sigmoid'),
    #     # BatchNormalization(),
    #     MaxPooling1D(pool_size=2),
    #     Flatten(),
    #     Dense(128, activation='linear'),
    #     # BatchNormalization(),
    #     Dense(3, activation='softmax'),
    # ])
    model = Sequential([
        Conv1D(32, 30, activation='linear', padding='same', input_shape=(650, 30)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(64, 30, activation='linear', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(64, 30, activation='linear'),
        # BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='linear'),
        # BatchNormalization(),
        Dense(3, activation='softmax'),
    ])
    chunk_res = keras.utils.to_categorical(chunk_res, 3)
    chunk_arr = np.expand_dims(chunk_arr, axis=0)

    b, a = sgn.butter(3, (8, 13), btype='bandpass', fs=250)

    zi = numpy.zeros((1, 300, 650, max(len(a), len(b)) - 1))

    chunk_arr, zi = sgn.lfilter(b, a, chunk_arr, zi=zi)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(chunk_arr[0], chunk_res, epochs=25, validation_split=0.2)
    return model


# model.fit(chunk_arr[0], chunk_res, epochs=45, validation_split=0.2)
# for i in range(10):
#     chunk, t_ = inlet.pull_chunk(timeout=3)  # Chunk [750x31(?)]
#     chunk = chunk[:650]
#     x = np.expand_dims(chunk, axis=0)
#     res = model.predict(x)
#     if np.where(res[0] == max(res[0]))[0] != chunk_r[0]:
#         print("ERROR")
#     print(res)
#     print(max(res[0]))