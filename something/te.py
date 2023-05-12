import numpy
import pylsl
import scipy.signal as sgn
import time
import pickle
from pylsl import StreamInlet
import numpy as np

import keras
from matplotlib import pyplot as plt

from EEGModels import EEGNet, ShallowConvNet, DeepConvNet

from tensorflow.keras.datasets import mnist
from tensorflow import keras
from keras.models import Sequential
from keras.layers import MaxPooling1D, BatchNormalization
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Conv1D

# model = Sequential([
#     Conv1D(32, 30, activation='linear', padding='same', input_shape=(650, 30)),
#     BatchNormalization(),
#     MaxPooling1D(pool_size=2),
#     Conv1D(64, 30, activation='linear', padding='same'),
#     BatchNormalization(),
#     MaxPooling1D(pool_size=2),
#     Conv1D(64, 30, activation='linear'),
#     # BatchNormalization(),
#     MaxPooling1D(pool_size=2),
#     Flatten(),
#     Dense(128, activation='linear'),
#     # BatchNormalization(),
#     Dense(3, activation='softmax'),
# ])
with open('data.bin', mode='rb') as file:
    chunk_arr = pickle.load(file)
with open('data2.bin', mode='rb') as f:
    chunk_res = pickle.load(f)
chunk_res = keras.utils.to_categorical(chunk_res, 3)
chunk_arr = np.expand_dims(chunk_arr, axis=0)

b, a = sgn.butter(3, (8, 13), btype='bandpass', fs=250)

print(chunk_arr.shape)

zi = numpy.zeros((300,650, 30, max(len(a), len(b)) - 1))


# z, _ = sgn.lfilter(b, a, chunk_arr, zi=zi)
# z2, _ = sgn.lfilter(b, a, z, zi=zi)
# y = sgn.filtfilt(b,a,chunk_arr)

chunk_arr = chunk_arr.reshape(300,650,30,1)
chunk_arr, z = sgn.lfilter(b, a, chunk_arr, zi=zi)
print(chunk_arr.shape)
# X_train      = X_train.reshape(X_train.shape[0], chans, samples, kernels)
model = EEGNet(nb_classes = 3, Chans = 650, Samples = 30,
               dropoutRate = 0.5, kernLength = 8)
print(model.summary())
# print(chunk_arr[0].shape)
# print(chunk_arr.shape)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(len(chunk_arr))
print(len(chunk_res))
history = model.fit(chunk_arr, chunk_res, epochs=5, validation_split=0.2)

stream_info = pylsl.resolve_stream()
inlet = StreamInlet(stream_info[0], max_chunklen=750)

# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['accuracy', 'val_accuracy'], loc='upper left')
# plt.show()
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['loss', 'val_loss'], loc='upper left')
# plt.show()
while True:
    chunk, t_ = inlet.pull_chunk(timeout=3)  # Chunk [750x31(?)]
    chunk = chunk[:650]
    x = np.expand_dims(chunk, axis=0)
    print(x.shape)
    x = x.reshape(1,650,30, 1)
    print(x.shape)
    res = model.predict(x)
    if np.where(res[0] == max(res[0]))[0] == 0:
        print("Left")
    if np.where(res[0] == max(res[0]))[0] == 1:
        print("Right")
    if np.where(res[0] == max(res[0]))[0] == 2:
        print("Legs")
    print(res)
    print(max(res[0]))