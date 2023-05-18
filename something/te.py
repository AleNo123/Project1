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
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import MaxPooling1D, BatchNormalization
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Conv1D

stream_info = pylsl.resolve_stream()
inlet = StreamInlet(stream_info[0], max_chunklen=750)

with open('data.bin', mode='rb') as file:
    chunk_arr = pickle.load(file)
with open('data2.bin', mode='rb') as f:
    chunk_res = pickle.load(f)

chunk_res = keras.utils.to_categorical(chunk_res, 3)
chunk_arr = np.expand_dims(chunk_arr, axis=0)

chunk_arr = chunk_arr[0, :, :, :]
print(chunk_arr.shape)
b, a = sgn.butter(3, (8, 13), btype='bandpass', fs=250)
zi = numpy.zeros((max(len(a), len(b)) - 1, 30))

plt.plot(chunk_arr[53])
plt.show()
for i in range(chunk_arr.shape[0]):
    chunk_arr[i] = sgn.lfilter(b, a, chunk_arr[i], axis=0)
plt.plot(chunk_arr[53])
plt.show()
# plt.plot(chunk_arr[0][0])
# plt.title('before')
# plt.show()
# chunk_arr = chunk_arr.reshape(300,30,650,1)


# zi = numpy.zeros((1,300,650, max(len(a), len(b)) - 1))
# chunk_arr, z = sgn.lfilter(b, a, chunk_arr, zi=zi)


# plt.plot(chunk_arr[0][0])
# plt.title('after')
# plt.show()
# print(chunk_arr.shape)
chunk_arr = chunk_arr.reshape(300, 30, 650)
# plt.plot(chunk_arr[0][0])
# plt.title('after')
# plt.show()
print(chunk_arr.shape)

model = EEGNet(nb_classes = 3, Chans = 30, Samples = 650,
               dropoutRate = 0.5,  kernLength = 32, F1 = 8, D = 2, F2 = 16,
               dropoutType = 'Dropout')
print(model.summary())


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(len(chunk_arr))
print(len(chunk_res))
checkpointer = ModelCheckpoint(filepath='data/checkpoint.h5', verbose=1,
                               save_best_only=True)
class_weights = {0: 1, 1: 1, 2: 1}
history = model.fit(chunk_arr, chunk_res, epochs=300, validation_split=0.2, class_weight=class_weights,
                    callbacks=[checkpointer], batch_size = 16)
model.load_weights('data/checkpoint.h5')
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy', 'val_accuracy'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss'], loc='upper left')
plt.show()
while True:
    chunk, t_ = inlet.pull_chunk(timeout=3)  # Chunk [750x31(?)]
    chunk = chunk[:650]
    x = np.expand_dims(chunk, axis=0)
    b, a = sgn.butter(3, (8, 13), btype='bandpass', fs=250)
    zi = numpy.zeros((max(len(a), len(b)) - 1, 30))
    x = sgn.lfilter(b, a, x, axis=0)
    # print(x.shape)
    x = x.reshape(1, 30, 650)
    # print(x.shape)
    res = model.predict(x)
    if np.where(res[0] == max(res[0]))[0] == 0:
        print("Left")
    if np.where(res[0] == max(res[0]))[0] == 1:
        print("Right")
    if np.where(res[0] == max(res[0]))[0] == 2:
        print("Legs")
    print(res)
    print(max(res[0]))