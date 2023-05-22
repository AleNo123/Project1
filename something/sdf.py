from scipy.io import loadmat
import numpy
import pylsl
import scipy.signal as sgn
import time
import pickle
from pylsl import StreamInlet
import numpy as np

import keras
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from EEGModels import EEGNet, ShallowConvNet, DeepConvNet

data_array = loadmat('data/record_data_10.mat')
data = data_array['recording_data'][:,:]
signs = data_array['signs'][0,:]
print(data.shape)
print(signs.shape)
print(signs)
b, a = sgn.butter(3, (13, 31), btype='bandpass', fs=250)
zi = numpy.zeros((max(len(a), len(b)) - 1, 30))
data = sgn.lfilter(b, a, data, axis=0)
print(data.shape)
num=0
check=1
chunk_arr=[]
chunk_res=[]
ok = 0
ok2 = 0
for i in signs:
    if i!= check:
        # print(data[:,:num])
        # print(data[:, :num].shape)
        if i == 2 and ok!=0:
            num = 0
            check = i
        elif i == 0 and ok2!=0:
            num = 0
            check = i
        else:
            chunk_arr.append(data[:,:650])
            chunk_res.append(i)
            # print(str(num)+', '+str(i))
            num=0
            if i == 2:
                ok = 1
            if i == 0:
                ok2 = 1
            check =i
    else:
        num+=1
chunk_arr = np.array(chunk_arr)
chunk_res = keras.utils.to_categorical(chunk_res, 3)
b, a = sgn.butter(3, (8, 13), btype='bandpass', fs=250)
print(chunk_arr.shape)
print(chunk_res.shape)
chunk_arr = chunk_arr.reshape(chunk_arr.shape[0], 650, 14)
# plt.plot(chunk_arr[3])
# plt.show()
for i in range(chunk_arr.shape[0]):
    chunk_arr[i] = sgn.lfilter(b, a, chunk_arr[i],axis=0)
# plt.plot(chunk_arr[3])
# plt.show()
chunk_arr = chunk_arr.reshape(chunk_arr.shape[0], 14, 650)
# plt.plot(chunk_arr[3])
# plt.show()
model = EEGNet(nb_classes=3, Chans=14, Samples=650,
               dropoutRate=0.5, kernLength=32, F1=16, D=4, F2=64,
               dropoutType='Dropout')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='data/checkpoint_sdf.h5', verbose=1,
                               save_best_only=True)
class_weights = {0: 1, 1: 1, 2: 1}
history = model.fit(chunk_arr, chunk_res, epochs=200, validation_split=0.8, class_weight=class_weights,
                    callbacks=[checkpointer], batch_size = 1)
model.load_weights('data/checkpoint_sdf.h5')
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

data_array1 = loadmat('data/record_data_11.mat')
data1 = data_array1['recording_data'][:,:]
signs1 = data_array1['signs'][0,:]
chunk_arr1=[]
chunk_res1=[]
num=0
check=0
for i in signs1:
    if i!= check:
        if (num<650):
            check = i
            num = 0
        else:
            chunk_arr1.append(data1[:,:650])
            chunk_res1.append(i)
            # print(str(num) + ', ' + str(i))
            num=0
            check =i
    else:
        num+=1
chunk_arr1 = np.array(chunk_arr1)
print(chunk_arr1.shape)
chunk_arr1 = chunk_arr1.reshape(chunk_arr1.shape[0], 650, 14)
for i in range(chunk_arr1.shape[0]):
    chunk_arr1[i] = sgn.lfilter(b, a, chunk_arr1[i])
chunk_arr1 = chunk_arr1.reshape(chunk_arr1.shape[0], 14, 650)
probs  = model.predict(chunk_arr1)
num =0
# print(probs)
for i in probs:
    print(str(np.where(probs[num] == max(probs[num]))[0])+',  '+str(chunk_res1[num])+',  '+str(probs[num]))
    num+=1
