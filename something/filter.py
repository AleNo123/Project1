import scipy.signal as sgn
from scipy.io import loadmat
import numpy
import matplotlib.pyplot as plt

data_array = loadmat("data/record_data_8.mat")
data = data_array['recording_data'][::]

low_border = 8
high_border = 13
number_of_electrodes = 31

# plt.plot(data)
# plt.show()

b, a = sgn.butter(3, (low_border, high_border), btype='bandpass', fs=250)
zi = numpy.zeros((max(len(a), len(b)) - 1, number_of_electrodes))

data, zi = sgn.lfilter(b, a, data.T, zi=zi)

plt.plot(data)
plt.show()



