import numpy as np

data = np.load('embds.npy')

data = data.mean(axis=(2, 3))

np.save('embds_mean.npy', data)