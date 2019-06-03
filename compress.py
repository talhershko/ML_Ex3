import numpy as np

train_x = np.loadtxt("train_x")
train_y = np.loadtxt("train_y")
test_x = np.loadtxt("test_x")

np.save('train_x.npy', train_x)
np.save('train_y.npy', train_y)
np.save('test_x.npy', test_x)



