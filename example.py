import numpy as np
import matplotlib.pyplot as plt
import random

test_x = np.loadtxt("test_x")
l = len(test_x)  # 5000
index = random.randrange(l)
print(index)
print(len(test_x[index]))  # 784 = 28*28
plt.imshow(test_x[index].reshape((28, 28)))
plt.show()
