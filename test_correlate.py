import numpy as np
from scipy.signal import correlate

np.random.seed(0)
input_img = np.random.rand(10, 10, 3)
kernel = np.random.rand(3, 3, 3)

# original
out1 = np.zeros((8, 8))
for i in range(3):
    out1 += correlate(input_img[:,:,i], kernel[:,:,i], mode='valid')

# 3d
out2 = correlate(input_img, kernel, mode='valid')[:,:,0]

print("Max diff:", np.max(np.abs(out1 - out2)))
