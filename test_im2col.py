import numpy as np
from scipy.signal import correlate
from numpy.lib.stride_tricks import as_strided
import time

np.random.seed(0)
input_img = np.random.rand(512, 512, 3)
kernel = np.random.rand(11, 11, 3, 96)
bias = np.random.rand(96)

# Method 1: Original
t0 = time.time()
in_height, in_width, in_channels = input_img.shape
k_height, k_width, in_channels, out_channels = kernel.shape
out_height = int(np.ceil(float(in_height - k_height + 1) / float(4)))
out_width = int(np.ceil(float(in_width - k_width + 1) / float(4)))

output_data1 = np.zeros((out_height, out_width, out_channels))
for j in range(out_channels):
    for i in range(in_channels):
        output_data1[:, :, j] += correlate(
            input_img[:, :, i],
            kernel[:, :, i, j],
            mode='valid'
        )[::4, ::4]
for j in range(out_channels):
    output_data1[:, :, j] += bias[j]
t1 = time.time()
print(f"Original: {t1-t0:.4f}s")

# Method 2: Patch extraction
t0 = time.time()
s0, s1, s2 = input_img.strides
patches = as_strided(
    input_img,
    shape=(out_height, out_width, k_height, k_width, in_channels),
    strides=(s0*4, s1*4, s0, s1, s2)
)
# reshape patches to (out_height * out_width, k_height * k_width * in_channels)
patches_flat = patches.reshape(-1, k_height * k_width * in_channels)
# reshape kernel to (k_height * k_width * in_channels, out_channels)
kernel_flat = kernel.reshape(-1, out_channels)
# actually, correlate does correlation, which means flipping the kernel!
# scipy's correlate flips the kernel. So we should flip the kernel.
kernel_flipped = kernel[::-1, ::-1, :, :]
kernel_flat_flipped = kernel_flipped.reshape(-1, out_channels)
output_data2 = np.dot(patches_flat, kernel_flat_flipped).reshape(out_height, out_width, out_channels)
output_data2 += bias

t1 = time.time()
print(f"Patch (flipped): {t1-t0:.4f}s")
print("Max diff:", np.max(np.abs(output_data1 - output_data2)))
