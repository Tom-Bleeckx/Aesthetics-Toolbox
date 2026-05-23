import numpy as np
from scipy.signal import correlate
import torch
import torch.nn.functional as F

np.random.seed(42)
img = np.random.rand(512, 512, 3).astype(np.float32)
kernel = np.random.rand(11, 11, 3, 96).astype(np.float32)
bias = np.random.rand(96).astype(np.float32)

# SciPy implementation
out_height = int(np.ceil(float(512 - 11 + 1) / float(4)))
out_width = int(np.ceil(float(512 - 11 + 1) / float(4)))
output_data = np.zeros((out_height, out_width, 96))

for j in range(96):
    for i in range(3):
        output_data[:, :, j] += correlate(
            img[:, :, i],
            kernel[:, :, i, j],
            mode='valid'
        )[::4, ::4]
    output_data[:, :, j] += bias[j]

output_data[output_data < 0] = 0
output_data = np.swapaxes(output_data, 2, 0)
output_data = np.swapaxes(output_data, 1, 2)

# PyTorch implementation
input_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
kernel_tensor = torch.from_numpy(kernel).permute(3, 2, 0, 1)
bias_tensor = torch.from_numpy(bias)

out = F.conv2d(input_tensor, kernel_tensor, bias=bias_tensor, stride=4)
out = F.relu(out)
out_pt = out[0].numpy()

diff = np.abs(output_data - out_pt).max()
print(f"Max diff: {diff}")
