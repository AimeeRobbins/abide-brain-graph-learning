import numpy as np
import matplotlib.pyplot as plt

A = np.load("ec_graphs/50555_ec_graph.npy")

print("Shape:", A.shape)
print("Number of non-zero entries:", np.count_nonzero(A))
print("Expected edges:", 200 * 5)

print("Min:", A.min())
print("Max:", A.max())
print("Number of zeros:", np.sum(A == 0))

data = np.loadtxt("abide_data/outputs/cpac/filt_global/rois_cc200/Yale_0050626_rois_cc200.1D")
print(data)
print("Number of zeros:", np.sum(data == 0))
roi = 0

zero_mask = (data == 0)

print("Total zeros:", zero_mask.sum())

zero_rois = np.where(np.all(data == 0, axis=0))[0]

print("Number of entirely-zero ROIs:", len(zero_rois))
print("ROIs:", zero_rois)

plt.figure(figsize=(12, 6))
plt.imshow(zero_mask, aspect="auto")
plt.xlabel("ROI")
plt.ylabel("Timepoint")
plt.title("Zero values in raw ROI time series")
plt.colorbar(label="Zero (1 = yes, 0 = no)")
plt.show()
