import numpy as np


def standardize_data(x):
    x = x.astype(np.float32)
    mean = np.mean(x, axis=(0, 1, 2, 3))
    std = np.std(x, axis=(0, 1, 2, 3))
    x = (x - mean) / (std + 1e-7)

    return x
