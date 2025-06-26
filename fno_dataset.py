import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def generate_data(n_samples=1000, size=64, m_max=5, n_max=5):
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y)

    f_list, u_list = [], []

    for _ in range(n_samples):
        m = np.random.randint(1, m_max + 1)
        n = np.random.randint(1, n_max + 1)

        f = np.sin(m * np.pi * X) * np.sin(n * np.pi * Y)
        u = f / (np.pi**2 * (m**2 + n**2))  # Exact Poisson solution

        f_list.append(f[..., None])  # shape: [H, W, 1]
        u_list.append(u[..., None])

    return np.array(f_list).astype(np.float32), np.array(u_list).astype(np.float32)
