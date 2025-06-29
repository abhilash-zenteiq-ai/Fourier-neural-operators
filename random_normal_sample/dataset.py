# importing required libraries
import numpy as np

# generating required data
def generate_data(n_samples=1000, size=64):

    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)

    X, Y = np.meshgrid(x, y)
    h = x[1] - x[0]

    a_list = []
    f_list = []
    u_list = []

    for _ in range(n_samples):

        u = np.sin(np.pi * X) * np.sin(np.pi * Y)

        g = np.random.randn(size, size) # sampling from normal distibution
        a = np.exp(g).astype(np.float32) # gets a positive diffusion coefficient
    
        u_x = np.gradient(u, x, axis=1)
        u_y = np.gradient(u, y, axis=0)

        ax = a * u_x
        ay = a * u_y

        div_x = np.gradient(ax, x, axis=1)
        div_y = np.gradient(ay, y, axis=0)

        f = -(div_x + div_y)

        a_list.append(a[..., None])
        f_list.append(f[..., None])
        u_list.append(u[..., None])

    return (
        np.array(a_list, dtype= np.float32),
        np.array(f_list, dtype= np.float32),
        np.array(u_list, dtype= np.float32)
    )