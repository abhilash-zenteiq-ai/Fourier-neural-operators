import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from dolfin import *
import numpy as np
import os
import matplotlib.pyplot as plt

np.random.seed(42)

# Output directory
output_dir = "Outputs/e_main_output"
os.makedirs(output_dir, exist_ok=True)

def random_e_expression():
    """Random spatially varying coefficient e(x,y)."""
    a = np.random.uniform(1.0, 3.0)
    b = np.random.uniform(1.0, 3.0)
    return Expression("0.5 + 0.5*sin(a*pi*x[0])*cos(b*pi*x[1])", degree=3, a=a, b=b)

def f_expression():
    return Expression("sin(pi*x[0])*sin(pi*x[1])", degree=3)

def solve_pde(N=64, n_samples=100):
    E_list, F_list, U_list = [], [], []

    mesh = UnitSquareMesh(N, N)
    V = FunctionSpace(mesh, "P", 1)

    for i in range(n_samples):
        e_expr = random_e_expression()
        f_expr = f_expression()

        e = interpolate(e_expr, V)
        f = interpolate(f_expr, V)

        u = TrialFunction(V)
        v = TestFunction(V)

        a = dot(e * grad(u), grad(v)) * dx
        L = f * v * dx

        bc = DirichletBC(V, Constant(0.0), "on_boundary")

        u_sol = Function(V)
        solve(a == L, u_sol, bc)

        coords = V.tabulate_dof_coordinates().reshape((N + 1, N + 1, 2))
        u_vals = u_sol.compute_vertex_values(mesh).reshape(N + 1, N + 1).T
        e_vals = e.compute_vertex_values(mesh).reshape(N + 1, N + 1).T
        f_vals = f.compute_vertex_values(mesh).reshape(N + 1, N + 1).T

        # Resize to 64x64
        u_vals = u_vals[:64, :64]
        e_vals = e_vals[:64, :64]
        f_vals = f_vals[:64, :64]

        E_list.append(e_vals)
        F_list.append(f_vals)
        U_list.append(u_vals)

        print(f"[INFO] Sample {i + 1}/{n_samples} complete")

    E = np.expand_dims(np.array(E_list), axis=-1)
    F = np.expand_dims(np.array(F_list), axis=-1)
    U = np.expand_dims(np.array(U_list), axis=-1)

    np.save(os.path.join(output_dir, "e_field.npy"), E)
    np.save(os.path.join(output_dir, "f_field.npy"), F)
    np.save(os.path.join(output_dir, "u_field.npy"), U)

    # Example plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("e(x,y)")
    plt.imshow(E[0, ..., 0], cmap='plasma')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("f(x,y)")
    plt.imshow(F[0, ..., 0], cmap='viridis')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("u(x,y)")
    plt.imshow(U[0, ..., 0], cmap='inferno')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "example_plot.png"))
    print("[INFO] Sample plot saved to:", os.path.join(output_dir, "example_plot.png"))

if __name__ == "__main__":
    solve_pde(N=64, n_samples=500)

# Load data
data_dir = "fem_data"
E = np.load(os.path.join(data_dir, "e_field.npy"))
F = np.load(os.path.join(data_dir, "f_field.npy"))
U = np.load(os.path.join(data_dir, "u_field.npy"))

inputs = np.concatenate([E, F], axis=-1)
targets = U

# Train-test split
train_x, test_x = inputs[:400], inputs[400:]
train_y, test_y = targets[:400], targets[400:]

# FNO Model
class FNO2D(tf.keras.Model):
    def __init__(self, width=32):
        super().__init__()
        self.width = width
        self.input_proj = tf.keras.layers.Conv2D(width, 1)
        self.conv1 = tf.keras.layers.Conv2D(width, 1)
        self.conv2 = tf.keras.layers.Conv2D(width, 1)
        self.output_proj = tf.keras.layers.Conv2D(1, 1)

    def fft_layer(self, x):
        x_ft = tf.signal.fft2d(tf.cast(x, tf.complex64))
        x_ft = tf.math.real(tf.signal.ifft2d(x_ft))
        return x_ft

    def call(self, x):
        x = self.input_proj(x)
        x = tf.nn.gelu(self.conv1(x + self.fft_layer(x)))
        x = tf.nn.gelu(self.conv2(x + self.fft_layer(x)))
        return self.output_proj(x)

# Train
model = FNO2D()
model.compile(optimizer='adam', loss='mse')

history = model.fit(train_x, train_y, validation_split=0.1, batch_size=20, epochs=50)

# Evaluate
pred = model.predict(test_x[:1])[0, ..., 0]
true = test_y[0, ..., 0]

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Predicted u")
plt.imshow(pred, cmap='viridis')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.title("True u")
plt.imshow(true, cmap='viridis')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.title("Absolute Error")
plt.imshow(np.abs(pred - true), cmap='Reds')
plt.colorbar()

plt.tight_layout()
plt.savefig("pred_vs_true.png")

# Metrics
l2 = np.linalg.norm(pred - true) / np.linalg.norm(true)
mae = np.mean(np.abs(pred - true))

with open("metrics.txt", "w") as f:
    f.write(f"L2 Relative Error: {l2:.6f}\n")
    f.write(f"MAE: {mae:.6f}\n")

print(f"[INFO] L2 Relative Error: {l2:.6f}, MAE: {mae:.6f}")
