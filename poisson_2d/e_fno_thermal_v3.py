import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from scipy.sparse import diags, kron, identity
from scipy.sparse.linalg import spsolve

# ========== Output Directory ==========
experiment_name = "EXP_thermal_diffusion_fixed_v3_try3"
output_path = os.path.join("Outputs", experiment_name)
os.makedirs(output_path, exist_ok=True)

# ========== PDE Solver ==========
def solve_thermal_diffusion(k, Q, h):
    n = k.shape[0]
    k = k.reshape((n, n))
    Q = Q.reshape((n, n))

    k_xp = np.pad((k[1:, :] + k[:-1, :]) / 2, ((0,1),(0,0)), mode='edge')
    k_xm = np.pad((k[1:, :] + k[:-1, :]) / 2, ((1,0),(0,0)), mode='edge')
    k_yp = np.pad((k[:,1:] + k[:,:-1]) / 2, ((0,0),(0,1)), mode='edge')
    k_ym = np.pad((k[:,1:] + k[:,:-1]) / 2, ((0,0),(1,0)), mode='edge')

    main_diag = (k_xp + k_xm + k_yp + k_ym).flatten()
    off_xp = -k_xp[:-1,:].flatten()
    off_xm = -k_xm[1:,:].flatten()
    off_yp = -k_yp[:,:-1].flatten()
    off_ym = -k_ym[:,1:].flatten()

    Ix = identity(n)
    Tx = diags([off_xm, main_diag, off_xp], [-1, 0, 1], shape=(n, n))
    Ty = diags([off_ym, off_yp], [-1, 1], shape=(n, n))
    A = kron(Ix, Tx) + kron(Ty, Ix)

    rhs = Q.flatten()
    T_flat = spsolve(A / (h**2), rhs)
    return T_flat.reshape((n, n))

# ========== Data Generator ==========
def generate_thermal_diffusion_data(n_samples=1200, size=64):
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y, indexing='ij')
    h = 1.0 / (size - 1)

    K, Q_list, T_list = [], [], []
    for _ in range(n_samples):
        k = 0.5 + 0.5 * np.sin(2 * np.pi * X * np.random.uniform(1, 3)) * np.cos(2 * np.pi * Y * np.random.uniform(1, 3))
        a, b = np.random.uniform(1, 3), np.random.uniform(1, 3)
        Q = np.sin(a * np.pi * X) * np.sin(b * np.pi * Y)
        T = solve_thermal_diffusion(k, Q, h)

        # Normalize
        k = (k - k.mean()) / k.std()
        Q = (Q - Q.mean()) / Q.std()
        T = (T - T.mean()) / T.std()

        K.append(k)
        Q_list.append(Q)
        T_list.append(T)

    K = np.expand_dims(np.array(K), axis=-1)
    Q_list = np.expand_dims(np.array(Q_list), axis=-1)
    T_list = np.expand_dims(np.array(T_list), axis=-1)
    input_fields = np.concatenate([K, Q_list], axis=-1)
    return input_fields.astype(np.float32), T_list.astype(np.float32)

# ========== Fourier Layer ==========
class FourierLayer(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, modes=12):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        initializer = tf.keras.initializers.RandomNormal(stddev=1e-2)
        self.weights_real = self.add_weight(name="w_real", shape=[in_channels, out_channels, modes, modes], initializer=initializer, trainable=True)
        self.weights_imag = self.add_weight(name="w_imag", shape=[in_channels, out_channels, modes, modes], initializer=initializer, trainable=True)

    def call(self, x):
        x = tf.transpose(x, [0, 3, 1, 2])
        x_ft = tf.signal.fft2d(tf.cast(x, tf.complex64))
        x_ft = x_ft[:, :, :self.modes, :self.modes]
        w = tf.complex(self.weights_real, self.weights_imag)
        out_ft = tf.einsum("bcmn,comn->bomn", x_ft, w)

        B = tf.shape(x)[0]
        H, W = tf.shape(x)[2], tf.shape(x)[3]
        out_full = tf.concat(
            [tf.concat([out_ft, tf.zeros([B, self.out_channels, self.modes, W - self.modes], tf.complex64)], axis=-1),
             tf.zeros([B, self.out_channels, H - self.modes, W], tf.complex64)], axis=-2)

        out = tf.signal.ifft2d(out_full)
        out = tf.transpose(tf.math.real(out), [0, 2, 3, 1])
        return out

# ========== FNO Block ==========
class FNOBlock(tf.keras.layers.Layer):
    def __init__(self, width, modes=12):
        super().__init__()
        self.fourier = FourierLayer(width, width, modes)
        self.conv = tf.keras.layers.Conv2D(width, 1)

    def call(self, x):
        return tf.nn.gelu(self.fourier(x) + self.conv(x))

# ========== FNO Model ==========
class FNO2D(tf.keras.Model):
    def __init__(self, modes=12, width=64):
        super().__init__()
        self.input_proj = tf.keras.layers.Conv2D(width, 1)
        self.blocks = [FNOBlock(width, modes) for _ in range(2)]
        self.output_proj = tf.keras.layers.Conv2D(1, 1)

    def call(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)

# ========== Training ==========
inputs, targets = generate_thermal_diffusion_data(n_samples=1000, size=64)
train_x, train_y = inputs[:800], targets[:800]
test_x, test_y = inputs[800:], targets[800:]

model = FNO2D()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mse')

history = model.fit(train_x, train_y, validation_split=0.1, batch_size=64, epochs=200)

# ========== Loss Plot ==========
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_path, "loss_curve.png"))

# ========== Evaluate ==========
pred = model.predict(test_x[:1])[0, ..., 0]
true = test_y[0, ..., 0]
error = np.abs(pred - true)

rel_error = np.linalg.norm(error) / np.linalg.norm(true)
l2_error = np.mean((pred - true)**2)

with open(os.path.join(output_path, "metrics.txt"), "w") as f:
    f.write(f"Relative Error: {rel_error:.6f}\n")
    f.write(f"L2 Error: {l2_error:.6f}\n")

# ========== Prediction Plots ==========
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Predicted T")
plt.imshow(pred, cmap='inferno')
plt.colorbar()
plt.subplot(1, 3, 2)
plt.title("True T")
plt.imshow(true, cmap='inferno')
plt.colorbar()
plt.subplot(1, 3, 3)
plt.title("|T_pred - T_true|")
plt.imshow(error, cmap='Reds')
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(output_path, "pred_vs_true_error.png"))
