import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from dataset import generate_data  # dataset generator

# ============ Tiny FNO ============
class FNO2D(tf.keras.Model):
    def __init__(self, modes=12, width=32):
        super().__init__()
        self.width = width
        self.modes = modes

        self.input_proj = tf.keras.layers.Conv2D(width, 1)
        self.conv1 = tf.keras.layers.Conv2D(width, 1)
        self.conv2 = tf.keras.layers.Conv2D(width, 1)
        self.output_proj = tf.keras.layers.Conv2D(1, 1)

        # Add learning layers for Fourier domain
        self.fft_dense1 = tf.keras.layers.Dense(width)
        self.fft_dense2 = tf.keras.layers.Dense(width)

    def fft_layer(self, x, dense_layer):
        x_ft = tf.signal.fft2d(tf.cast(x, tf.complex64))  # (B, H, W, C)
        # Reshape for Dense: merge spatial dims, keep channels
        shape = tf.shape(x_ft)
        B, H, W, C = shape[0], shape[1], shape[2], shape[3]
        x_ft_flat = tf.reshape(x_ft, [B, H * W, C])
        # Apply Dense layer (real and imag separately)
        x_ft_real = dense_layer(tf.math.real(x_ft_flat))
        x_ft_imag = dense_layer(tf.math.imag(x_ft_flat))
        x_ft_out = tf.complex(x_ft_real, x_ft_imag)
        x_ft_out = tf.reshape(x_ft_out, [B, H, W, self.width])
        x_ifft = tf.signal.ifft2d(x_ft_out)
        x_ifft = tf.math.real(x_ifft)
        return x_ifft

    def call(self, x):
        x = self.input_proj(x)
        x1 = self.fft_layer(x, self.fft_dense1)
        x = tf.nn.gelu(self.conv1(x + x1))
        x2 = self.fft_layer(x, self.fft_dense2)
        x = tf.nn.gelu(self.conv2(x + x2))
        return self.output_proj(x)

# ============ Data ============
f, u, e = generate_data(1000, 64, return_epsilon=True)
f_with_e = np.concatenate([f, e], axis=-1)
train_f, train_u = f_with_e[:800], u[:800]
test_f, test_u = f_with_e[800:], u[800:]

# ============ Model ============
model = FNO2D()
model.compile(optimizer='adam', loss='mse')

# ============ Training ============
batch_size = 20
epochs = 100
train_losses = []

for epoch in range(epochs):
    idx = np.random.permutation(len(train_f))
    train_f_shuffled = train_f[idx]
    train_u_shuffled = train_u[idx]

    epoch_loss = 0
    for i in range(0, len(train_f), batch_size):
        x_batch = train_f_shuffled[i:i + batch_size]
        y_batch = train_u_shuffled[i:i + batch_size]
        loss = model.train_on_batch(x_batch, y_batch)
        epoch_loss += loss

    avg_loss = epoch_loss / (len(train_f) // batch_size)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

# ============ Loss Curve ============
plt.figure()
plt.plot(range(1, epochs + 1), train_losses, marker='o')
plt.title("Epoch vs. Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.savefig("new_channel/plots/loss_curve.png")
print("Saved loss curve at: new_channel/plots/loss_curve.png")

# ============ Prediction ============
pred = model.predict(test_f[:1])[0, ..., 0]
true = test_u[0, ..., 0]

# ============ Plot Prediction vs True ============
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Predicted u")
plt.imshow(pred, cmap='viridis')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("True u")
plt.imshow(true, cmap='viridis')
plt.colorbar()
plt.tight_layout()
plt.savefig("new_channel/plots/pred_vs_true.png")
print("Saved plot at: new_channel/plots/pred_vs_true.png")

# ============ Error Metrics ============
error = np.abs(pred - true)
l1_loss = np.mean(error)
l2_loss = np.sqrt(np.mean((pred - true) ** 2))
linf_loss = np.max(error)
print("Max Error (L-infinity):", linf_loss)
print("Mean Absolute Error (L1):", l1_loss)
print("Root Mean Squared Error (L2):", l2_loss)

# ============ Error Plot ============
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.title("L1 Error Map")
plt.imshow(error, cmap='Reds')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.title("L2 Error Map")
plt.imshow((pred - true) ** 2, cmap='Blues')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.title("L-infinity Error Map")
linf_map = np.zeros_like(error)
linf_map[np.unravel_index(np.argmax(error), error.shape)] = linf_loss
plt.imshow(linf_map, cmap='Greens')
plt.colorbar()

plt.tight_layout()
plt.savefig("new_channel/plots/error_plot.png")
print("Saved error plot at: new_channel/plots/error_plot.png")
