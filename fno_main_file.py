import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from fno_dataset import generate_data  # dataset generator

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

    def fft_layer(self, x):
        x_ft = tf.signal.fft2d(tf.cast(x, tf.complex64))
        x_ft = tf.math.real(tf.signal.ifft2d(x_ft))
        return x_ft

    def call(self, x):
        x = self.input_proj(x)
        x1 = self.fft_layer(x)
        x = tf.nn.gelu(self.conv1(x + x1))
        x2 = self.fft_layer(x)
        x = tf.nn.gelu(self.conv2(x + x2))
        return self.output_proj(x)

# ============ Data ============
f, u = generate_data(1000, 64)
train_f, train_u = f[:800], u[:800]
test_f, test_u = f[800:], u[800:]

# ============ Model ============
model = FNO2D()
model.compile(optimizer='adam', loss='mse')

# ============ Training ============
batch_size = 20
epochs = 50
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
plt.savefig("Plots/loss_curve.png")
print("Saved loss curve at: Plots/loss_curve.png")

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
plt.savefig("Plots/pred_vs_true.png")
print("Saved plot at: Plots/pred_vs_true.png")

# ============ Error Metrics ============
error = np.abs(pred - true)
print("Max Error:", np.max(error))
print("Mean Absolute Error:", np.mean(error))

# ============ Error Plot ============
plt.figure()
plt.imshow(error, cmap='Reds')
plt.title("Absolute Error |u_pred - u_true|")
plt.colorbar()
plt.tight_layout()
plt.savefig("Plots/error_plot.png")
print("Saved error plot at: Plots/error_plot.png")
