import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from dataset import generate_data

#--------------model------------------

class FNOBlock(tf.keras.layers.Layer):
    def __init__(self, width):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(width, 1)
    def call(self, x):
        x_fft = tf.signal.fft2d(tf.cast(x, tf.complex64))
        x_fft = tf.math.real(tf.signal.ifft2d(x_fft))
        return tf.nn.gelu(x + self.conv(x_fft))
    
class model(tf.keras.Model):
    def __init__(self, width=32):
        super().__init__()
        self.input_proj = tf.keras.layers.Conv2D(width, 1)
        self.fno1 = FNOBlock(width)
        self.fno2 = FNOBlock(width)
        self.output_proj = tf.keras.layers.Conv2D(1, 1)

    def call(self, x):
        x = self.input_proj(x)
        x = self.fno1(x)
        x = self.fno2(x)
        return self.output_proj(x)
    
#-----------------data-----------------------
a, f, u = generate_data(n_samples=1000, size=64)
X = np.concatenate([a, f], axis=-1)  
Y = u 

train_x, test_x = X[:800], X[800:]
train_y, test_y = Y[:800], Y[800:]

#----------------initialize model------------

model = model()
model.compile(optimizer='adam', loss='mse')

# ============ Training ============
batch_size = 20
epochs = 50
train_losses = []

for epoch in range(epochs):
    idx = np.random.permutation(len(train_x))
    train_f_shuffled = train_x[idx]
    train_u_shuffled = train_y[idx]

    epoch_loss = 0
    for i in range(0, len(train_x), batch_size):
        x_batch = train_f_shuffled[i:i + batch_size]
        y_batch = train_u_shuffled[i:i + batch_size]
        loss = model.train_on_batch(x_batch, y_batch)
        epoch_loss += loss

    avg_loss = epoch_loss / (len(train_x) // batch_size)
    train_losses.append(avg_loss)
    if epoch == epochs - 1 or epoch%10 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss}")

# ============ Loss Curve ============
plt.figure()
plt.plot(range(1, epochs + 1), train_losses, marker='o')
plt.title("Epoch vs. Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.savefig("random_normal_sample/results/loss_curve.png")
print("Saved loss curve at: random_normal_sample/results/loss_curve.png")

# ============ Prediction ============
pred = model.predict(test_x[:1])[0, ..., 0]
true = test_x[0, ..., 0]

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
plt.savefig("random_normal_sample/results/pred_vs_true.png")
print("Saved plot at: random_normal_sample/results/pred_vs_true.png")

# ============ Error Metrics ============
error = np.abs(pred - true)
print("Max Error:", np.max(error))
print("Mean Absolute Error:", np.mean(error))

# Compute Relative L2 Error
rel_l2_error = np.linalg.norm(pred - true)/np.linalg.norm(true)
print("Relative L2 Error:", rel_l2_error)

# ============ Error Plot ============
plt.figure()
plt.imshow(error, cmap='Reds')
plt.title("Absolute Error |u_pred - u_true|")
plt.colorbar()
plt.tight_layout()
plt.savefig("random_normal_sample/results/error_plot.png")
print("Saved error plot at: random_normal_sample/results/error_plot.png")

