# Copyright (c) 2024 Zenteiq Aitech Innovations Private Limited and
# AiREX Lab, Indian Institute of Science, Bangalore.
# All rights reserved.
#
# This file is part of SciREX
# (Scientific Research and Engineering eXcellence Platform),
# developed jointly by Zenteiq Aitech Innovations and AiREX Lab
# under the guidance of Prof. Sashikumaar Ganesan.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# For any clarifications or special considerations,
# please contact: contact@scirex.org

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import os

from scirex.core.sciml.fno.models.fno_2d import FNO2d


def generate_poisson_data(n_samples=1200, nx=64, ny=64, m_max=5, n_max=5, key=jax.random.PRNGKey(0)):
    """Generate Poisson data with f(x,y) = sin(mπx)sin(nπy), and analytical solution"""
    x = jnp.linspace(0, 1, nx)
    y = jnp.linspace(0, 1, ny)
    X, Y = jnp.meshgrid(x, y)

    key1, key2 = jax.random.split(key)
    m_vals = jax.random.randint(key1, shape=(n_samples,), minval=1, maxval=m_max + 1)
    n_vals = jax.random.randint(key2, shape=(n_samples,), minval=1, maxval=n_max + 1)

    f_list, u_list = [], []

    for i in range(n_samples):
        m, n = m_vals[i], n_vals[i]
        f = jnp.sin(m * jnp.pi * X) * jnp.sin(n * jnp.pi * Y)
        denom = (jnp.pi ** 2) * (m ** 2 + n ** 2)
        u = f / denom

        f_list.append(f)
        u_list.append(u)

    f_stack = jnp.stack(f_list)
    u_stack = jnp.stack(u_list)
    mesh_x = jnp.broadcast_to(X, f_stack.shape)
    mesh_y = jnp.broadcast_to(Y, f_stack.shape)

    input_data = jnp.stack([f_stack, mesh_x, mesh_y], axis=1)  # (N, 3, nx, ny)
    output_data = u_stack[:, jnp.newaxis, :, :]                # (N, 1, nx, ny)

    return input_data, output_data, x, y, m_vals, n_vals


# Generate data
input_data, output_data, x, y, m_vals, n_vals = generate_poisson_data()

# Train/test split
train_x, test_x = input_data[:1000], input_data[1000:]
train_y, test_y = output_data[:1000], output_data[1000:]

# Model setup
model = FNO2d(
    in_channels=3,    # f(x,y), x, y
    out_channels=1,   # u(x,y)
    modes1=12,
    modes2=12,
    width=64,
    activation=jax.nn.gelu,
    n_blocks=4,
    key=jax.random.PRNGKey(0),
)

# Optimizer
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))


@eqx.filter_jit
def make_step(model, opt_state, batch):
    def loss_fn(model):
        pred = jax.vmap(model)(batch[0])
        return jnp.mean((pred - batch[1]) ** 2)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


# Training loop
batch_size = 50
n_epochs = 50
losses = []

for epoch in range(n_epochs):
    for i in range(0, len(train_x), batch_size):
        batch = (train_x[i:i+batch_size], train_y[i:i+batch_size])
        loss, model, opt_state = make_step(model, opt_state, batch)
        losses.append(loss)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

# Evaluation
test_pred = jax.vmap(model)(test_x)
test_error = jnp.mean((test_pred - test_y) ** 2)
print(f"Test MSE: {test_error:.6f}")

# Save outputs
output_dir = os.path.join(os.path.dirname(__file__), "outputs", "poisson_sine")
os.makedirs(output_dir, exist_ok=True)

# Plot predictions
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.imshow(test_x[0, 0], cmap="viridis")
plt.colorbar()
plt.title("Source $f(x,y) = \\sin(m\\pi x) \\sin(n\\pi y)$")

plt.subplot(132)
plt.imshow(test_y[0, 0], cmap="viridis")
plt.colorbar()
plt.title("True solution $u(x,y)$")

plt.subplot(133)
plt.imshow(test_pred[0, 0], cmap="viridis")
plt.colorbar()
plt.title("FNO prediction $\\hat{u}(x,y)$")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "prediction.png"))

# Plot training loss
plt.figure()
plt.semilogy(losses)
plt.title("Training Loss")
plt.xlabel("Step")
plt.ylabel("MSE")
plt.savefig(os.path.join(output_dir, "training_loss.png"))

# Plot absolute error
plt.figure()
plt.imshow(jnp.abs(test_pred[0, 0] - test_y[0, 0]), cmap="inferno")
plt.colorbar()
plt.title("Absolute Error $|u - \\hat{u}|$")
plt.savefig(os.path.join(output_dir, "absolute_error.png"))
