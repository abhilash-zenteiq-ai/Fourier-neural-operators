
# Solving 2D Poisson Equation Using Fourier Neural Operators (FNO)

This project demonstrates solving the **2D Poisson equation** using a Tiny Fourier Neural Operator (FNO) model implemented in TensorFlow.

---

## 📘 Equation

We solve the 2D Poisson equation of the form:

$$
-\\Delta u(x, y) = f(x, y), \quad \text{for } (x, y) \in [0, 1]^2
$$

with **Dirichlet boundary conditions** and exact analytical solution:

$$
u(x, y) = \\frac{\\sin(m\\pi x) \\sin(n\\pi y)}{\\pi^2(m^2 + n^2)}
$$

---

## 🧠 Methodology

We implement a **tiny FNO (Fourier Neural Operator)** in TensorFlow that learns to map the source term \( f(x, y) \) to the solution \( u(x, y) \) using synthetic data.

---

## 📁 Directory Structure
```Fourier-neural-operators
├── fno_main_file.py # Main training + evaluation script
├── fno_dataset.py # Generates (f, u) pairs using analytical formula
├── Plots/ # Stores prediction, error and loss plots
└── README.md # This file

