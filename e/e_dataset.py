# from dolfin import *
# import numpy as np
# import os
# import matplotlib.pyplot as plt

# np.random.seed(42)

# # Output directory
# output_dir = "Outputs/e_main_output"
# os.makedirs(output_dir, exist_ok=True)

# def random_e_expression():
#     """Random spatially varying coefficient e(x,y)."""
#     a = np.random.uniform(1.0, 3.0)
#     b = np.random.uniform(1.0, 3.0)
#     return Expression("0.5 + 0.5*sin(a*pi*x[0])*cos(b*pi*x[1])", degree=3, a=a, b=b)

# def f_expression():
#     return Expression("sin(pi*x[0])*sin(pi*x[1])", degree=3)

# def solve_pde(N=64, n_samples=100):
#     E_list, F_list, U_list = [], [], []

#     mesh = UnitSquareMesh(N, N)
#     V = FunctionSpace(mesh, "P", 1)

#     for i in range(n_samples):
#         e_expr = random_e_expression()
#         f_expr = f_expression()

#         e = interpolate(e_expr, V)
#         f = interpolate(f_expr, V)

#         u = TrialFunction(V)
#         v = TestFunction(V)

#         a = dot(e * grad(u), grad(v)) * dx
#         L = f * v * dx

#         bc = DirichletBC(V, Constant(0.0), "on_boundary")

#         u_sol = Function(V)
#         solve(a == L, u_sol, bc)

#         coords = V.tabulate_dof_coordinates().reshape((N + 1, N + 1, 2))
#         u_vals = u_sol.compute_vertex_values(mesh).reshape(N + 1, N + 1).T
#         e_vals = e.compute_vertex_values(mesh).reshape(N + 1, N + 1).T
#         f_vals = f.compute_vertex_values(mesh).reshape(N + 1, N + 1).T

#         # Resize to 64x64
#         u_vals = u_vals[:64, :64]
#         e_vals = e_vals[:64, :64]
#         f_vals = f_vals[:64, :64]

#         E_list.append(e_vals)
#         F_list.append(f_vals)
#         U_list.append(u_vals)

#         print(f"[INFO] Sample {i + 1}/{n_samples} complete")

#     E = np.expand_dims(np.array(E_list), axis=-1)
#     F = np.expand_dims(np.array(F_list), axis=-1)
#     U = np.expand_dims(np.array(U_list), axis=-1)

#     np.save(os.path.join(output_dir, "e_field.npy"), E)
#     np.save(os.path.join(output_dir, "f_field.npy"), F)
#     np.save(os.path.join(output_dir, "u_field.npy"), U)

#     # Example plot
#     plt.figure(figsize=(12, 4))
#     plt.subplot(1, 3, 1)
#     plt.title("e(x,y)")
#     plt.imshow(E[0, ..., 0], cmap='plasma')
#     plt.colorbar()

#     plt.subplot(1, 3, 2)
#     plt.title("f(x,y)")
#     plt.imshow(F[0, ..., 0], cmap='viridis')
#     plt.colorbar()

#     plt.subplot(1, 3, 3)
#     plt.title("u(x,y)")
#     plt.imshow(U[0, ..., 0], cmap='inferno')
#     plt.colorbar()

#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, "example_plot.png"))
#     print("[INFO] Sample plot saved to:", os.path.join(output_dir, "example_plot.png"))

# if __name__ == "__main__":
#     solve_pde(N=64, n_samples=500)
