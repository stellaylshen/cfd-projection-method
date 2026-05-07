import numpy as np

grid_list = [21, 31, 41, 51]

u_errors = [0.0456, 0.0285, 0.0207, 0.0164]
v_errors = [0.0175, 0.0111, 0.0084, 0.0069]

print("=== Observed convergence order ===")

for k in range(len(grid_list) - 1):
    N1 = grid_list[k]
    N2 = grid_list[k + 1]

    h1 = 1.0 / N1
    h2 = 1.0 / N2
    r = h1 / h2

    p_u = np.log(u_errors[k] / u_errors[k + 1]) / np.log(r)
    p_v = np.log(v_errors[k] / v_errors[k + 1]) / np.log(r)

    print(
        f"N={N1}->{N2}: "
        f"p_u = {p_u:.2f}, p_v = {p_v:.2f}"
    )
