# Diagnostic functions for staggered_grid.py
import numpy as np

# [DIAG]
def debug_one_cell(u, v, i, j):
    print(f"\n=== Cell ({i},{j}) flux ===")

    u_left  = u[i, j]
    u_right = u[i+1, j]
    v_bot   = v[i, j]
    v_top   = v[i, j+1]

    print(f"u_left  = {u_left}")
    print(f"u_right = {u_right}")
    print(f"v_bot   = {v_bot}")
    print(f"v_top   = {v_top}")

    print("\nFlux contribution:")
    print(f"(u_right - u_left) = {u_right - u_left}")
    print(f"(v_top   - v_bot ) = {v_top - v_bot}")

def get_ghia_re100_data():
    # Ghia et al. (1982), Re=100, u velocity along horizontal centerline x = 0.5
    y_ghia = np.array([
        1.0000, 0.9766, 0.9688, 0.9609, 0.9531,
        0.8516, 0.7344, 0.6172, 0.5000,
        0.4531, 0.2813, 0.1719, 0.1016,
        0.0703, 0.0625, 0.0547, 0.0000
    ])
    u_ghia = np.array([
        1.0000, 0.84123, 0.78871, 0.73722, 0.68717,
        0.23151, 0.00332, -0.13641, -0.20581,
        -0.21090, -0.15662, -0.10150, -0.06434,
        -0.04775, -0.04192, -0.03717, 0.00000
    ])

    # Ghia et al. (1982), Re=100, v velocity along horizontal centerline y = 0.5
    x_ghia = np.array([
        1.0000, 0.9688, 0.9609, 0.9531, 0.9453,
        0.9063, 0.8594, 0.8047, 0.5000,
        0.2344, 0.2266, 0.1563, 0.0938,
        0.0781, 0.0703, 0.0625, 0.0000
    ])

    v_ghia = np.array([
        0.00000, -0.05906, -0.07391, -0.08864, -0.10313,
        -0.16914, -0.22445, -0.24533, 0.05454,
        0.17527, 0.17507, 0.16077, 0.12317,
        0.10890, 0.10091, 0.09233, 0.00000
    ])

    return {
        "y": y_ghia,
        "u": u_ghia,
        "x": x_ghia,
        "v": v_ghia,
    }

def compute_ghia_errors(u_c, v_c, Xp, Yp, ghia):
    Nx, Ny = u_c.shape
    i_mid = Nx // 2
    j_mid = Ny // 2

    u_sim = np.interp(ghia["y"], Yp[i_mid, :], u_c[i_mid, :])
    v_sim = np.interp(ghia["x"], Xp[:, j_mid], v_c[:, j_mid])

    u_err = u_sim - ghia["u"]
    v_err = v_sim - ghia["v"]

    return {
        "u_max": np.max(np.abs(u_err)),
        "u_L2": np.sqrt(np.mean(u_err**2)),
        "v_max": np.max(np.abs(v_err)),
        "v_L2": np.sqrt(np.mean(v_err**2)),
    }


