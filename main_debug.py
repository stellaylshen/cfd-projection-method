# main_debug.py
# 專門看中間過程 例如看 boundary artifact、projection 到底修了什麼和沒修什麼
# 7x7 or 15x15 grid

import numpy as np
import matplotlib.pyplot as plt

from core import setup_mac_grid
from test import test_diffusion_projection_mac
from plots import plot_projection_dashboard

if __name__ == "__main__":
    # -------------------------------------------------
    # Layer 2: debug / understand numerical behavior
    # -------------------------------------------------
    Nx, Ny = 7, 7
    dt = 1e-1
    nu = 0.1
    U_lid = 1.0

    dx, dy, p, u, v, Xp, Yp, Xu, Yu, Xv, Yv = setup_mac_grid(Nx, Ny)

    results = test_diffusion_projection_mac(
        Nx, Ny, dx, dy,
        dt=dt,
        nu=nu,
        U_lid=U_lid
    )

    print("\n=== u_star top diagnostics ===")
    print("u_star top row      =", results["u_star"][:, -1])
    print("u_star 2nd row      =", results["u_star"][:, -2])
    print("u_star left wall    =", results["u_star"][0, :])
    print("u_star right wall   =", results["u_star"][-1, :])

    # rhs heatmap
    rhs = results["div_star"] / dt

    plt.figure(figsize=(5, 4))
    plt.imshow(rhs.T, origin="lower")
    plt.colorbar(label="rhs = div_star / dt")
    plt.title("rhs heatmap")
    plt.show()

    # dashboard
    plot_projection_dashboard(
        Xp, Yp, Xu, Yu, Xv, Yv,
        results["u_star"], results["v_star"],
        results["u_new"], results["v_new"],
        results["p"],
        results["div_star"], results["div_new"],
        dx, dy, Nx, Ny
    )
