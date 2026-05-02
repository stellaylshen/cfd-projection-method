# main_simulation.py
# 專門跑正式 cavity simulation 給最終結果 之後拿去接 benchmark
# 31x31 or finer grid size

import numpy as np
import matplotlib.pyplot as plt

from core import setup_mac_grid, run_ns_projection_mac

if __name__ == "__main__":
# -------------------------------------------------
# Layer 3: actual cavity simulation
# -------------------------------------------------
    Nx, Ny = 31, 31
    nsteps = 500
    dt = 5e-3
    Re = 100.0
    U_lid = 1.0
    L = 1.0
    nu = U_lid * L / Re
    print(f"Re = {Re}, nu = {nu}")

    dx, dy, p, u, v, Xp, Yp, Xu, Yu, Xv, Yv = setup_mac_grid(Nx, Ny)

    history = run_ns_projection_mac(
    Nx, Ny, dx, dy,
    nsteps=nsteps,
    dt=dt,
    nu=nu,
    U_lid=U_lid
    )

    final = history[-1]

    # face -> center velocity
    u_c = 0.5 * (final["u"][:-1, :] + final["u"][1:, :])
    v_c = 0.5 * (final["v"][:, :-1] + final["v"][:, 1:])

    print("\n=== Final flow diagnostics ===")
    print("max |u_center| overall =", np.max(np.abs(u_c)))
    print("max |v_center| overall =", np.max(np.abs(v_c)))
    print("max |u_center| top row =", np.max(np.abs(u_c[:, -1])))
    print("max |u_center| 2nd row =", np.max(np.abs(u_c[:, -2])))
    print("max |u_center| 3rd row =", np.max(np.abs(u_c[:, -3])))
    print("max |u_center| 4th row =", np.max(np.abs(u_c[:, -4])))

    # Ghia et al. (1982), Re=100
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


    # 1) final velocity field
    plt.figure(figsize=(6, 5))
    plt.quiver(Xp, Yp, u_c, v_c)
    plt.title("Final velocity field")
    plt.gca().set_aspect("equal")
    plt.show()

    # 2) final div_new
    plt.figure(figsize=(6, 5))
    plt.contourf(Xp, Yp, final["div_proj"], levels=20)
    plt.colorbar(label="div_proj")
    plt.title("Final div_proj")
    plt.gca().set_aspect("equal")
    plt.show()

    # 3) centerline u profile
    i_mid = Nx // 2

    plt.figure(figsize=(5, 4))
    plt.plot(u_c[i_mid, :], Yp[i_mid, :], marker="o", label="simulation")
    plt.plot(u_ghia, y_ghia, "s", label="Ghia et al. 1982")
    plt.xlabel("u velocity at x = 0.5")
    plt.ylabel("y")
    plt.title(f"Centerline u profile, Re={Re:.0f}")
    plt.grid(True)
    plt.legend()
    plt.show()


