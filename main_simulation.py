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
    Nx, Ny = 15, 15
    nsteps = 200
    dt = 5e-3
    nu = 0.2
    U_lid = 1.0

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

    # 1) final velocity field
    plt.figure(figsize=(6, 5))
    plt.quiver(Xp, Yp, u_c, v_c)
    plt.title("Final velocity field")
    plt.gca().set_aspect("equal")
    plt.show()

    # 2) final div_new
    plt.figure(figsize=(6, 5))
    plt.contourf(Xp, Yp, final["div_new"], levels=20)
    plt.colorbar(label="div_new")
    plt.title("Final div_new")
    plt.gca().set_aspect("equal")
    plt.show()

    # 3) centerline u profile
    i_mid = Nx // 2
    plt.figure(figsize=(5, 4))
    plt.plot(u_c[i_mid, :], Yp[i_mid, :], marker="o")
    plt.xlabel("u_center")
    plt.ylabel("y")
    plt.title("Centerline u profile")
    plt.grid(True)
    plt.show()
