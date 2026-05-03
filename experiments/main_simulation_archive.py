# main_simulation.py
# 專門跑正式 cavity simulation 給最終結果 之後拿去接 benchmark
# 31x31 or finer grid size

import numpy as np
import matplotlib.pyplot as plt

from core import setup_mac_grid, face_to_center_velocity, get_cell_center_coordinates, run_ns_projection_mac

def compute_centerline_errors(Nx, Ny, nsteps=300, dt=5e-3, Re=100.0):
    dx = 1.0 / Nx
    dy = 1.0 / Ny
    nu = 1.0 / Re

    history = run_ns_projection_mac(Nx, Ny, dx, dy, nsteps=nsteps, dt=dt, nu=nu)
    final = history[-1]

    u = final["u"]
    v = final["v"]

    # 轉 center velocity
    u_c, v_c = face_to_center_velocity(u, v)

    Xp, Yp = get_cell_center_coordinates(Nx, Ny)

    i_mid = Nx // 2
    j_mid = Ny // 2

    # ---- Ghia data ----

    # u error
    u_sim = np.interp(y_ghia, Yp[i_mid, :], u_c[i_mid, :])
    u_err = u_sim - u_ghia
    u_L2 = np.sqrt(np.mean(u_err**2))

    # v error
    v_sim = np.interp(x_ghia, Xp[:, j_mid], v_c[:, j_mid])
    v_err = v_sim - v_ghia
    v_L2 = np.sqrt(np.mean(v_err**2))

    return u_L2, v_L2

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

    # 4) centerline v profile
    j_mid = Ny // 2

    plt.figure(figsize=(5, 4))
    plt.plot(Xp[:, j_mid], v_c[:, j_mid], marker="o", label="simulation")
    plt.plot(x_ghia, v_ghia, "s", label="Ghia et al. 1982")
    plt.xlabel("x")
    plt.ylabel("v velocity at y = 0.5")
    plt.title("Centerline v profile, Re=100")
    plt.grid(True)
    plt.legend()
    plt.show()


    # -------------------------------------------------
    # Quantitative benchmark error
    # -------------------------------------------------

    # u-profile error: compare simulation u(x=0.5, y) at Ghia y locations
    u_sim_at_ghia = np.interp(y_ghia, Yp[i_mid, :], u_c[i_mid, :])
    u_err = u_sim_at_ghia - u_ghia

    print("\n=== Ghia benchmark error: u centerline ===")
    print("max abs error =", np.max(np.abs(u_err)))
    print("L2 error      =", np.sqrt(np.mean(u_err**2)))

    # v-profile error: compare simulation v(x, y=0.5) at Ghia x locations
    v_sim_at_ghia = np.interp(x_ghia, Xp[:, j_mid], v_c[:, j_mid])
    v_err = v_sim_at_ghia - v_ghia

    print("\n=== Ghia benchmark error: v centerline ===")
    print("max abs error =", np.max(np.abs(v_err)))
    print("L2 error      =", np.sqrt(np.mean(v_err**2)))


    # grid_refinement
    grid_list = [21, 31, 41, 51]

    u_errors = []
    v_errors = []

    for N in grid_list:
        print(f"\nRunning N = {N} ...")
        u_L2, v_L2 = compute_centerline_errors(N, N)

        print(f"u_L2 = {u_L2:.4f}, v_L2 = {v_L2:.4f}")

        u_errors.append(u_L2)
        v_errors.append(v_L2)

    plt.figure(figsize=(5,4))
    plt.plot(grid_list, u_errors, 'o-', label="u centerline L2 error")
    plt.plot(grid_list, v_errors, 's-', label="v centerline L2 error")

    plt.xlabel("Grid size (N)")
    plt.ylabel("L2 error")
    plt.title("Grid refinement study (Re=100)")
    plt.grid(True)
    plt.legend()
    plt.show()

