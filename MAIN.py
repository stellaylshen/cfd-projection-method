# Main function of staggered_grid.py
import numpy as np
from CORE import *
from TEST import *
from PLOTS import *
from DIAG import *


# =========================================================
# 4. main
# =========================================================

if __name__ == "__main__":
    Nx, Ny = 7, 7

    dx, dy, p, u, v, Xp, Yp, Xu, Yu, Xv, Yv = setup_mac_grid(Nx, Ny)

    print("=== Shapes ===")
    print("p:", p.shape)
    print("u:", u.shape)
    print("v:", v.shape)

    test_divergence(Nx, Ny, dx, dy)
    test_gradient(Nx, Ny, dx, dy)
    #closure_results = test_div_grad_closure(Nx, Ny, dx, dy)
    test_div_grad_closure(Nx, Ny, dx, dy)

    proj_results = test_projection_mac(Nx, Ny, dx, dy, dt=1.0)
    

    # u = np.zeros((Nx + 1, Ny))
    # v = np.zeros((Nx, Ny + 1))

    # u, v = apply_velocity_bc_mac(u, v, U_lid=1.0)

    # print("u top row     =", u[:, -1])
    # print("u bottom row  =", u[:, 0])
    # print("u left wall   =", u[0, :])
    # print("u right wall  =", u[-1, :])

    # print("v bottom wall =", v[:, 0])
    # print("v top wall    =", v[:, -1])
    # print("v left wall   =", v[0, :])
    # print("v right wall  =", v[-1, :])

    diff_proj_results = test_diffusion_projection_mac(
    Nx, Ny, dx, dy,
    dt=1e-1,
    nu=0.1,
    U_lid=1.0
    )


    print("\n=== u_star top diagnostics ===")
    print("u_star top row      =", diff_proj_results["u_star"][:, -1])
    print("u_star 2nd row      =", diff_proj_results["u_star"][:, -2])
    print("u_star left wall    =", diff_proj_results["u_star"][0, :])
    print("u_star right wall   =", diff_proj_results["u_star"][-1, :])

    # 確認rhs = div_star / dt 到底是不是只出現在 boundary？？？
    rhs_test = diff_proj_results["div_star"] / 1e-1

    plt.figure(figsize=(5, 4))
    plt.imshow(rhs_test.T, origin="lower")
    plt.colorbar(label="rhs")
    plt.title("rhs = div_star / dt")
    plt.show()

    # history = run_diffusion_projection_mac(
    # Nx, Ny, dx, dy,
    # nsteps=20,
    # dt=1e-3,
    # nu=0.1,
    # U_lid=1.0
    # )

    # final = history[-1]
    # -------------------------------------------------
    # Minimal advection + diffusion + projection run
    # -------------------------------------------------
    Nx_run, Ny_run = 15, 15
    dx_run, dy_run, p_run, u_run, v_run, Xp_run, Yp_run, Xu_run, Yu_run, Xv_run, Yv_run = setup_mac_grid(Nx_run, Ny_run)

    history = run_ns_projection_mac(
        Nx_run, Ny_run, dx_run, dy_run,
        nsteps=200,
        dt=5e-3,
        nu=0.2,
        U_lid=1.0
    )   

    final = history[-1]

    # plots decided here, not inside test functions
    # plot_projection_dashboard(
    #     Xp, Yp, Xu, Yu, Xv, Yv,
    #     diff_proj_results["u_star"], diff_proj_results["v_star"],
    #     diff_proj_results["u_new"], diff_proj_results["v_new"],
    #     diff_proj_results["p"],
    #     diff_proj_results["div_star"], diff_proj_results["div_new"],
    #     dx, dy, Nx, Ny
    # )

    # center velocity for quick diagnostics
    u_c = 0.5 * (final["u"][:-1, :] + final["u"][1:, :])
    v_c = 0.5 * (final["v"][:, :-1] + final["v"][:, 1:])

    print("\n=== Final flow diagnostics ===")
    print("max |u_center| overall =", np.max(np.abs(u_c)))
    print("max |v_center| overall =", np.max(np.abs(v_c)))
    print("max |u_center| top row =", np.max(np.abs(u_c[:, -1])))
    print("max |u_center| 2nd row =", np.max(np.abs(u_c[:, -2])))
    print("max |u_center| 3rd row =", np.max(np.abs(u_c[:, -3])))
    print("max |u_center| 4th row =", np.max(np.abs(u_c[:, -4])))

    # plot_projection_dashboard(
    #     Xp_run, Yp_run, Xu_run, Yu_run, Xv_run, Yv_run,
    #     final["u_star"], final["v_star"],
    #     final["u"], final["v"],
    #     final["p"],
    #     final["div_star"], final["div_new"],
    #     dx_run, dy_run, Nx_run, Ny_run
    # )

    # ----------------------------------------
    # Minimal final diagnostics only
    # ----------------------------------------

    # face -> center velocity
    u_c = 0.5 * (final["u"][:-1, :] + final["u"][1:, :])
    v_c = 0.5 * (final["v"][:, :-1] + final["v"][:, 1:])

    # 1) final velocity field
    plt.figure(figsize=(6, 5))
    plt.quiver(Xp_run, Yp_run, u_c, v_c)
    plt.title("Final velocity field")
    plt.gca().set_aspect("equal")
    plt.show()

    # 2) final div_new
    plt.figure(figsize=(6, 5))
    plt.contourf(Xp_run, Yp_run, final["div_new"], levels=20)
    plt.colorbar(label="div_new")
    plt.title("Final div_new")
    plt.gca().set_aspect("equal")
    plt.show()

    # 3) centerline u profile (vertical direction at middle x)
    i_mid = Nx_run // 2
    plt.figure(figsize=(5, 4))
    plt.plot(u_c[i_mid, :], Yp_run[i_mid, :], marker="o")
    plt.xlabel("u_center")
    plt.ylabel("y")
    plt.title("Centerline u profile")
    plt.grid(True)
    plt.show()

