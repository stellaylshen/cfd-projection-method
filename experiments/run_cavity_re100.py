import numpy as np

from core import setup_mac_grid, face_to_center_velocity, run_ns_projection_mac
from diag import get_ghia_re100_data, compute_ghia_errors
from plots import (
    plot_final_velocity,
    plot_divergence,
    plot_centerline_u,
    plot_centerline_v,
)

if __name__ == "__main__":
    Nx, Ny = 31, 31
    nsteps = 300
    dt = 5e-3
    Re = 100.0
    U_lid = 1.0
    nu = U_lid / Re

    print(f"Re = {Re}, nu = {nu}")

    dx, dy, _, _, _, Xp, Yp, _, _, _, _ = setup_mac_grid(Nx, Ny)

    history = run_ns_projection_mac(
        Nx, Ny, dx, dy,
        nsteps=nsteps,
        dt=dt,
        nu=nu,
        U_lid=U_lid,
    )

    final = history[-1]
    u_c, v_c = face_to_center_velocity(final["u"], final["v"])

    print("\n=== Final flow diagnostics ===")
    print("max |u_center| overall =", np.max(np.abs(u_c)))
    print("max |v_center| overall =", np.max(np.abs(v_c)))
    print("final CFL =", final["cfl"])
    print("max |div_new| =", np.max(np.abs(final["div_new"])))

    ghia = get_ghia_re100_data()
    errors = compute_ghia_errors(u_c, v_c, Xp, Yp, ghia)

    print("\n=== Ghia benchmark error ===")
    print(f"u: max={errors['u_max']:.4f}, L2={errors['u_L2']:.4f}")
    print(f"v: max={errors['v_max']:.4f}, L2={errors['v_L2']:.4f}")

    i_mid = Nx // 2
    j_mid = Ny // 2

    plot_final_velocity(Xp, Yp, u_c, v_c)
    plot_divergence(Xp, Yp, final["div_proj"], title="Final div_proj")
    plot_centerline_u(Yp[i_mid, :], u_c[i_mid, :], ghia, Re)
    plot_centerline_v(Xp[:, j_mid], v_c[:, j_mid], ghia, Re)