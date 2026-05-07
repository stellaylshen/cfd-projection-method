import numpy as np

from core import (
    setup_mac_grid_ghost,
    face_to_center_velocity_ghost,
    run_ns_projection_mac_ghost,
)
from diag import get_ghia_re100_data, compute_ghia_errors
from plots import (
    plot_grid_convergence, 
    plot_grid_convergence_loglog, 
)

def run_case(N, nsteps=2000, dt=5e-3, Re=100.0):
    dx, dy, _, _, _, Xp, Yp, _, _, _, _ = setup_mac_grid_ghost(N, N)
    nu = 1.0 / Re

    history = run_ns_projection_mac_ghost(
        N, N, dx, dy,
        nsteps=nsteps,
        dt=dt,
        nu=nu,
        U_lid=1.0,
        steady_tol=1e-5,
        min_steps=100,
        print_every=100,
    )

    final = history[-1]
    u_c, v_c = face_to_center_velocity_ghost(final["u"], final["v"])

    ghia = get_ghia_re100_data()
    errors = compute_ghia_errors(u_c, v_c, Xp, Yp, ghia)

    return errors, history

if __name__ == "__main__":
    grid_list = [21, 31, 41, 51]

    u_errors = []
    v_errors = []

    results_table = []

    for N in grid_list:
        nsteps = 2000 if N <= 31 else 3000
        print(f"\nRunning GHOST N = {N}, nsteps = {nsteps} ...")
        errors, history = run_case(N, nsteps=nsteps)

        final = history[-1]

        results_table.append({
        "N": N,
        "u_L2": errors["u_L2"],
        "v_L2": errors["v_L2"],
        "u_max": errors["u_max"],
        "v_max": errors["v_max"],
        "max_div": np.max(np.abs(final["div_new"])),
        "vel_change": final["velocity_change"],
        "steps": len(history),
        })

        print(f"u_L2 = {errors['u_L2']:.4f}, v_L2 = {errors['v_L2']:.4f}")
        u_errors.append(errors["u_L2"])
        v_errors.append(errors["v_L2"])

    print("\n=== Diagnostics Table ===")
    for row in results_table:
        print(
            f"N={row['N']:2d} | "
            f"u_L2={row['u_L2']:.4f} | "
            f"v_L2={row['v_L2']:.4f} | "
            f"u_max={row['u_max']:.4f} | "
            f"v_max={row['v_max']:.4f} | "
            f"max_div={row['max_div']:.2e} | "
            f"vel_change={row['vel_change']:.2e} | "
            f"steps={row['steps']}"
        )

    plot_grid_convergence(grid_list, u_errors, v_errors)

    plot_grid_convergence_loglog(
    grid_list,
    u_errors,
    v_errors,
    )
