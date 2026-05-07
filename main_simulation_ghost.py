import numpy as np

from core import (
    setup_mac_grid_ghost,
    run_ns_projection_mac_ghost,
    face_to_center_velocity_ghost,
)

from plots import (
    plot_final_velocity,
    plot_centerline_u, 
    plot_centerline_v,
    plot_streamlines,
    animate_velocity_field_pretty,
)

from diag import (
    get_ghia_re100_data,
    compute_ghia_errors,
)

# -------------------------------------------------
# Simulation parameters
# -------------------------------------------------
Nx = 51
Ny = 51

Re = 100.0
nu = 1.0 / Re

dt = 5e-3
nsteps = 200

# -------------------------------------------------
# Setup grid
# -------------------------------------------------
dx, dy, p, u, v, Xp, Yp, Xu, Yu, Xv, Yv = setup_mac_grid_ghost(Nx, Ny)

# -------------------------------------------------
# Run simulation
# -------------------------------------------------
history = run_ns_projection_mac_ghost(
    Nx, Ny,
    dx, dy,
    nsteps=nsteps,
    dt=dt,
    nu=nu,
    U_lid=1.0,
    steady_tol=1e-6,
    min_steps=100,
    print_every=100,
)

final = history[-1]

# -------------------------------------------------
# Convert to cell-center velocity
# -------------------------------------------------
u_c, v_c = face_to_center_velocity_ghost(
    final["u"],
    final["v"],
)
# -------------------------------------------------
# Ghia benchmark comparison
# -------------------------------------------------
ghia = get_ghia_re100_data()

errors = compute_ghia_errors(u_c, v_c, Xp, Yp, ghia)

print("\n=== Ghia Benchmark ===")
print(f"u_L2 = {errors['u_L2']:.4f}")
print(f"v_L2 = {errors['v_L2']:.4f}")
print(f"u_max = {errors['u_max']:.4f}")
print(f"v_max = {errors['v_max']:.4f}")

# -------------------------------------------------
# Diagnostics
# -------------------------------------------------
print("\n=== Final Diagnostics ===")

print(
    "max |div_new| =",
    np.max(np.abs(final["div_new"]))
)

print(
    "final velocity_change =",
    final["velocity_change"]
)

print(
    "final umax =",
    final["umax"]
)

print(
    "final vmax =",
    final["vmax"]
)

# -------------------------------------------------
# Plot
# -------------------------------------------------
plot_final_velocity(
    Xp,
    Yp,
    u_c,
    v_c,
)

i_mid = Nx // 2
j_mid = Ny // 2

plot_centerline_u(Yp[i_mid, :], u_c[i_mid, :], ghia, Re)
plot_centerline_v(Xp[:, j_mid], v_c[:, j_mid], ghia, Re)

plot_streamlines(Xp, Yp, u_c, v_c, Re)

anim = animate_velocity_field_pretty(
    history,
    Xp,
    Yp,
    face_to_center_velocity_ghost,
    skip=20,
)
#anim.save("velocity_evolution.gif", writer="pillow", fps=10)
